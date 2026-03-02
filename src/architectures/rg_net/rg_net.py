"""
src/architectures/rg_net/rg_net.py

RG-Net architecture variants.

Variants:
  RGNetShallow    — L=2-5   layers (baseline)
  RGNetStandard   — L=10-50 layers (main experiments)
  RGNetDeep       — L=50-200 layers (scaling study)
  RGNetUltraDeep  — L=200-1000 layers (extreme depth, gradient checkpointing)
  RGNetVariableWidth — width schedule adapts to ξ_data
  RGNetMultiScale — parallel streams at multiple resolutions
  RGNetResidual   — skip connections every D_skip layers
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class RGLayer(nn.Module):
    """Single RG transformation layer: h^(k) = σ(W_k h^(k-1) + b_k)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "tanh",
        sigma_w: float = 1.0,
        sigma_b: float = 0.05,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act    = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            activation, nn.Tanh()
        )
        # Critical initialisation
        nn.init.normal_(self.linear.weight, std=sigma_w / math.sqrt(in_features))
        nn.init.normal_(self.linear.bias,   std=sigma_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


# ---------------------------------------------------------------------------
# Variant 1: Shallow
# ---------------------------------------------------------------------------

class RGNetShallow(nn.Module):
    """RG-Net with L=2-5 layers. Used as baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 3,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        assert 2 <= depth <= 5, f"RGNetShallow expects depth 2-5, got {depth}"
        layers = [RGLayer(input_dim, hidden_dim, activation)]
        for _ in range(depth - 2):
            layers.append(RGLayer(hidden_dim, hidden_dim, activation))
        self.layers = nn.ModuleList(layers)
        self.head   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Variant 2: Standard
# ---------------------------------------------------------------------------

class RGNetStandard(nn.Module):
    """RG-Net with L=10-50 layers. Main experiment model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 20,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.embed  = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            RGLayer(hidden_dim, hidden_dim, activation)
            for _ in range(depth)
        ])
        self.head   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.embed(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Variant 3: Deep
# ---------------------------------------------------------------------------

class RGNetDeep(nn.Module):
    """RG-Net with L=50-200 layers. Scaling study model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 100,
        activation: str = "tanh",
        skip_interval: int = 10,
    ) -> None:
        super().__init__()
        self.embed         = nn.Linear(input_dim, hidden_dim)
        self.skip_interval = skip_interval
        self.layers        = nn.ModuleList([
            RGLayer(hidden_dim, hidden_dim, activation) for _ in range(depth)
        ])
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.embed(x))
        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (i + 1) % self.skip_interval == 0:
                x = x + residual
                residual = x
        return self.head(x)


# ---------------------------------------------------------------------------
# Variant 4: UltraDeep (gradient checkpointing)
# ---------------------------------------------------------------------------

class RGNetUltraDeep(nn.Module):
    """
    RG-Net with L=200-1000 layers. Uses gradient checkpointing to reduce
    activation memory by ~60% at cost of ~25% extra compute.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 500,
        activation: str = "tanh",
        checkpoint_segments: int = 10,
    ) -> None:
        super().__init__()
        self.embed      = nn.Linear(input_dim, hidden_dim)
        self.layers     = nn.ModuleList([
            RGLayer(hidden_dim, hidden_dim, activation) for _ in range(depth)
        ])
        self.head       = nn.Linear(hidden_dim, output_dim)
        self.segments   = checkpoint_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.embed(x))
        functions = list(self.layers)
        seg_size  = max(1, len(functions) // self.segments)
        for i in range(0, len(functions), seg_size):
            seg = nn.Sequential(*functions[i: i + seg_size])
            if self.training:
                x = cp.checkpoint(seg, x, use_reentrant=False)
            else:
                x = seg(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Variant 5: VariableWidth
# ---------------------------------------------------------------------------

class RGNetVariableWidth(nn.Module):
    """
    RG-Net with width schedule that mirrors the correlation-length decay.
    Width shrinks geometrically from max_width to min_width over depth.
    """

    def __init__(
        self,
        input_dim: int,
        max_width: int,
        min_width: int,
        output_dim: int,
        depth: int = 20,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        widths = [
            max(min_width, int(max_width * (min_width / max_width) ** (k / max(depth - 1, 1))))
            for k in range(depth)
        ]
        dims = [input_dim] + widths

        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(RGLayer(dims[i], dims[i + 1], activation))
        self.layers = nn.ModuleList(layers)
        self.head   = nn.Linear(widths[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Variant 6: MultiScale
# ---------------------------------------------------------------------------

class RGNetMultiScale(nn.Module):
    """
    RG-Net with parallel processing streams at different scale depths.
    Combines representations from depth L/4, L/2, 3L/4, L before head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 40,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RGLayer(hidden_dim if i > 0 else input_dim, hidden_dim, activation)
            for i in range(depth)
        ])
        self.checkpoints = {depth // 4, depth // 2, 3 * depth // 4, depth - 1}
        self.head = nn.Linear(hidden_dim * len(self.checkpoints), output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        collected = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.checkpoints:
                collected.append(x)
        return self.head(torch.cat(collected, dim=-1))


# ---------------------------------------------------------------------------
# Variant 7: Residual
# ---------------------------------------------------------------------------

class RGNetResidual(nn.Module):
    """RG-Net with residual connections every D_skip layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 50,
        activation: str = "tanh",
        skip_interval: int = 5,
    ) -> None:
        super().__init__()
        self.embed    = nn.Linear(input_dim, hidden_dim)
        self.layers   = nn.ModuleList([
            RGLayer(hidden_dim, hidden_dim, activation) for _ in range(depth)
        ])
        self.skip_int = skip_interval
        self.head     = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.embed(x))
        skip = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (i + 1) % self.skip_int == 0:
                x = x + skip
                skip = x
        return self.head(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VARIANTS = {
    "shallow":       RGNetShallow,
    "standard":      RGNetStandard,
    "deep":          RGNetDeep,
    "ultra_deep":    RGNetUltraDeep,
    "variable_width": RGNetVariableWidth,
    "multiscale":    RGNetMultiScale,
    "residual":      RGNetResidual,
}


def build_rg_net(variant: str, **kwargs) -> nn.Module:
    """Factory function for RG-Net variants."""
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(_VARIANTS)}")
    return _VARIANTS[variant](**kwargs)
 