from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
class RGLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "tanh",
        sigma_w: float = 1.4,
        sigma_b: float = 0.3,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act    = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            activation, nn.Tanh()
        )
        nn.init.normal_(self.linear.weight, std=sigma_w / math.sqrt(in_features))
        nn.init.normal_(self.linear.bias,   std=sigma_b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
class RGNetShallow(nn.Module):
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
class RGNetStandard(nn.Module):
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        output_dim: int = None,
        depth: int = 20,
        activation: str = "tanh",
        # Backward-compatible aliases
        in_features: int = None,
        n_classes: int = None,
        width: int = None,
    ) -> None:
        super().__init__()
        # Resolve aliases
        if input_dim is None:
            input_dim = in_features if in_features is not None else 784
        if hidden_dim is None:
            hidden_dim = width if width is not None else 512
        if output_dim is None:
            output_dim = n_classes if n_classes is not None else 10
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
class RGNetDeep(nn.Module):
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
class RGNetUltraDeep(nn.Module):
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
class RGNetVariableWidth(nn.Module):
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
class RGNetMultiScale(nn.Module):
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
class RGNetResidual(nn.Module):
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
_VARIANTS = {
    'shallow':        RGNetShallow,
    'standard':       RGNetStandard,
    'deep':           RGNetDeep,
    'ultra_deep':     RGNetUltraDeep,
    'variable_width': RGNetVariableWidth,
    'multiscale':     RGNetMultiScale,
    'residual':       RGNetResidual,
}
def build_rg_net(variant: str, **kwargs) -> nn.Module:
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(_VARIANTS)}")
    return _VARIANTS[variant](**kwargs)