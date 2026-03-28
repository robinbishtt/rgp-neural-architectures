from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from src.architectures.rg_net.rg_net import RGLayer


class RGNetUltraDeep(nn.Module):
    """Ultra-deep RG-Net (depth >= 500) with gradient checkpointing.

    Accepts both ``input_dim`` and ``in_features``, and ``hidden_dim`` / ``width``.
    """

    def __init__(
        self,
        input_dim: int = None,
        n_classes: int = 10,
        output_dim: int = None,
        depth: int = 1000,
        hidden_dim: int = 1024,
        activation: str = "tanh",
        sigma_w: float = 1.4,
        sigma_b: float = 0.05,
        checkpoint_segments: int = 10,
        use_gradient_checkpointing: bool = True,
        # Backward-compatible aliases
        in_features: int = None,
        width: int = None,
    ) -> None:
        super().__init__()
        if input_dim is None:
            input_dim = in_features if in_features is not None else 784
        if width is not None:
            hidden_dim = width
        if output_dim is not None:
            n_classes = output_dim
        if depth < 500:
            raise ValueError(f"RGNetUltraDeep expects depth >= 500, got {depth}")

        self.use_ckpt = use_gradient_checkpointing
        self.segments = checkpoint_segments

        self.embed = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.embed.weight, std=sigma_w / math.sqrt(input_dim))
        nn.init.normal_(self.embed.bias, std=sigma_b)

        self.layers = nn.ModuleList([
            RGLayer(hidden_dim, hidden_dim, activation, sigma_w, sigma_b)
            for _ in range(depth)
        ])

        self.head = nn.Linear(hidden_dim, n_classes)
        nn.init.normal_(self.head.weight, std=1.0 / math.sqrt(hidden_dim))
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.embed(x))
        functions = list(self.layers)
        seg_size = max(1, len(functions) // self.segments)
        for i in range(0, len(functions), seg_size):
            seg = nn.Sequential(*functions[i: i + seg_size])
            if self.use_ckpt and self.training:
                x = cp.checkpoint(seg, x, use_reentrant=False)
            else:
                x = seg(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
