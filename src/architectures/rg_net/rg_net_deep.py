from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from src.architectures.rg_net.rg_net import RGLayer


class RGNetDeep(nn.Module):
    """Deep RG-Net with skip connections and gradient checkpointing.

    Accepts both ``input_dim`` and ``in_features``, and ``hidden_dim`` / ``width``.
    """

    def __init__(
        self,
        input_dim: int = None,
        n_classes: int = 10,
        output_dim: int = None,
        depth: int = 500,
        hidden_dim: int = 512,
        activation: str = "tanh",
        sigma_w: float = 1.4,
        sigma_b: float = 0.05,
        skip_interval: int = 10,
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

        self.use_ckpt = use_gradient_checkpointing
        self.skip_interval = skip_interval

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
        residual = x
        for i, layer in enumerate(self.layers):
            if self.use_ckpt and self.training:
                x = cp.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if (i + 1) % self.skip_interval == 0:
                x = x + residual
                residual = x
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
