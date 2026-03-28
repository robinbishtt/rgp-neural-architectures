from __future__ import annotations
import math
import torch
import torch.nn as nn
from src.architectures.rg_net.rg_net import RGLayer


class RGNetShallow(nn.Module):
    """Shallow RG-Net with depth in [10, 50].

    Accepts both ``input_dim`` and ``in_features`` (alias), and both
    ``hidden_dim`` / ``width`` for the hidden width.
    """

    def __init__(
        self,
        input_dim: int = None,
        n_classes: int = 10,
        depth: int = 20,
        hidden_dim: int = 128,
        activation: str = "tanh",
        sigma_w: float = 1.4,
        sigma_b: float = 0.05,
        # Backward-compatible aliases
        in_features: int = None,
        width: int = None,
    ) -> None:
        super().__init__()
        # Resolve aliases
        if input_dim is None:
            input_dim = in_features if in_features is not None else 784
        if width is not None:
            hidden_dim = width

        if not (10 <= depth <= 50):
            raise ValueError(f"RGNetShallow expects depth 10-50, got {depth}")

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
        for layer in self.layers:
            x = layer(x)
        return self.head(x)