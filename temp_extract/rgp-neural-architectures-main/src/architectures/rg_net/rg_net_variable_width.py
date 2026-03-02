"""
src/architectures/rg_net/rg_net_variable_width.py

RGNetVariableWidth — non-uniform width for information bottleneck studies.
"""
from __future__ import annotations
from typing import List
import torch.nn as nn
from src.rg_flow.operators.operators import StandardRGOperator


class RGNetVariableWidth(nn.Module):
    """
    RG-Net with non-uniform width schedule through depth.

    Supports: hourglass (wide-narrow-wide), funnel (decreasing),
    pyramid (increasing), and custom width schedules.
    """

    def __init__(
        self,
        in_features: int,
        width_schedule: List[int],
        n_classes: int = 10,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, width_schedule[0])
        layers = []
        for i in range(len(width_schedule) - 1):
            layers.append(StandardRGOperator(width_schedule[i], width_schedule[i + 1], activation))
        self.layers    = nn.ModuleList(layers)
        self.head      = nn.Linear(width_schedule[-1], n_classes)

    def forward(self, x):
        import torch.nn.functional as F
        h = F.tanh(self.input_proj(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

    @classmethod
    def hourglass(cls, in_features: int, n_classes: int, max_width: int = 512,
                  min_width: int = 64, total_depth: int = 40) -> "RGNetVariableWidth":
        """Hourglass schedule: wide -> narrow -> wide."""
        half = total_depth // 2
        down = [int(max_width - (max_width - min_width) * i / half) for i in range(half)]
        up   = [int(min_width + (max_width - min_width) * i / half) for i in range(half)]
        return cls(in_features, down + up, n_classes)
