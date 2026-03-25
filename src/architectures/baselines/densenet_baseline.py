from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
class DenseNetBaseline(nn.Module):
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 20, growth_rate: int = 32) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, growth_rate)
        self.layers     = nn.ModuleList()
        in_ch = growth_rate
        for _ in range(depth):
            self.layers.append(nn.Linear(in_ch, growth_rate))
            in_ch += growth_rate
        self.head = nn.Linear(in_ch, n_classes)
    def forward(self, x):
        h = F.relu(self.input_proj(x))
        features = [h]
        for layer in self.layers:
            h_new = F.relu(layer(torch.cat(features, dim=-1)))
            features.append(h_new)
        return self.head(torch.cat(features, dim=-1))