from __future__ import annotations
import torch.nn as nn
class MLPBaseline(nn.Module):
    def __init__(self, in_features: int = None, n_classes: int = 10,
                 depth: int = 5, width: int = 512,
                 # Backward-compatible alias
                 input_dim: int = None) -> None:
        super().__init__()
        if in_features is None:
            in_features = input_dim if input_dim is not None else 784
        layers = [nn.Linear(in_features, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, x):
        return self.net(x)