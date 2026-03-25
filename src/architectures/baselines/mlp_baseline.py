from __future__ import annotations
import torch.nn as nn
class MLPBaseline(nn.Module):
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 5, width: int = 512) -> None:
        super().__init__()
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