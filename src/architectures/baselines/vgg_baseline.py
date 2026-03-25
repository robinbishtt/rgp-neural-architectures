from __future__ import annotations
import torch.nn as nn
class VGGBaseline(nn.Module):
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 n_blocks: int = 5, block_depth: int = 2, width: int = 512) -> None:
        super().__init__()
        blocks = [nn.Linear(in_features, width), nn.BatchNorm1d(width), nn.ReLU()]
        for _ in range(n_blocks):
            for _ in range(block_depth):
                blocks += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
            blocks.append(nn.Dropout(0.3))
        blocks.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        return self.net(x)