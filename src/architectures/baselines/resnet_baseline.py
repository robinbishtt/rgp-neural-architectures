from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
class ResNetBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))
class ResNetBaseline(nn.Module):
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 100, width: int = 512) -> None:
        super().__init__()
        self.proj   = nn.Linear(in_features, width)
        self.blocks = nn.ModuleList([ResNetBlock(width) for _ in range(depth)])
        self.head   = nn.Linear(width, n_classes)
    def forward(self, x):
        h = F.relu(self.proj(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)