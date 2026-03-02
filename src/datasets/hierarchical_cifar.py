"""
src/datasets/hierarchical_cifar.py

CIFAR-10/100 with hierarchical label structure for multi-scale evaluation.
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset


class HierarchicalCIFAR(Dataset):
    """
    CIFAR with coarse+fine hierarchical labels.
    Returns (image, fine_label, coarse_label) triples.
    """
    COARSE_LABELS = {
        0: 0, 1: 0, 8: 0, 9: 0,   # vehicles
        2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # animals
    }

    def __init__(self, root: str = "data/", train: bool = True, seed: int = 42) -> None:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        self.base = CIFAR10(root, train=train, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,)*3, (0.5,)*3),
                            ]))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, fine_label = self.base[idx]
        coarse = self.COARSE_LABELS.get(fine_label, 0)
        return x.view(-1), fine_label, coarse
 