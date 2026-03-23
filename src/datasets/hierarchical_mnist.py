"""
src/datasets/hierarchical_mnist.py

MNIST augmented with controlled correlation structure.
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset


class HierarchicalMNIST(Dataset):
    """
    MNIST with added multi-scale correlation structure.
    Adds Gaussian spatial correlations at configurable length scales.
    """

    def __init__(
        self,
        root: str = "data/",
        train: bool = True,
        correlation_length: float = 5.0,
        seed: int = 42,
    ) -> None:
        from torchvision.datasets import MNIST
        from torchvision import transforms
        self.base = MNIST(root, train=train, download=True,
                          transform=transforms.ToTensor())
        self.xi   = correlation_length
        self.rng  = torch.Generator().manual_seed(seed)

    def _add_correlations(self, x: torch.Tensor) -> torch.Tensor:
        """Add spatially correlated noise at scale xi."""
        noise = torch.randn(x.shape, generator=self.rng) * 0.1
        kernel_size = max(3, int(self.xi) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        import torch.nn.functional as F
        k1d = torch.exp(-torch.arange(kernel_size).float() ** 2 / (2 * self.xi ** 2))
        k1d /= k1d.sum()
        kernel = k1d.outer(k1d).unsqueeze(0).unsqueeze(0)
        corr_noise = F.conv2d(noise.unsqueeze(0), kernel, padding=kernel_size // 2)[0]
        return x + corr_noise

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return self._add_correlations(x).view(-1), y
 