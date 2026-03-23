"""
src/datasets/synthetic_hierarchy.py

Fully synthetic hierarchical dataset with exact correlation structure control.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticHierarchy(Dataset):
    """
    Synthetic multi-scale dataset where correlation length is exactly controlled.

    Generates samples x = Σ_j a_j φ_j(scale_j) where φ_j are scale-j basis
    functions and a_j ~ N(0, σ_j²). The label is determined by the dominant scale.
    """

    def __init__(
        self,
        n_samples:     int   = 10000,
        n_features:    int   = 784,
        n_scales:      int   = 4,
        correlation_length: float = 10.0,
        n_classes:     int   = 10,
        seed:          int   = 42,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.data, self.labels = self._generate(
            n_samples, n_features, n_scales, correlation_length, n_classes
        )

    def _generate(self, n, d, n_scales, xi, n_classes) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.zeros((n, d))
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            dominant = self.rng.integers(n_scales)
            for s in range(n_scales):
                scale_len = max(1, int(xi * (2.0 ** (s - n_scales // 2))))
                freq      = d // (scale_len * 2)
                amplitude = 1.0 if s == dominant else 0.3
                coef      = self.rng.standard_normal(max(1, freq)) * amplitude
                for k, c in enumerate(coef):
                    idx = np.linspace(0, d - 1, scale_len, dtype=int)
                    data[i, idx[:min(len(idx), d)]] += c
            labels[i] = dominant * (n_classes // n_scales)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]
 