"""
src/datasets/medical_hierarchy.py

Medical imaging dataset with anatomical hierarchy (organ -> sub-region -> cell).
Uses synthetic generation when real data unavailable.
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from src.datasets.synthetic_hierarchy import SyntheticHierarchy


class MedicalHierarchy(Dataset):
    """
    Multi-scale medical imaging dataset.

    In the absence of real medical data (requires IRB/license), generates
    synthetic data with anatomy-inspired correlation structure. Structure:
    coarse scale (organ) -> mid scale (tissue) -> fine scale (cell).
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_features: int = 1024,
        seed: int = 42,
        use_synthetic: bool = True,
    ) -> None:
        super().__init__()
        if use_synthetic:
            self._base = SyntheticHierarchy(
                n_samples=n_samples,
                n_features=n_features,
                n_scales=3,
                correlation_length=20.0,
                n_classes=6,
                seed=seed,
            )
        else:
            raise NotImplementedError("Real medical data loading requires IRB approval.")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        return self._base[idx]
 