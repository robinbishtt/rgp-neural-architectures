"""
src/datasets/hierarchical_dataset.py

Hierarchical dataset generators with tunable correlation structure.

Datasets:
  HierarchicalGaussianDataset  — nested Gaussian clusters mimicking RG structure
  IIDGaussianDataset           — IID baseline for H3 comparison
  CorrelatedDataset            — tunable correlation length ξ_data
  OODDataset                   — out-of-distribution shift for generalisation tests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.seed_registry import SeedRegistry


@dataclass
class DatasetConfig:
    n_samples:   int   = 10000
    input_dim:   int   = 64
    n_classes:   int   = 10
    xi_data:     float = 10.0     # correlation length
    noise_std:   float = 0.1
    seed:        int   = 42
    fast_track:  bool  = False

    def __post_init__(self) -> None:
        if self.fast_track:
            self.n_samples = 1000


class HierarchicalGaussianDataset(Dataset):
    """
    Hierarchically structured dataset.

    Generation procedure:
    1. Sample n_classes cluster centres in R^input_dim
    2. Each class samples sub-clusters at scale ξ_data / 2
    3. Samples drawn from sub-cluster Gaussians with σ = noise_std

    The resulting data has a genuine multi-scale correlation structure
    that directly tests H3 (whether depth ~ log(ξ_data) is necessary).
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        rng = np.random.default_rng(cfg.seed)
        D, K = cfg.input_dim, cfg.n_classes
        xi   = cfg.xi_data

        # Class-level centres
        class_centres = rng.standard_normal((K, D)) * xi

        # Sub-cluster centres (two per class)
        sub_centres = np.stack([
            class_centres + rng.standard_normal((K, D)) * (xi / 2)
            for _ in range(2)
        ], axis=1)  # (K, 2, D)

        X_list, y_list = [], []
        per_class = cfg.n_samples // K
        for k in range(K):
            for sub in range(2):
                n = per_class // 2
                x = sub_centres[k, sub] + rng.standard_normal((n, D)) * cfg.noise_std
                X_list.append(x)
                y_list.extend([k] * n)

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Shuffle
        perm  = rng.permutation(len(y))
        self.X = torch.from_numpy(X[perm])
        self.y = torch.from_numpy(y[perm])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class IIDGaussianDataset(Dataset):
    """IID baseline — no correlation structure."""

    def __init__(self, cfg: DatasetConfig) -> None:
        rng = np.random.default_rng(cfg.seed)
        D, K = cfg.input_dim, cfg.n_classes
        class_centres = rng.standard_normal((K, D)) * cfg.xi_data
        n_per = cfg.n_samples // K

        X_list, y_list = [], []
        for k in range(K):
            X_list.append(class_centres[k] + rng.standard_normal((n_per, D)) * cfg.noise_std)
            y_list.extend([k] * n_per)

        X  = np.vstack(X_list).astype(np.float32)
        y  = np.array(y_list, dtype=np.int64)
        perm = rng.permutation(len(y))
        self.X = torch.from_numpy(X[perm])
        self.y = torch.from_numpy(y[perm])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class OODDataset(Dataset):
    """
    Out-of-distribution dataset for H3 generalisation evaluation.
    Applies a correlation shift by rotating the covariance structure.
    """

    def __init__(self, cfg: DatasetConfig, shift_magnitude: float = 0.5) -> None:
        rng = np.random.default_rng(cfg.seed + 1000)
        D, K = cfg.input_dim, cfg.n_classes

        # Rotation matrix for covariance shift
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        scale = 1.0 + shift_magnitude
        cov   = Q @ np.diag([scale] * (D // 2) + [1.0] * (D - D // 2)) @ Q.T

        class_centres = rng.standard_normal((K, D)) * cfg.xi_data
        n_per = cfg.n_samples // K

        X_list, y_list = [], []
        for k in range(K):
            samples = rng.multivariate_normal(class_centres[k], cov * cfg.noise_std ** 2, n_per)
            X_list.append(samples.astype(np.float32))
            y_list.extend([k] * n_per)

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)
        perm = rng.permutation(len(y))
        self.X = torch.from_numpy(X[perm])
        self.y = torch.from_numpy(y[perm])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloaders(
    cfg: DatasetConfig,
    dataset_type: str = "hierarchical",
    train_fraction: float = 0.8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for the specified dataset type."""
    reg = SeedRegistry.get_instance()
    if dataset_type == "hierarchical":
        ds = HierarchicalGaussianDataset(cfg)
    elif dataset_type == "iid":
        ds = IIDGaussianDataset(cfg)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    n_train = int(len(ds) * train_fraction)
    n_val   = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.n_samples // 20,
        shuffle=True, num_workers=num_workers,
        worker_init_fn=reg.worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.n_samples // 10,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader
 