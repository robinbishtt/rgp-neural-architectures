"""
src/datasets/loaders/cached_loader.py

In-memory caching DataLoader for small datasets.
Pre-loads entire dataset to GPU/CPU RAM on first pass.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CachedDataLoader:
    """
    Caches full dataset in RAM on first access. 
    Subsequent passes use in-memory tensors (no disk I/O).
    Suitable for datasets < available RAM (typically < 16GB).
    """

    def __init__(self, dataset: Dataset, batch_size: int = 256,
                 device: torch.device = torch.device("cpu"),
                 num_workers: int = 0, **kwargs) -> None:
        self.batch_size = batch_size
        self._cache     = self._build_cache(dataset, device)
        self._loader    = DataLoader(self._cache, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, **kwargs)

    def _build_cache(self, dataset: Dataset, device: torch.device) -> TensorDataset:
        all_x, all_y = [], []
        for x, y in DataLoader(dataset, batch_size=512, num_workers=0):
            all_x.append(x.to(device))
            all_y.append(y.to(device) if torch.is_tensor(y) else torch.tensor(y).to(device))
        return TensorDataset(torch.cat(all_x), torch.cat(all_y))

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)
