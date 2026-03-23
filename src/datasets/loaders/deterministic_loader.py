"""
src/datasets/loaders/deterministic_loader.py

Fixed-seed, reproducible DataLoader wrapping the SeedRegistry worker_init_fn.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils.seed_registry import SeedRegistry


class DeterministicDataLoader(DataLoader):
    """
    DataLoader guaranteed to produce identical batches across runs.
    All workers are seeded via SeedRegistry.get_worker_seed(worker_id).
    """

    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 num_workers: int = 4, **kwargs) -> None:
        reg = SeedRegistry.get_instance()
        kwargs.setdefault("worker_init_fn", reg.worker_init_fn)
        kwargs.setdefault("shuffle", True)
        kwargs.setdefault("pin_memory", torch.cuda.is_available())
        super().__init__(dataset, batch_size=batch_size,
                         num_workers=num_workers, **kwargs)
 