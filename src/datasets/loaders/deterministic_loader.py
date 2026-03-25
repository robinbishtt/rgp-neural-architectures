from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils.seed_registry import SeedRegistry
class DeterministicDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 num_workers: int = 4, **kwargs) -> None:
        reg = SeedRegistry.get_instance()
        kwargs.setdefault("worker_init_fn", reg.worker_init_fn)
        kwargs.setdefault("shuffle", True)
        kwargs.setdefault("pin_memory", torch.cuda.is_available())
        super().__init__(dataset, batch_size=batch_size,
                         num_workers=num_workers, **kwargs)