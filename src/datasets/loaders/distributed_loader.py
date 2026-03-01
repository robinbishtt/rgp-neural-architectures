"""
src/datasets/loaders/distributed_loader.py

Multi-process DataLoader with DistributedSampler for distributed training.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DistributedDataLoader(DataLoader):
    """
    DataLoader using DistributedSampler for torch.distributed DDP training.
    Ensures non-overlapping data partitions across ranks.
    """

    def __init__(self, dataset: Dataset, rank: int, world_size: int,
                 batch_size: int = 32, num_workers: int = 4, **kwargs) -> None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        kwargs.pop("shuffle", None)  # shuffle handled by DistributedSampler
        super().__init__(dataset, batch_size=batch_size, sampler=sampler,
                         num_workers=num_workers, **kwargs)
        self.dist_sampler = sampler

    def set_epoch(self, epoch: int) -> None:
        """Call at start of each epoch to re-shuffle across ranks."""
        self.dist_sampler.set_epoch(epoch)
