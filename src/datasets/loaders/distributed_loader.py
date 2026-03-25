from __future__ import annotations
from torch.utils.data import DataLoader, Dataset, DistributedSampler
class DistributedDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, rank: int, world_size: int,
                 batch_size: int = 32, num_workers: int = 4, **kwargs) -> None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        kwargs.pop("shuffle", None)  
        super().__init__(dataset, batch_size=batch_size, sampler=sampler,
                         num_workers=num_workers, **kwargs)
        self.dist_sampler = sampler
    def set_epoch(self, epoch: int) -> None:
        self.dist_sampler.set_epoch(epoch)