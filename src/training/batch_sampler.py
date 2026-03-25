from __future__ import annotations
import logging
from typing import Iterator, List, Optional
import numpy as np
from torch.utils.data import Sampler
logger = logging.getLogger(__name__)
class DeterministicBatchSampler(Sampler):
    def __init__(
        self,
        dataset_size: int,
        batch_size:   int,
        drop_last:    bool = False,
        shuffle:      bool = True,
        master_seed:  Optional[int] = None,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size   = batch_size
        self.drop_last    = drop_last
        self.shuffle      = shuffle
        self._epoch:  int = 0
        if master_seed is not None:
            self._master_seed = master_seed
        else:
            try:
                from src.utils.seed_registry import SeedRegistry
                self._master_seed = SeedRegistry.get_instance().master_seed
            except Exception:
                self._master_seed = 42
                logger.warning(
                )
    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
    def _epoch_seed(self) -> int:
        return int((self._master_seed * 6_364_136_223_846_793_005 + self._epoch) % (2**32))
    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            rng = np.random.RandomState(self._epoch_seed())
            rng.shuffle(indices)
        batches: List[List[int]] = []
        for start in range(0, self.dataset_size, self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)
        return iter(batches)
    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
class CurriculumBatchSampler(DeterministicBatchSampler):
    def __init__(
        self,
        dataset_size:      int,
        batch_size:        int,
        difficulty_scores: Optional[List[float]] = None,
        warmup_epochs:     int = 5,
        drop_last:         bool = False,
        master_seed:       Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_size = dataset_size,
            batch_size   = batch_size,
            drop_last    = drop_last,
            shuffle      = True,
            master_seed  = master_seed,
        )
        self.warmup_epochs     = warmup_epochs
        self.difficulty_scores = (
            difficulty_scores if difficulty_scores is not None
            else [1.0] * dataset_size
        )
    def __iter__(self) -> Iterator[List[int]]:
        if self._epoch < self.warmup_epochs and self.difficulty_scores is not None:
            scored = sorted(
                range(self.dataset_size),
                key=lambda i: self.difficulty_scores[i],
            )
            indices = scored
        else:
            indices = list(range(self.dataset_size))
            rng = np.random.RandomState(self._epoch_seed())
            rng.shuffle(indices)
        batches: List[List[int]] = []
        for start in range(0, self.dataset_size, self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)
        return iter(batches)