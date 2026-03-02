"""
src/training/batch_sampler.py

DeterministicBatchSampler: reproducible batch ordering for RGP training.

Motivation
----------
Standard PyTorch DataLoader shuffles batches non-deterministically even when
a seed is set, because the shuffle seed is not serialised in the loader state.
For rigorous reproducibility the batch ordering must be identical across:
    (a) two runs on the same machine with the same master seed
    (b) runs on different machines (different CPU, GPU, OS versions)
    (c) resumed training after a checkpoint

This module provides DeterministicBatchSampler, which derives its per-epoch
shuffle seed from the SeedRegistry master seed and the epoch index:

    epoch_seed = hash(master_seed || epoch_idx)  (mod 2^32)

This ensures:
    * Epoch 0 always produces the same batch order regardless of resume
    * Different epochs always produce different (but deterministic) orderings
    * No state needs to be serialised beyond the epoch index

CurriculumBatchSampler adds difficulty-ordered sampling for progressive
training of ultra-deep networks: easy examples (low loss) are presented
first during the warmup phase, then sampling transitions to uniform random.
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DeterministicBatchSampler(Sampler):
    """
    Epoch-deterministic batch sampler backed by SeedRegistry.

    Parameters
    ----------
    dataset_size : int
        Total number of samples in the dataset.
    batch_size : int
        Number of samples per batch.
    drop_last : bool
        If True, drop the last incomplete batch.
    shuffle : bool
        If True, shuffle within each epoch using a deterministic epoch seed.
    master_seed : int, optional
        Override master seed instead of reading from SeedRegistry singleton.
    """

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
                    "DeterministicBatchSampler: SeedRegistry unavailable, "
                    "defaulting to master_seed=42."
                )

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to advance the deterministic shuffle."""
        self._epoch = epoch

    def _epoch_seed(self) -> int:
        """
        Derive a deterministic per-epoch seed.

        Uses a multiplicative hash to spread seeds across the integer space:
            epoch_seed = (master_seed * 6364136223846793005 + epoch) mod 2^32
        This is the multiplier from the PCG family of hash functions.
        """
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
    """
    Curriculum sampler that transitions from difficulty-ordered to random.

    During warmup epochs (epoch < warmup_epochs) samples are ordered from
    easiest to hardest based on a pre-computed per-sample difficulty score
    (e.g., loss on previous epoch, or a heuristic like distance from class
    centroid).  After warmup, sampling becomes fully random (like
    DeterministicBatchSampler with shuffle=True).

    This is used by CurriculumTrainer for ultra-deep network warm-up.

    Parameters
    ----------
    difficulty_scores : list of float
        Per-sample difficulty scores.  Higher = harder.  Must have length
        equal to ``dataset_size``.
    warmup_epochs : int
        Number of epochs with curriculum ordering before switching to random.
    """

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
            # Curriculum phase: easy-to-hard ordering
            scored = sorted(
                range(self.dataset_size),
                key=lambda i: self.difficulty_scores[i],
            )
            indices = scored
        else:
            # Post-curriculum phase: deterministic random shuffle
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
 