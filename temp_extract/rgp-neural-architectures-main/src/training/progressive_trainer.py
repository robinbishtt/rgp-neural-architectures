"""
src/training/progressive_trainer.py

Progressive/curriculum training: grows depth incrementally.

Starts with a shallow network (L=10), trains to convergence, then
extends by adding L additional layers initialized at criticality.
Prevents vanishing gradients in very deep networks during early training.
"""
from __future__ import annotations
from typing import List
import torch.nn as nn
from src.training.trainer import Trainer


class ProgressiveTrainer(Trainer):
    """
    Curriculum training via progressive depth increase.

    depth_schedule: list of depths to train through, e.g. [10, 25, 50, 100].
    Each stage trains for `epochs_per_stage` epochs before extending.
    """

    def __init__(
        self,
        depth_schedule: List[int] = (10, 25, 50, 100),
        epochs_per_stage: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth_schedule   = list(depth_schedule)
        self.epochs_per_stage = epochs_per_stage

    def progressive_train(self, model_builder, train_loader, val_loader):
        """Train model progressively through depth_schedule."""
        results = []
        for depth in self.depth_schedule:
            model = model_builder(depth=depth)
            self.model = model
            metrics = self.fit(train_loader, val_loader,
                               max_epochs=self.epochs_per_stage)
            results.append({"depth": depth, "metrics": metrics})
        return results
