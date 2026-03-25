from __future__ import annotations
from typing import List
from src.training.trainer import Trainer
class ProgressiveTrainer(Trainer):
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
        results = []
        for depth in self.depth_schedule:
            model = model_builder(depth=depth)
            self.model = model
            metrics = self.fit(train_loader, val_loader,
                               max_epochs=self.epochs_per_stage)
            results.append({"depth": depth, "metrics": metrics})
        return results