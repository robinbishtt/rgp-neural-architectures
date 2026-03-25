from __future__ import annotations
import torch.nn as nn
from src.training.trainer import Trainer
class GradientCheckpointingTrainer(Trainer):
    def __init__(self, n_segments: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_segments = n_segments
    def enable_checkpointing(self, model: nn.Module) -> nn.Module:
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
        return model