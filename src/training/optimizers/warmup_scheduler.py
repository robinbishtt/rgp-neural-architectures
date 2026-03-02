"""
src/training/optimizers/warmup_scheduler.py

WarmupScheduler: linear learning rate warmup combined with configurable
decay schedule. Essential for training ultra-deep RG-Net architectures
(L=500+) where cold-start with full learning rate causes gradient explosions.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by configurable decay.

    Phase 1 (0 ≤ step < warmup_steps):
        lr = base_lr * step / warmup_steps
    Phase 2 (step ≥ warmup_steps):
        lr = base_lr * decay_fn(step - warmup_steps)

    For deep RG-Net training, warmup_steps = 100-500 is recommended
    to allow the critical initialization to stabilize before full-rate
    gradient updates begin.
    """

    def __init__(
        self,
        optimizer:     Optimizer,
        warmup_steps:  int,
        decay:         str   = "cosine",
        max_steps:     int   = 10_000,
        min_lr_ratio:  float = 0.01,
        last_epoch:    int   = -1,
    ) -> None:
        self.warmup_steps  = warmup_steps
        self.decay         = decay
        self.max_steps     = max_steps
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(self.warmup_steps, 1)
        elif self.decay == "cosine":
            import math
            progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            factor   = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        elif self.decay == "linear":
            progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            factor   = max(self.min_lr_ratio, 1.0 - progress * (1.0 - self.min_lr_ratio))
        elif self.decay == "constant":
            factor = 1.0
        else:
            factor = 1.0

        return [base_lr * factor for base_lr in self.base_lrs]
 