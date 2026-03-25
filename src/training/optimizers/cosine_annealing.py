from __future__ import annotations
import math
from typing import List, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWithRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer:  Optimizer,
        T_0:        int,
        T_mult:     float = 2.0,
        eta_min:    float = 0.0,
        last_epoch: int   = -1,
    ) -> None:
        self.T_0      = T_0
        self.T_mult   = T_mult
        self.eta_min  = eta_min
        self.T_cur    = 0
        self.T_i      = T_0
        super().__init__(optimizer, last_epoch)
    def get_lr(self) -> List[float]:
        cos_factor = (1.0 + math.cos(math.pi * self.T_cur / self.T_i)) / 2.0
        return [
            self.eta_min + (base_lr - self.eta_min) * cos_factor
            for base_lr in self.base_lrs
        ]
    def step(self, epoch: Optional[int] = None) -> None:
        if self.T_cur == self.T_i:
            self.T_cur = 0
            self.T_i   = int(self.T_i * self.T_mult)
        else:
            self.T_cur += 1
        super().step(epoch)