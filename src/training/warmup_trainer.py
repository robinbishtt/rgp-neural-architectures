from __future__ import annotations
import logging
import math
from typing import Callable, List
import torch.nn as nn
from torch.optim import Optimizer
logger = logging.getLogger(__name__)
_WarmupFn = Callable[[int, int, float], float]
def _linear_warmup(step: int, total: int, lr_max: float) -> float:
    return lr_max * step / max(total, 1)
def _cosine_warmup(step: int, total: int, lr_max: float) -> float:
    return lr_max * 0.5 * (1.0 - math.cos(math.pi * step / max(total, 1)))
def _exp_warmup(step: int, total: int, lr_max: float, gamma: float = 5.0) -> float:
    return lr_max * math.exp((step / max(total, 1) - 1.0) * gamma)
_WARMUP_SCHEDULES = {
    :      _linear_warmup,
    :      _cosine_warmup,
    : _exp_warmup,
}
class LRWarmupScheduler:
    def __init__(
        self,
        optimizer:    Optimizer,
        warmup_steps: int,
        lr_max:       float,
        mode:         str = "linear",
        last_step:    int = 0,
    ) -> None:
        if mode not in _WARMUP_SCHEDULES:
            raise ValueError(f"mode must be one of {list(_WARMUP_SCHEDULES)}, got '{mode}'")
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.lr_max       = lr_max
        self._fn          = _WARMUP_SCHEDULES[mode]
        self._step        = last_step
        self._base_lrs: List[float] = [
            pg.get("initial_lr", lr_max)
            for pg in optimizer.param_groups
        ]
    def step(self) -> None:
        self._step += 1
        if self._step <= self.warmup_steps:
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = self._fn(self._step, self.warmup_steps, base_lr)
    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
    def is_warming_up(self) -> bool:
        return self._step < self.warmup_steps
    def state_dict(self) -> dict:
        return {"step": self._step, "warmup_steps": self.warmup_steps,
                : self.lr_max}
    def load_state_dict(self, state: dict) -> None:
        self._step        = state["step"]
        self.warmup_steps = state["warmup_steps"]
        self.lr_max       = state["lr_max"]
class LayerWiseUnlocker:
    def __init__(
        self,
        layers:           List[nn.Module],
        initial_unlocked: int = 5,
        unlock_every:     int = 5,
        block_size:       int = 5,
    ) -> None:
        self.layers           = layers
        self.unlock_every     = unlock_every
        self.block_size       = block_size
        self._n_unlocked      = 0
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad_(False)
        self._unlock_from_end(initial_unlocked)
    def step(self, epoch: int) -> int:
        if epoch > 0 and epoch % self.unlock_every == 0:
            unlocked = self._unlock_from_end(self.block_size)
            if unlocked > 0:
                logger.info(
                    ,
                    unlocked, epoch, self._n_unlocked, len(self.layers),
                )
        return self._n_unlocked
    def unlock_all(self) -> None:
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad_(True)
        self._n_unlocked = len(self.layers)
        logger.info("LayerWiseUnlocker: all %d layers unlocked.", self._n_unlocked)
    def _unlock_from_end(self, n: int) -> int:
        unlocked = 0
        for layer in reversed(self.layers):
            if self._n_unlocked >= len(self.layers):
                break
            any_frozen = any(not p.requires_grad for p in layer.parameters())
            if any_frozen:
                for p in layer.parameters():
                    p.requires_grad_(True)
                self._n_unlocked += 1
                unlocked += 1
            if unlocked >= n:
                break
        return unlocked