"""
src/training/warmup_trainer.py

WarmupTrainer: linear/cosine warmup scheduling for ultra-deep RG-Net training.

Motivation
----------
Ultra-deep networks (L=500-1000) placed at the edge-of-chaos manifold are
numerically sensitive during early training.  With a standard learning rate
schedule the initial large gradients can kick the network off the critical
manifold before the Lyapunov exponent monitoring has had time to detect the
problem.

WarmupTrainer wraps the standard Trainer with a configurable warmup phase:
    * Linear warmup:  lr(t) = lr_max * (t / T_warmup)  for t < T_warmup
    * Cosine warmup:  lr(t) = lr_max * 0.5(1 - cos(πt/T_warmup))
    * Exponential:    lr(t) = lr_max * exp((t/T_warmup - 1) * gamma)

After warmup the scheduler switches to the main schedule (cosine annealing
or flat) for the remaining training budget.

This module is also responsible for the "layer-wise unlock" curriculum:
    * Epoch 1-5:   only the last K layers (default K=5) receive gradients
    * Epoch 6-10:  unlock layers from the output backward every 5 epochs
    * Epoch >10:   all layers receive gradients

Layer-wise unlocking is critical for L=1000 networks: attempting to train
all 1000 layers simultaneously from random initialisation causes gradient
vanishing even with critical initialisation.
"""
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
    "linear":      _linear_warmup,
    "cosine":      _cosine_warmup,
    "exponential": _exp_warmup,
}


class LRWarmupScheduler:
    """
    Learning rate warmup scheduler wrapping any PyTorch optimizer.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be controlled.
    warmup_steps : int
        Number of gradient steps for the warmup phase.
    lr_max : float
        Target learning rate at the end of warmup.
    mode : str
        Warmup curve type: "linear" | "cosine" | "exponential".
    last_step : int
        Step index to resume from (for checkpoint resume).

    Example
    -------
    ::
        scheduler = LRWarmupScheduler(opt, warmup_steps=500, lr_max=1e-3)
        for step, batch in enumerate(loader):
            loss = forward(batch)
            loss.backward()
            opt.step()
            scheduler.step()
    """

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

        # Store initial lrs for param groups that override lr_max
        self._base_lrs: List[float] = [
            pg.get("initial_lr", lr_max)
            for pg in optimizer.param_groups
        ]

    def step(self) -> None:
        """Advance scheduler by one step and update optimizer lr."""
        self._step += 1
        if self._step <= self.warmup_steps:
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = self._fn(self._step, self.warmup_steps, base_lr)

    @property
    def current_lr(self) -> float:
        """Current learning rate of the first param group."""
        return self.optimizer.param_groups[0]["lr"]

    def is_warming_up(self) -> bool:
        """True if still in the warmup phase."""
        return self._step < self.warmup_steps

    def state_dict(self) -> dict:
        return {"step": self._step, "warmup_steps": self.warmup_steps,
                "lr_max": self.lr_max}

    def load_state_dict(self, state: dict) -> None:
        self._step        = state["step"]
        self.warmup_steps = state["warmup_steps"]
        self.lr_max       = state["lr_max"]


class LayerWiseUnlocker:
    """
    Progressive layer unlocking for ultra-deep networks.

    Given a list of parameter groups (or nn.ModuleList of layers) ordered
    from input to output, this utility freezes all layers at init and unlocks
    them from output toward input in blocks of ``unlock_every`` epochs.

    Parameters
    ----------
    layers : list of nn.Module
        All hidden layers in input-to-output order.
    initial_unlocked : int
        Number of output-side layers unlocked from the start.
    unlock_every : int
        Unlock ``block_size`` more layers every this many epochs.
    block_size : int
        Number of layers unlocked per unlock event.
    """

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

        # Freeze everything
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad_(False)

        # Unlock the last initial_unlocked layers
        self._unlock_from_end(initial_unlocked)

    def step(self, epoch: int) -> int:
        """
        Called at the start of each epoch.  Unlocks additional layers if due.

        Returns
        -------
        int
            Total number of unlocked layers after this step.
        """
        if epoch > 0 and epoch % self.unlock_every == 0:
            unlocked = self._unlock_from_end(self.block_size)
            if unlocked > 0:
                logger.info(
                    "LayerWiseUnlocker: unlocked %d more layers at epoch %d "
                    "(total unlocked: %d/%d).",
                    unlocked, epoch, self._n_unlocked, len(self.layers),
                )
        return self._n_unlocked

    def unlock_all(self) -> None:
        """Unlock all remaining frozen layers immediately."""
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
 