"""
src/training/learning_rate_scheduler.py

Composite learning rate scheduler registry for RGP experiments.
Provides a unified factory interface for all LR schedules used across
H1, H2, H3 validation experiments.
"""
from __future__ import annotations
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, _LRScheduler
)
from src.training.optimizers.warmup_scheduler import LinearWarmupScheduler
from src.training.optimizers.cosine_annealing import CosineAnnealingWithRestarts


def build_scheduler(
    optimizer:    Optimizer,
    schedule:     str,
    total_steps:  int,
    warmup_steps: int   = 0,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function: construct LR scheduler from string name.

    Args:
        optimizer:    PyTorch optimizer
        schedule:     one of "cosine", "warmup_cosine", "step", "plateau",
                      "warmup_cosine_restarts", "constant"
        total_steps:  total number of training steps
        warmup_steps: steps for linear warmup (used if schedule includes "warmup")
        **kwargs:     additional arguments passed to the underlying scheduler

    Returns:
        Configured LR scheduler.
    """
    schedule = schedule.lower()

    if schedule == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=total_steps,
            eta_min=kwargs.get("eta_min", 0.0)
        )
    elif schedule == "warmup_cosine":
        return LinearWarmupScheduler(
            optimizer, warmup_steps=warmup_steps,
            decay="cosine", max_steps=total_steps,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.01),
        )
    elif schedule == "warmup_cosine_restarts":
        return CosineAnnealingWithRestarts(
            optimizer,
            T_0=kwargs.get("T_0", total_steps // 4),
            T_mult=kwargs.get("T_mult", 2.0),
        )
    elif schedule == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", total_steps // 3),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif schedule == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            patience=kwargs.get("patience", 10),
            factor=kwargs.get("factor", 0.5),
        )
    elif schedule == "constant":
        return StepLR(optimizer, step_size=total_steps + 1, gamma=1.0)
    else:
        raise ValueError(f"Unknown LR schedule: '{schedule}'")
 