"""
src/training/optimizers/learning_rate_finder.py

LearningRateFinder: Smith (2017) LR range test for automatic learning
rate selection. Sweeps LR over a range and identifies the optimal
learning rate as the point of maximum gradient descent before loss
divergence.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


class LearningRateFinder:
    """
    Learning rate range test (Smith, 2017) for RG-Net architectures.

    Procedure:
        1. Initialize LR to lr_min.
        2. Run one mini-batch forward/backward at increasing LR.
        3. Track the loss at each step.
        4. Identify the LR with the steepest loss decrease.
        5. Return that LR / 10 as the recommended training LR.

    For ultra-deep networks (L=1000+), LR finder is critical because
    the optimal LR scales as 1/L due to gradient accumulation across layers.
    """

    def __init__(
        self,
        model:        nn.Module,
        optimizer:    Optimizer,
        criterion:    Callable,
        n_steps:      int   = 100,
        lr_min:       float = 1e-7,
        lr_max:       float = 10.0,
        beta:         float = 0.98,
        device:       Optional[torch.device] = None,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_steps   = n_steps
        self.lr_min    = lr_min
        self.lr_max    = lr_max
        self.beta      = beta
        self.device    = device or torch.device("cpu")
        self._lrs: List[float] = []
        self._losses: List[float] = []

    def run(
        self, loader
    ) -> Tuple[List[float], List[float]]:
        """
        Run the LR range test.

        Args:
            loader: iterable data loader of (inputs, targets)

        Returns:
            (lrs, smoothed_losses): lists of LR values and smoothed losses
        """
        # Exponentially increasing LR schedule
        lr_mult = (self.lr_max / self.lr_min) ** (1.0 / self.n_steps)
        lr      = self.lr_min
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        avg_loss = 0.0
        best_loss = float("inf")
        lrs, losses = [], []
        step = 0

        for batch in loader:
            if step >= self.n_steps:
                break
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out  = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            avg_loss = self.beta * avg_loss + (1.0 - self.beta) * loss_val
            smoothed = avg_loss / (1.0 - self.beta ** (step + 1))

            if smoothed > 4.0 * best_loss:
                break
            if smoothed < best_loss:
                best_loss = smoothed

            lrs.append(lr)
            losses.append(smoothed)

            lr *= lr_mult
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            step += 1

        self._lrs    = lrs
        self._losses = losses
        return lrs, losses

    def suggest_lr(self) -> float:
        """
        Suggest optimal LR as the LR at the point of steepest descent / 10.

        Returns:
            Suggested learning rate.
        """
        if not self._losses:
            return self.lr_min
        gradients = np.gradient(np.array(self._losses))
        min_idx   = int(np.argmin(gradients))
        return float(self._lrs[min_idx]) / 10.0
