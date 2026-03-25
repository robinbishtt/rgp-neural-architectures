from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
class LearningRateFinder:
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
        if not self._losses:
            return self.lr_min
        gradients = np.gradient(np.array(self._losses))
        min_idx   = int(np.argmin(gradients))
        return float(self._lrs[min_idx]) / 10.0