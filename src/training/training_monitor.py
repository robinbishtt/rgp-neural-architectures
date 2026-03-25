from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
@dataclass
class TrainingEvent:
    step:    int
    event:   str
    value:   float
    details: str = ""
class TrainingMonitor:
    def __init__(
        self,
        grad_clip:      float = 1.0,
        patience:       int   = 20,
        min_delta:      float = 1e-4,
        log_freq:       int   = 10,
    ) -> None:
        self.grad_clip   = grad_clip
        self.patience    = patience
        self.min_delta   = min_delta
        self.log_freq    = log_freq
        self._events: List[TrainingEvent] = []
        self._best_loss  = float("inf")
        self._no_improve = 0
        self._step       = 0
    def check_step(
        self,
        loss:   float,
        model:  nn.Module,
    ) -> bool:
        self._step += 1
        healthy = True
        if not torch.isfinite(torch.tensor(loss)):
            self._events.append(TrainingEvent(
                self._step, "NaN_loss", float("nan"), f"loss={loss}"
            ))
            return False
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > self.grad_clip * 1000:
            self._events.append(TrainingEvent(
                self._step, "gradient_explosion", total_norm,
                f"grad_norm={total_norm:.2e}"
            ))
            healthy = False
        if total_norm < 1e-7 and self._step > 10:
            self._events.append(TrainingEvent(
                self._step, "gradient_vanishing", total_norm,
                f"grad_norm={total_norm:.2e}"
            ))
        return healthy
    def check_epoch(self, val_loss: float) -> bool:
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss  = val_loss
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            self._events.append(TrainingEvent(
                self._step, "early_stop", val_loss,
                f"No improvement for {self.patience} epochs"
            ))
            return False
        return True
    @property
    def events(self) -> List[TrainingEvent]:
        return list(self._events)
    def has_anomaly(self) -> bool:
        return any(e.event in ("NaN_loss", "gradient_explosion")
                   for e in self._events)