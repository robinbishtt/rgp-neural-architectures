from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
class CurriculumTrainer:
    def __init__(
        self,
        model:         nn.Module,
        optimizer:     torch.optim.Optimizer,
        criterion:     Callable,
        schedule:      str   = "linear",
        warmup_epochs: int   = 10,
        device:        Optional[torch.device] = None,
    ) -> None:
        self.model         = model
        self.optimizer     = optimizer
        self.criterion     = criterion
        self.schedule      = schedule
        self.warmup_epochs = warmup_epochs
        self.device        = device or torch.device("cpu")
    def _difficulty_fraction(self, epoch: int, n_epochs: int) -> float:
        if epoch >= self.warmup_epochs:
            return 1.0
        frac = epoch / max(self.warmup_epochs, 1)
        if self.schedule == "linear":
            return 0.2 + 0.8 * frac
        elif self.schedule == "exponential":
            return 0.2 + 0.8 * (frac ** 0.5)
        elif self.schedule == "step":
            return 1.0 if epoch >= self.warmup_epochs else 0.3
        return 1.0
    def train_epoch(
        self,
        dataset:       Dataset,
        epoch:         int,
        n_epochs:      int,
        batch_size:    int = 64,
    ) -> float:
        frac    = self._difficulty_fraction(epoch, n_epochs)
        n       = int(len(dataset) * frac)
        indices = torch.randperm(len(dataset))[:n].tolist()
        subset  = Subset(dataset, indices)
        loader  = DataLoader(subset, batch_size=batch_size, shuffle=True)
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out  = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)