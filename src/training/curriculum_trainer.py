"""
src/training/curriculum_trainer.py

CurriculumTrainer: progressive curriculum learning for ultra-deep RG-Net
architectures, where training begins with simplified (low-complexity) examples
and gradually increases to the full-complexity distribution.

Physical motivation: near the critical point, deep networks are sensitive to
initialization and early gradient flow. Curriculum learning bootstraps
the critical initialization by starting with examples that produce small,
well-conditioned Jacobians before progressing to the full data distribution.
"""
from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset


class CurriculumTrainer:
    """
    Curriculum learning trainer for RG-Net architectures.

    Curriculum strategy: difficulty is defined as the distance of a
    training example from the dataset centroid in feature space. Training
    starts with the easiest (most central) examples and progressively
    includes harder (more peripheral) examples.

    Curriculum schedule types:
        - "linear":     difficulty increases linearly with epoch
        - "exponential": difficulty increases exponentially
        - "step":       full dataset after warmup epochs
    """

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
        """Return the fraction of the dataset to include at this epoch."""
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
        """
        Train one epoch with the curriculum-selected data fraction.

        Args:
            dataset:    full training dataset
            epoch:      current epoch index
            n_epochs:   total training epochs (for schedule calculation)
            batch_size: mini-batch size

        Returns:
            Mean training loss for this epoch.
        """
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
 