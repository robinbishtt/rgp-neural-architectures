"""
src/training/mixed_precision_trainer.py

FP16/BF16 training via torch.amp for memory efficiency on modern GPUs.
"""
from __future__ import annotations
import torch
from torch.cuda.amp import GradScaler, autocast
from src.training.trainer import Trainer


class MixedPrecisionTrainer(Trainer):
    """
    Automatic mixed precision training.

    Uses FP16 on CUDA < Ampere, BF16 on Ampere+ (A100/H100).
    Maintains FP32 master weights. GradScaler prevents underflow.
    """

    def __init__(self, dtype: torch.dtype = torch.float16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype   = dtype
        self.scaler  = GradScaler() if dtype == torch.float16 else None

    def train_step(self, batch):
        x, y = batch
        with autocast(dtype=self.dtype):
            logits = self.model(x)
            loss   = self.criterion(logits, y)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        return loss.item()
 