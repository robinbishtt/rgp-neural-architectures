from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from src.utils.seed_registry import SeedRegistry
from src.utils.error_handler import NaNRecoveryHandler
logger = logging.getLogger(__name__)
@dataclass
class TrainingConfig:
    n_epochs:       int   = 100
    batch_size:     int   = 256
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp:        bool  = True
    log_interval:   int   = 10
    checkpoint_interval: int = 25
    seed:           int   = 42
    fast_track:     bool  = False
    def __post_init__(self) -> None:
        if self.fast_track:
            self.n_epochs  = 2
            self.batch_size = 64
@dataclass
class TrainingResult:
    train_losses: List[float] = field(default_factory=list)
    val_losses:   List[float] = field(default_factory=list)
    val_accs:     List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    total_epochs: int = 0
    elapsed_s:    float = 0.0
class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        cfg: TrainingConfig = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        telemetry=None,
    ) -> None:
        self.model    = model
        self.cfg      = cfg or TrainingConfig()
        self.device   = device or torch.device("cpu")
        self.ckpt_dir = Path(checkpoint_dir or "checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry = telemetry
        SeedRegistry.get_instance().set_master_seed(self.cfg.seed)
        self.optimizer = None
        self.scaler    = None
        self.nan_handler = NaNRecoveryHandler()
        self._best_val_acc = 0.0
        if self.model is not None:
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
            self.scaler = GradScaler(enabled=self.cfg.use_amp and self.device.type == "cuda")
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> TrainingResult:
        if self.model is None or self.optimizer is None or self.scaler is None:
            raise ValueError(
                "Trainer is not fully initialized: model, optimizer, and scaler "
                "must all be available before calling train()."
            )
        criterion = criterion or nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.n_epochs)
        result    = TrainingResult()
        t0        = time.perf_counter()
        for epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, epoch)
            val_loss, val_acc = self._val_epoch(val_loader, criterion)
            scheduler.step()
            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.val_accs.append(val_acc)
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, best=True)
            if epoch % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_acc)
            if epoch % self.cfg.log_interval == 0:
                logger.info(
                    "Epoch [%d/%d]  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                    epoch, self.cfg.n_epochs, train_loss, val_loss, val_acc,
                )
                if self.telemetry:
                    self.telemetry.log_scalar("train/loss", train_loss, epoch)
                    self.telemetry.log_scalar("val/loss",   val_loss,   epoch)
                    self.telemetry.log_scalar("val/acc",    val_acc,    epoch)
        result.best_val_acc = self._best_val_acc
        result.total_epochs = self.cfg.n_epochs
        result.elapsed_s    = time.perf_counter() - t0
        return result
    def _train_epoch(self, loader: DataLoader, criterion: nn.Module, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.use_amp and self.device.type == "cuda"):
                out  = self.model(x)
                loss = criterion(out, y)
            if not self.nan_handler.check(loss, self.optimizer):
                logger.error("Unrecoverable NaN at epoch %d", epoch)
                break
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)
    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, criterion: nn.Module):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out  = self.model(x)
            total_loss += criterion(out, y).item()
            preds   = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return total_loss / max(len(loader), 1), correct / max(total, 1)
    def _save_checkpoint(self, epoch: int, val_acc: float, best: bool = False) -> None:
        fname = "best.pt" if best else f"epoch_{epoch:04d}.pt"
        path  = self.ckpt_dir / fname
        torch.save({
            "epoch":               epoch,
            "val_acc":             val_acc,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state":           SeedRegistry.get_instance().snapshot_state(),
        }, str(path))
