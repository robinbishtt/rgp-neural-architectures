"""
src/utils/error_handler.py

Error handling and recovery strategies for long-running experiments.
Implements automatic retry, checkpoint resume, OOM recovery, and NaN recovery.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import torch

logger = logging.getLogger(__name__)


class DataIntegrityError(Exception):
    """Raised when SHA-256 checksum verification fails."""


class NaNRecoveryHandler:
    """
    Detects NaN loss and rolls back to last valid checkpoint with reduced LR.
    """

    def __init__(self, patience: int = 3, lr_scale: float = 0.5) -> None:
        self.patience  = patience
        self.lr_scale  = lr_scale
        self._nan_count = 0

    def check(self, loss: torch.Tensor, optimizer, checkpoint_manager=None) -> bool:
        """
        Returns True if training should continue, False if unrecoverable.
        """
        if torch.isnan(loss) or torch.isinf(loss):
            self._nan_count += 1
            logger.warning("NaN/Inf loss detected (count=%d).", self._nan_count)

            if self._nan_count > self.patience:
                logger.error("Exceeded NaN patience=%d. Stopping.", self.patience)
                return False

            # Scale down learning rate
            for pg in optimizer.param_groups:
                pg["lr"] *= self.lr_scale
            logger.info("LR reduced by %.1f. New LR=%.2e.",
                        self.lr_scale, optimizer.param_groups[0]["lr"])

            # Restore from checkpoint if available
            if checkpoint_manager is not None:
                checkpoint_manager.restore_latest()

        else:
            self._nan_count = 0  # reset counter on valid loss

        return True


class OOMRecoveryHandler:
    """
    Handles CUDA out-of-memory by halving batch size and retrying.
    """

    def __init__(self, min_batch_size: int = 4, max_retries: int = 3) -> None:
        self.min_batch_size = min_batch_size
        self.max_retries    = max_retries

    def run_with_recovery(self, fn: Callable, batch_size: int, *args, **kwargs):
        """
        Execute fn(batch_size, *args, **kwargs), halving batch on OOM.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return fn(batch_size, *args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_bs = max(self.min_batch_size, batch_size // 2)
                if new_bs == batch_size:
                    raise
                logger.warning("OOM: reducing batch %d -> %d", batch_size, new_bs)
                batch_size = new_bs

        raise RuntimeError("OOM recovery failed after max retries.")


class CheckpointResumeHandler:
    """
    Automatically resumes training from the latest checkpoint on failure.
    """

    def __init__(self, max_retries: int = 5, retry_delay: float = 30.0) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def run_with_resume(self, train_fn: Callable, checkpoint_manager, **kwargs):
        """Run train_fn; on exception, restore checkpoint and retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return train_fn(**kwargs)
            except (RuntimeError, KeyboardInterrupt) as exc:
                if attempt == self.max_retries:
                    raise
                logger.error("Training failed (attempt %d/%d): %s",
                             attempt, self.max_retries, exc)
                logger.info("Restoring checkpoint and retrying in %.0fs...",
                            self.retry_delay)
                checkpoint_manager.restore_latest()
                time.sleep(self.retry_delay)


class TimeoutHandler:
    """
    Saves a checkpoint before cluster walltime expiration.
    """

    def __init__(self, walltime_seconds: float, checkpoint_manager=None) -> None:
        self._deadline         = time.time() + walltime_seconds
        self._checkpoint_manager = checkpoint_manager

    def check_deadline(self, buffer_seconds: float = 120.0) -> bool:
        """Return True if we should stop now (within buffer of deadline)."""
        remaining = self._deadline - time.time()
        if remaining <= buffer_seconds:
            logger.warning("%.0f s to deadline - saving checkpoint.", remaining)
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.save()
            return True
        return False
 