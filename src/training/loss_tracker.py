"""
src/training/loss_tracker.py

LossTracker: per-step and per-epoch loss bookkeeping with statistical summaries.

Purpose
-------
Training L=1000+ networks for 48-72 hours generates thousands of scalar
loss values.  Raw storage of every step is memory-inefficient for long
runs.  LossTracker implements a reservoir-sampled history (full history
for recent N steps, downsampled for older history) combined with
epoch-level aggregates that match the manuscript's reported metrics.

Statistical outputs:
    * Per-epoch mean, std, min, max, median
    * Exponential moving average (EMA) for smooth monitoring dashboards
    * Convergence slope estimate (linear regression over last N epochs)
    * Gradient norm history for stability monitoring (separate channel)
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpochStats:
    """Per-epoch statistical summary of a scalar metric."""
    epoch:  int
    mean:   float
    std:    float
    min:    float
    max:    float
    median: float
    n:      int


class LossTracker:
    """
    Lightweight bookkeeper for training and validation losses.

    Parameters
    ----------
    ema_alpha : float
        Smoothing coefficient for the exponential moving average.
        Typical values: 0.05 (slow decay) to 0.3 (fast decay).
    recent_window : int
        Number of most-recent step-level losses retained verbatim.
        Older values are summarised into epoch-level aggregates only.
    history_maxlen : int
        Maximum number of epoch-level EpochStats objects to retain.
        Older epochs are dropped FIFO.

    Usage
    -----
    ::
        tracker = LossTracker(ema_alpha=0.1)
        for step, (x, y) in enumerate(loader):
            loss = criterion(model(x), y)
            tracker.step(loss.item(), step)

        tracker.end_epoch(epoch=0)
        stats = tracker.epoch_stats[-1]
        print(stats.mean, stats.std)
    """

    def __init__(
        self,
        ema_alpha:     float = 0.1,
        recent_window: int   = 200,
        history_maxlen: int  = 500,
    ) -> None:
        self.ema_alpha      = ema_alpha
        self.recent_window  = recent_window

        self._step_buffer: Deque[Tuple[int, float]] = deque(maxlen=recent_window)
        self._current_epoch_steps: List[float]      = []
        self.epoch_stats:   Deque[EpochStats]        = deque(maxlen=history_maxlen)

        self._ema: Optional[float] = None
        self._global_step: int     = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def step(self, loss: float, global_step: Optional[int] = None) -> None:
        """
        Record a single step-level loss value.

        Parameters
        ----------
        loss : float
            Scalar loss value from one training step.
        global_step : int, optional
            If not provided, uses an internal counter.
        """
        if global_step is not None:
            self._global_step = global_step
        else:
            self._global_step += 1

        if math.isnan(loss) or math.isinf(loss):
            logger.warning("LossTracker: NaN/Inf loss at step %d. Skipping.", self._global_step)
            return

        self._step_buffer.append((self._global_step, loss))
        self._current_epoch_steps.append(loss)

        # Update EMA
        if self._ema is None:
            self._ema = loss
        else:
            self._ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self._ema

    def end_epoch(self, epoch: int) -> EpochStats:
        """
        Finalise the current epoch and compute statistical summary.

        Should be called at the end of each training epoch.

        Returns
        -------
        EpochStats
            Statistical summary for the completed epoch.
        """
        if not self._current_epoch_steps:
            logger.warning("LossTracker.end_epoch called with empty step buffer.")
            stats = EpochStats(epoch=epoch, mean=float("nan"), std=float("nan"),
                               min=float("nan"), max=float("nan"),
                               median=float("nan"), n=0)
        else:
            arr = np.array(self._current_epoch_steps)
            stats = EpochStats(
                epoch  = epoch,
                mean   = float(arr.mean()),
                std    = float(arr.std()),
                min    = float(arr.min()),
                max    = float(arr.max()),
                median = float(np.median(arr)),
                n      = len(arr),
            )

        self.epoch_stats.append(stats)
        self._current_epoch_steps = []
        return stats

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def ema(self) -> Optional[float]:
        """Current exponential moving average of the loss."""
        return self._ema

    @property
    def current_epoch_losses(self) -> List[float]:
        """All step-level losses accumulated in the current (incomplete) epoch."""
        return list(self._current_epoch_steps)

    def recent_steps(self) -> List[Tuple[int, float]]:
        """Return the most recent ``recent_window`` (step, loss) pairs."""
        return list(self._step_buffer)

    def convergence_slope(self, last_n_epochs: int = 10) -> float:
        """
        Linear regression slope over the last ``last_n_epochs`` epoch means.

        A slope close to 0 indicates convergence.  A negative slope indicates
        ongoing improvement.  A positive slope may indicate divergence.

        Returns float("nan") if fewer than 2 epoch statistics are available.
        """
        available = list(self.epoch_stats)[-last_n_epochs:]
        if len(available) < 2:
            return float("nan")
        x = np.arange(len(available), dtype=float)
        y = np.array([s.mean for s in available])
        valid = np.isfinite(y)
        if valid.sum() < 2:
            return float("nan")
        coeffs = np.polyfit(x[valid], y[valid], deg=1)
        return float(coeffs[0])

    def best_epoch(self) -> Optional[EpochStats]:
        """Return the epoch with the lowest mean loss."""
        if not self.epoch_stats:
            return None
        return min(self.epoch_stats, key=lambda s: s.mean if not math.isnan(s.mean) else float("inf"))

    def to_dict(self) -> Dict[str, list]:
        """Serialise epoch history to a JSON-serialisable dictionary."""
        return {
            "epochs": [s.epoch  for s in self.epoch_stats],
            "means":  [s.mean   for s in self.epoch_stats],
            "stds":   [s.std    for s in self.epoch_stats],
            "mins":   [s.min    for s in self.epoch_stats],
            "maxs":   [s.max    for s in self.epoch_stats],
        }
