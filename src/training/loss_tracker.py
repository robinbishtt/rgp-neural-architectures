from __future__ import annotations
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
logger = logging.getLogger(__name__)
@dataclass
class EpochStats:
    epoch:  int
    mean:   float
    std:    float
    min:    float
    max:    float
    median: float
    n:      int
class LossTracker:
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
    def step(self, loss: float, global_step: Optional[int] = None) -> None:
        if global_step is not None:
            self._global_step = global_step
        else:
            self._global_step += 1
        if math.isnan(loss) or math.isinf(loss):
            logger.warning("LossTracker: NaN/Inf loss at step %d. Skipping.", self._global_step)
            return
        self._step_buffer.append((self._global_step, loss))
        self._current_epoch_steps.append(loss)
        if self._ema is None:
            self._ema = loss
        else:
            self._ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self._ema
    def end_epoch(self, epoch: int) -> EpochStats:
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
    @property
    def ema(self) -> Optional[float]:
        return self._ema
    @property
    def current_epoch_losses(self) -> List[float]:
        return list(self._current_epoch_steps)
    def recent_steps(self) -> List[Tuple[int, float]]:
        return list(self._step_buffer)
    def convergence_slope(self, last_n_epochs: int = 10) -> float:
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
        if not self.epoch_stats:
            return None
        return min(self.epoch_stats, key=lambda s: s.mean if not math.isnan(s.mean) else float("inf"))
    def to_dict(self) -> Dict[str, list]:
        return {
            : [s.epoch  for s in self.epoch_stats],
            :  [s.mean   for s in self.epoch_stats],
            :   [s.std    for s in self.epoch_stats],
            :   [s.min    for s in self.epoch_stats],
            :   [s.max    for s in self.epoch_stats],
        }