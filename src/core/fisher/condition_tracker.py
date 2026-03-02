"""
src/core/fisher/condition_tracker.py

Tracks Fisher metric condition number through network depth.
Condition number κ = λ_max / λ_min is a key indicator of training difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class ConditionHistory:
    layer_indices: List[int] = field(default_factory=list)
    condition_numbers: List[float] = field(default_factory=list)
    lambda_max: List[float] = field(default_factory=list)
    lambda_min: List[float] = field(default_factory=list)

    def max_condition(self) -> float:
        return max(self.condition_numbers) if self.condition_numbers else 0.0

    def is_ill_conditioned(self, threshold: float = 1e6) -> bool:
        return self.max_condition() > threshold


class FisherConditionTracker:
    """
    Monitors metric condition number κ(G^(k)) = λ_max / λ_min through depth.

    A well-conditioned Fisher metric (κ ≈ 1) indicates uniform information
    flow; ill-conditioning signals dominant/degenerate parameter directions.
    """

    def __init__(
        self,
        warning_threshold: float = 1e6,
        clip_min: float = 1e-10,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.clip_min = clip_min
        self._history = ConditionHistory()

    def record(self, layer_idx: int, G: torch.Tensor) -> float:
        """
        Compute and record condition number of G at layer_idx.

        Returns the condition number κ.
        """
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, self.clip_min, None)
        lmin = float(ev[0])
        lmax = float(ev[-1])
        kappa = lmax / max(lmin, self.clip_min)

        self._history.layer_indices.append(layer_idx)
        self._history.condition_numbers.append(kappa)
        self._history.lambda_max.append(lmax)
        self._history.lambda_min.append(lmin)

        return kappa

    def get_history(self) -> ConditionHistory:
        return self._history

    def reset(self) -> None:
        self._history = ConditionHistory()

    def condition_at_layer(self, layer_idx: int) -> Optional[float]:
        try:
            pos = self._history.layer_indices.index(layer_idx)
            return self._history.condition_numbers[pos]
        except ValueError:
            return None

    def decay_rate(self) -> Optional[float]:
        """
        Estimate exponential growth rate of condition number with depth.
        Returns log(κ_final / κ_initial) / n_layers if monotone, else None.
        """
        kappas = self._history.condition_numbers
        if len(kappas) < 2:
            return None
        return float(np.log(kappas[-1] / max(kappas[0], 1.0)) / len(kappas))
 