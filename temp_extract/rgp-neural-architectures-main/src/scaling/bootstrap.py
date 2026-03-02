"""
src/scaling/bootstrap.py

Bootstrap resampling for confidence interval estimation on scaling exponents.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    bootstrap_samples: np.ndarray


class BootstrapConfidence:
    """
    Parametric and non-parametric bootstrap confidence intervals
    for scaling exponents (nu, k_c, ξ₀, etc.).
    """

    def __init__(self, n_bootstrap: int = 1000, ci_level: float = 0.95,
                 seed: int = 42) -> None:
        self.n_bootstrap = n_bootstrap
        self.ci_level    = ci_level
        self.rng         = np.random.default_rng(seed)

    def compute(
        self,
        data: np.ndarray,
        estimator: Callable[[np.ndarray], float],
    ) -> BootstrapResult:
        """
        Non-parametric bootstrap CI for scalar estimator.

        Parameters
        ----------
        data      : (N,) data array
        estimator : function mapping array -> float
        """
        point  = estimator(data)
        n      = len(data)
        samples = np.array([
            estimator(data[self.rng.integers(0, n, size=n)])
            for _ in range(self.n_bootstrap)
        ])
        alpha = 1.0 - self.ci_level
        lo    = float(np.percentile(samples, 100 * alpha / 2))
        hi    = float(np.percentile(samples, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            point_estimate=point,
            ci_lower=lo,
            ci_upper=hi,
            ci_level=self.ci_level,
            bootstrap_samples=samples,
        )

    def paired_bootstrap_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        estimator: Callable[[np.ndarray], float],
    ) -> float:
        """Bootstrap p-value for H0: estimator(A) == estimator(B)."""
        observed_diff = abs(estimator(data_a) - estimator(data_b))
        pooled = np.concatenate([data_a, data_b])
        n_a, n_b = len(data_a), len(data_b)
        count = 0
        for _ in range(self.n_bootstrap):
            perm  = self.rng.permutation(pooled)
            diff  = abs(estimator(perm[:n_a]) - estimator(perm[n_a:n_a + n_b]))
            if diff >= observed_diff:
                count += 1
        return float(count / self.n_bootstrap)
