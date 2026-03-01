"""
src/scaling/exponent_comparison.py

ExponentComparison: compares extracted scaling exponents against
theoretical predictions from RGP theory and established universality classes.

Provides statistical tests for whether extracted exponents are consistent
with the predicted values ν = 1, α = 0 (logarithmic scaling).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import t as t_dist


@dataclass
class ExponentComparisonResult:
    exponent_name:  str
    measured:       float
    predicted:      float
    std_error:      float
    t_statistic:    float
    p_value:        float
    consistent:     bool      # |t| < 2.0 (95% CI includes predicted value)
    effect_size:    float     # Cohen's d = (measured - predicted) / std_error


class ExponentComparison:
    """
    Statistical comparison of measured versus predicted scaling exponents.

    Performs one-sample t-tests against theoretical predictions:
        H2: L_min = A · log(ξ₀) + B  →  A ≈ 1 (logarithmic coefficient)
        H1: xi(k) = ξ₀ exp(-k/k_c)  →  k_c determined by χ₁

    Consistency is declared when the measured value falls within the
    95% confidence interval of the theoretical prediction.
    """

    def compare(
        self,
        name:           str,
        measured:       float,
        std_error:      float,
        predicted:      float,
        n_measurements: int = 1,
    ) -> ExponentComparisonResult:
        """
        Compare a single measured exponent against its theoretical prediction.

        Args:
            name:           human-readable name (e.g., "log_coefficient_A")
            measured:       point estimate of the exponent
            std_error:      standard error of the estimate
            predicted:      theoretical prediction to test against
            n_measurements: number of independent measurements (for df)

        Returns:
            ExponentComparisonResult with t-statistic and p-value.
        """
        t_stat = (measured - predicted) / (std_error + 1e-12)
        df     = max(n_measurements - 1, 1)
        p_val  = 2.0 * float(t_dist.sf(abs(t_stat), df=df))
        return ExponentComparisonResult(
            exponent_name=name,
            measured=measured,
            predicted=predicted,
            std_error=std_error,
            t_statistic=float(t_stat),
            p_value=p_val,
            consistent=(abs(t_stat) < 2.0),
            effect_size=float(abs(measured - predicted) / (std_error + 1e-12)),
        )

    def compare_all(
        self,
        measurements: Dict[str, Tuple[float, float]],
        predictions:  Dict[str, float],
        n_measurements: int = 1,
    ) -> List[ExponentComparisonResult]:
        """
        Compare multiple exponents simultaneously.

        Args:
            measurements: dict {name: (measured, std_error)}
            predictions:  dict {name: predicted_value}

        Returns:
            List of ExponentComparisonResult, one per exponent.
        """
        results = []
        for name, (measured, se) in measurements.items():
            pred = predictions.get(name, 0.0)
            results.append(self.compare(name, measured, se, pred, n_measurements))
        return results
