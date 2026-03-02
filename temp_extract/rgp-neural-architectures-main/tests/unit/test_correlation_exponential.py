"""
tests/unit/test_correlation_exponential.py

Checks xi(k) decay law against theoretical predictions.
"""
import numpy as np
from src.core.correlation.estimators import ExponentialDecayFitter


def test_perfect_exponential_recovery():
    xi0, kc = 5.0, 8.0
    k  = np.arange(30)
    xi = xi0 * np.exp(-k / kc)
    fitter = ExponentialDecayFitter()
    result = fitter.fit(xi)
    assert abs(result.xi_0 - xi0) < 0.1
    assert abs(result.k_c  - kc)  < 0.1
    assert result.r2 > 0.999


def test_r2_threshold_noisy():
    rng = np.random.default_rng(42)
    xi0, kc = 4.0, 10.0
    k  = np.arange(30)
    xi = xi0 * np.exp(-k / kc) + rng.normal(0, 0.05, 30)
    fitter = ExponentialDecayFitter()
    result = fitter.fit(xi)
    assert result.r2 > 0.95
