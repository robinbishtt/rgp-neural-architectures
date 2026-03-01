"""tests/scaling/test_exponential_decay.py — xi(k) decay with R² > 0.95."""
import numpy as np
from src.core.correlation.estimators import ExponentialDecayFitter


def test_noisy_exponential_r2():
    rng = np.random.default_rng(0)
    xi  = 4.0 * np.exp(-np.arange(25) / 7.0) + rng.normal(0, 0.1, 25)
    xi  = np.clip(xi, 0.01, None)
    r   = ExponentialDecayFitter().fit(xi)
    assert r.r2 > 0.90
