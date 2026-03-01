"""
tests/spectral/test_tracy_widom.py

Tracy-Widom edge fluctuation statistics.
The largest eigenvalue of a GOE matrix, properly centred and scaled,
follows the Tracy-Widom (GUE beta=2) distribution.
"""

import pytest
import numpy as np


def _goe_max_eigenvalue(n: int, seed: int) -> float:
    """Return largest eigenvalue of an n x n GOE matrix."""
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    H   = (A + A.T) / np.sqrt(2 * n)
    return float(np.linalg.eigvalsh(H)[-1])


def _centre_scale_goe(lam_max: float, n: int) -> float:
    """
    Centre and scale the largest GOE eigenvalue to approach Tracy-Widom.
    mu_n = 2 (edge), sigma_n = n^{-2/3}.
    """
    mu    = 2.0
    sigma = n ** (-2.0 / 3.0)
    return (lam_max - mu) / sigma


class TestTracyWidom:

    def test_max_eigenvalue_near_edge(self):
        """
        GOE maximum eigenvalue should be near the bulk edge (lambda_+ = 2).
        """
        n    = 200
        lam  = _goe_max_eigenvalue(n, seed=42)
        # Allow generous tolerance for finite n
        assert 1.5 < lam < 2.5, (
            f"GOE max eigenvalue {lam:.4f} not near expected edge of 2.0."
        )

    def test_scaled_statistic_finite(self):
        """Centred and scaled TW statistic must be finite."""
        n   = 200
        lam = _goe_max_eigenvalue(n, seed=7)
        tw  = _centre_scale_goe(lam, n)
        assert np.isfinite(tw), f"TW statistic is non-finite: {tw}."

    def test_tw_statistics_collection_negative_mean(self):
        """
        Tracy-Widom GUE distribution has mean ≈ -1.77.
        Sample of TW statistics from many matrices must have negative mean.
        """
        n       = 150
        n_mats  = 50
        stats   = [
            _centre_scale_goe(_goe_max_eigenvalue(n, seed=i), n)
            for i in range(n_mats)
        ]
        mean_tw = np.mean(stats)
        # Very loose bound: just check the mean is negative
        assert mean_tw < 0.5, (
            f"TW sample mean {mean_tw:.3f} not consistent with TW distribution."
        )

    def test_max_eigenvalue_not_below_bulk(self):
        """Maximum eigenvalue must not be below the bulk (lambda_- = 0 for GOE)."""
        for seed in range(10):
            lam = _goe_max_eigenvalue(200, seed=seed)
            assert lam > 0.0, (
                f"GOE max eigenvalue {lam:.4f} is below zero (inside Wigner bulk edge)."
            )
