"""
tests/spectral/test_level_spacing.py

Nearest-neighbour level spacing distribution — universality class identification.
GOE unfolded spacings follow Wigner surmise: P(s) = (pi/2) s exp(-pi s^2/4).
"""

import pytest
import numpy as np
from scipy.stats import kstest


def _unfold_eigenvalues(ev: np.ndarray, poly_degree: int = 5) -> np.ndarray:
    """
    Spectral unfolding via polynomial fit to the integrated density of states.
    Returns unfolded spacings with unit mean density.
    """
    ev   = np.sort(ev)
    n    = len(ev)
    rank = np.arange(1, n + 1)
    # Fit polynomial to rank vs eigenvalue
    coef = np.polyfit(ev, rank, poly_degree)
    poly = np.poly1d(coef)
    # Unfolded eigenvalues
    unfolded = poly(ev)
    spacings = np.diff(unfolded)
    # Normalise so mean spacing = 1
    spacings = spacings / spacings.mean()
    return spacings[spacings > 0]


def _goe_eigenvalues(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    H   = (A + A.T) / np.sqrt(2 * n)
    return np.linalg.eigvalsh(H)


def _wigner_surmise_cdf(s: np.ndarray) -> np.ndarray:
    """CDF of Wigner surmise: P(s) = (pi/2) s exp(-pi s^2/4)."""
    return 1.0 - np.exp(-np.pi * s**2 / 4.0)


class TestLevelSpacing:

    def test_spacings_non_negative(self):
        """All unfolded spacings must be non-negative."""
        ev = _goe_eigenvalues(200)
        spacings = _unfold_eigenvalues(ev)
        assert (spacings >= 0).all(), "Negative unfolded spacings detected."

    def test_spacings_mean_near_unity(self):
        """Unfolded spacings must have mean close to 1 by construction."""
        ev = _goe_eigenvalues(300)
        spacings = _unfold_eigenvalues(ev)
        assert abs(spacings.mean() - 1.0) < 0.1, (
            f"Mean spacing {spacings.mean():.3f} != 1."
        )

    @pytest.mark.parametrize("n", [200, 400])
    def test_goe_ks_wigner_surmise(self, n: int):
        """
        KS test of GOE unfolded spacings against Wigner surmise CDF.
        Cannot reject null at p > 0.05.
        """
        ev = _goe_eigenvalues(n, seed=n)
        spacings = _unfold_eigenvalues(ev)
        stat, pval = kstest(spacings, _wigner_surmise_cdf)
        assert pval > 0.01, (
            f"KS test against Wigner surmise: stat={stat:.4f}, p={pval:.4f} at n={n}."
        )

    def test_level_repulsion(self):
        """
        GOE eigenvalues exhibit level repulsion: P(s→0) → 0.
        Near-zero spacings must be rare.
        """
        ev = _goe_eigenvalues(400)
        spacings = _unfold_eigenvalues(ev)
        fraction_near_zero = (spacings < 0.1).mean()
        # Under Wigner surmise, P(s<0.1) ≈ 0.008 — much less than Poisson (P=1-e^{-0.1}≈0.1)
        assert fraction_near_zero < 0.05, (
            f"Level repulsion violated: {fraction_near_zero:.1%} of spacings < 0.1."
        )
