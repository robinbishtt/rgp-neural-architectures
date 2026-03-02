"""
tests/spectral/test_wigner_semicircle.py

Wigner semicircle law for symmetric random matrices.
Tests GOE/GUE eigenvalue bulk universality.
"""

import pytest
import numpy as np


def _goe_eigenvalues(n: int, seed: int = 42) -> np.ndarray:
    """Generate eigenvalues of a Gaussian Orthogonal Ensemble matrix."""
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    H   = (A + A.T) / np.sqrt(2 * n)
    return np.linalg.eigvalsh(H)


class TestWignerSemicircle:

    @pytest.mark.parametrize("n", [100, 200])
    def test_eigenvalues_within_semicircle_support(self, n: int):
        """
        GOE eigenvalues should lie within [-2, 2] (Wigner semicircle radius R=2).
        Allow 5% outliers for finite-n fluctuations.
        """
        ev = _goe_eigenvalues(n)
        fraction_inside = (np.abs(ev) <= 2.1).mean()
        assert fraction_inside >= 0.95, (
            f"Only {fraction_inside:.1%} eigenvalues within semicircle at n={n}."
        )

    def test_empirical_mean_near_zero(self):
        """GOE mean eigenvalue should be zero by symmetry."""
        ev = _goe_eigenvalues(200)
        assert abs(ev.mean()) < 0.1, f"GOE mean = {ev.mean():.4f}, expected ~0."

    def test_semicircle_pdf_shape(self):
        """
        Empirical density maximum should occur near λ=0 (peak of semicircle).
        """
        ev  = _goe_eigenvalues(300)
        hist, edges = np.histogram(ev, bins=40, density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        peak_idx    = hist.argmax()
        # Peak should be in central third of spectrum
        assert abs(bin_centers[peak_idx]) < 0.5, (
            f"Density peak at {bin_centers[peak_idx]:.3f}, expected near 0."
        )

    def test_variance_scales_correctly(self):
        """
        Eigenvalue variance for GOE(n) normalised by 1/sqrt(n) should be O(1).
        """
        n  = 200
        ev = _goe_eigenvalues(n)
        assert 2.0 < ev.var() * n < 6.0, (
            f"GOE eigenvalue variance {ev.var():.4f} does not scale as expected."
        )
 