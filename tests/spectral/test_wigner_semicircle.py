import pytest
import numpy as np
def _goe_eigenvalues(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    H   = (A + A.T) / np.sqrt(2 * n)
    return np.linalg.eigvalsh(H)
class TestWignerSemicircle:
    @pytest.mark.parametrize("n", [100, 200])
    def test_eigenvalues_within_semicircle_support(self, n: int):
        ev = _goe_eigenvalues(n)
        fraction_inside = (np.abs(ev) <= 2.1).mean()
        assert fraction_inside >= 0.95, (
            f"Only {fraction_inside:.1%} eigenvalues within semicircle at n={n}."
        )
    def test_empirical_mean_near_zero(self):
        ev = _goe_eigenvalues(200)
        assert abs(ev.mean()) < 0.1, f"GOE mean = {ev.mean():.4f}, expected ~0."
    def test_semicircle_pdf_shape(self):
        ev  = _goe_eigenvalues(1000)
        hist, edges = np.histogram(ev, bins=40, density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        peak_idx    = hist.argmax()
        assert abs(bin_centers[peak_idx]) < 0.8, (
            f"Density peak at {bin_centers[peak_idx]:.3f}, expected near 0."
        )
    def test_variance_scales_correctly(self):
        n  = 200
        ev = _goe_eigenvalues(n)
        assert 0.5 < ev.var() < 2.0, (
            f"GOE eigenvalue variance {ev.var():.4f} should be near 1.0 (semicircle variance)"
        )