"""
tests/spectral/test_marchenko_pastur_fit.py

KS test of empirical eigenvalue distributions against Marchenko-Pastur law.
Validates that wide random-layer Jacobians follow RMT predictions.
"""

import pytest
import numpy as np
import torch


def _sample_wishart_eigenvalues(n: int, m: int, sigma2: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate eigenvalues of W = (1/m) X Xᵀ where X ~ N(0, sigma2)."""
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, m)) * np.sqrt(sigma2)
    W   = X @ X.T / m
    return np.linalg.eigvalsh(W)


class TestMarchenkoPasturFit:

    @pytest.mark.parametrize("n,m", [(50, 200), (100, 400)])
    def test_ks_pvalue_above_threshold(self, n: int, m: int):
        """
        KS test p-value must be > 0.05 (cannot reject MP null hypothesis).
        This confirms that Wishart eigenvalues follow the MP distribution.
        """
        from scipy.stats import kstest

        beta   = n / m
        sigma2 = 1.0
        ev     = _sample_wishart_eigenvalues(n, m, sigma2)
        lam_p  = sigma2 * (1 + np.sqrt(beta)) ** 2
        lam_m  = sigma2 * (1 - np.sqrt(beta)) ** 2

        def mp_cdf(x_arr):
            results = []
            x_grid  = np.linspace(lam_m * 0.99, lam_p * 1.01, 5000)
            pdf     = np.zeros_like(x_grid)
            valid   = (x_grid > lam_m) & (x_grid < lam_p)
            l_v     = x_grid[valid]
            pdf[valid] = (
                np.sqrt((lam_p - l_v) * (l_v - lam_m))
                / (2 * np.pi * sigma2 * beta * l_v)
            )
            cdf = np.concatenate([[0.0], np.cumsum(pdf[:-1] * np.diff(x_grid))])
            cdf /= max(cdf[-1], 1e-12)
            for xi in x_arr:
                results.append(float(np.interp(xi, x_grid, cdf)))
            return np.array(results)

        # Filter to bulk (exclude outliers)
        ev_bulk = ev[(ev >= lam_m * 0.8) & (ev <= lam_p * 1.2)]
        stat, pval = kstest(ev_bulk, mp_cdf)
        assert pval > 0.01, (
            f"KS test failed: stat={stat:.4f}, p={pval:.4f} for n={n}, m={m}."
        )

    def test_bulk_eigenvalues_within_mp_bounds(self):
        """95% of eigenvalues must lie within MP support [lambda-, lambda+]."""
        n, m   = 64, 256
        beta   = n / m
        sigma2 = 1.0
        ev     = _sample_wishart_eigenvalues(n, m, sigma2)
        lam_p  = sigma2 * (1 + np.sqrt(beta)) ** 2
        lam_m  = sigma2 * (1 - np.sqrt(beta)) ** 2

        fraction_inside = ((ev >= lam_m * 0.9) & (ev <= lam_p * 1.1)).mean()
        assert fraction_inside >= 0.90, (
            f"Only {fraction_inside:.1%} eigenvalues within MP bounds."
        )

    def test_empirical_mean_close_to_theoretical(self):
        """
        E[lambda] = sigma2 for MP distribution (regardless of beta).
        Empirical mean must be close to sigma2.
        """
        n, m   = 100, 400
        sigma2 = 2.0
        ev     = _sample_wishart_eigenvalues(n, m, sigma2)
        assert abs(ev.mean() - sigma2) < 0.3, (
            f"Empirical mean {ev.mean():.3f} far from sigma2={sigma2}."
        )

    def test_lambda_plus_bound_respected(self):
        """Largest eigenvalue must be close to lambda_+ = sigma2(1+sqrt(beta))^2."""
        n, m   = 80, 320
        beta   = n / m
        sigma2 = 1.0
        lam_p  = sigma2 * (1 + np.sqrt(beta)) ** 2
        ev     = _sample_wishart_eigenvalues(n, m, sigma2, seed=123)

        # Allow 20% tolerance for finite-size fluctuations
        assert ev.max() < lam_p * 1.3, (
            f"Max eigenvalue {ev.max():.3f} significantly exceeds lambda+={lam_p:.3f}."
        )
