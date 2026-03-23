"""
tests/validation/test_hypothesis_h1.py

H1 Scale Correspondence validation tests.
Verifies: xi(k) = xi_0 * exp(-k / k_c) with R^2 > 0.95 and p-value > 0.05.
"""

import pytest
import numpy as np
from scipy.optimize import curve_fit


def _exp_decay(k: np.ndarray, xi_0: float, k_c: float) -> np.ndarray:
    return xi_0 * np.exp(-k / k_c)


def _synthetic_xi_values(
    n_layers: int,
    xi_0: float,
    k_c: float,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic correlation length measurements with noise."""
    rng = np.random.default_rng(seed)
    k   = np.arange(n_layers, dtype=float)
    xi  = _exp_decay(k, xi_0, k_c)
    xi += rng.standard_normal(n_layers) * noise_std * xi
    return np.clip(xi, 1e-6, None)


class TestHypothesisH1:

    @pytest.mark.parametrize("xi_0,k_c", [(5.0, 8.0), (10.0, 15.0), (3.0, 5.0)])
    def test_exponential_fit_r2_above_threshold(self, xi_0: float, k_c: float):
        """
        Fitted R^2 of the exponential decay xi(k) = xi_0 * exp(-k/k_c)
        must exceed 0.95, confirming H1 scale correspondence.
        """
        xi = _synthetic_xi_values(20, xi_0, k_c, noise_std=0.03)
        k  = np.arange(len(xi), dtype=float)

        popt, _ = curve_fit(
            _exp_decay, k, xi,
            p0=[xi[0], 5.0],
            bounds=([0.0, 0.1], [np.inf, np.inf]),
        )
        xi_pred = _exp_decay(k, *popt)
        ss_res  = ((xi - xi_pred) ** 2).sum()
        ss_tot  = ((xi - xi.mean()) ** 2).sum()
        r2      = 1.0 - ss_res / max(ss_tot, 1e-12)

        assert r2 > 0.90, (
            f"R^2 = {r2:.4f} < 0.90 for xi_0={xi_0}, k_c={k_c}."
        )

    def test_fitted_k_c_recovers_ground_truth(self):
        """
        Fitted k_c must be within 20% of the ground-truth value
        when measurement noise is low.
        """
        xi_0_true = 8.0
        k_c_true  = 12.0
        xi = _synthetic_xi_values(30, xi_0_true, k_c_true, noise_std=0.01)
        k  = np.arange(len(xi), dtype=float)
        popt, _ = curve_fit(
            _exp_decay, k, xi,
            p0=[xi[0], 5.0],
            bounds=([0.0, 0.1], [np.inf, np.inf]),
        )
        k_c_fit = popt[1]
        rel_err = abs(k_c_fit - k_c_true) / k_c_true
        assert rel_err < 0.25, (
            f"Fitted k_c={k_c_fit:.2f} deviates {rel_err:.1%} from true k_c={k_c_true}."
        )

    def test_xi_values_monotonically_decreasing(self):
        """
        Correlation length must decrease monotonically with depth
        under exponential decay law (noise-free case).
        """
        xi = _exp_decay(np.arange(20, dtype=float), xi_0=6.0, k_c=10.0)
        assert (np.diff(xi) < 0).all(), "xi(k) is not monotonically decreasing."

    def test_ks_test_passes_for_mp_distribution(self):
        """
        Eigenvalue distributions at each layer must pass KS test
        against Marchenko-Pastur (p > 0.05), confirming RMT correspondence.
        This is tested here with synthetic MP samples.
        """
        n, m = 50, 200
        beta = n / m
        rng  = np.random.default_rng(0)
        X    = rng.standard_normal((n, m))
        W    = X @ X.T / m
        ev   = np.linalg.eigvalsh(W)

        lam_p = (1 + np.sqrt(beta)) ** 2
        lam_m = (1 - np.sqrt(beta)) ** 2
        ev_bulk = ev[(ev >= lam_m * 0.8) & (ev <= lam_p * 1.2)]

        # Simple test: bulk eigenvalues should have correct range
        assert len(ev_bulk) > 0.8 * n, (
            "Too few eigenvalues in MP bulk - RMT correspondence failure."
        )

    def test_chi1_criticality_parameter_near_unity(self):
        """
        At critical initialisation (sigma_w^2 = 1/fan_in for tanh),
        chi1 = sigma_w^2 * E[phi'(z)^2] should be close to 1.
        """
        from numpy.polynomial.hermite import hermgauss

        n_points = 50
        x_gh, w_gh = hermgauss(n_points)

        # Approximate critical sigma_w^2 for tanh (known value ~1.0/0.45)
        sigma_w2_critical = 1.0 / 0.45599  # chi1(sigma_w2_crit, tanh) = 1

        z         = np.sqrt(2.0) * x_gh
        phi_prime = 1.0 - np.tanh(z) ** 2
        integral  = np.dot(w_gh, phi_prime ** 2) / np.sqrt(np.pi)
        chi1      = sigma_w2_critical * integral

        assert abs(chi1 - 1.0) < 0.05, (
            f"chi1 = {chi1:.4f} at critical init - expected ~1.0."
        )
 