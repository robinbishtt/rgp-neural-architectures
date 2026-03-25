import pytest
import numpy as np
from scipy.stats import pearsonr
def _compute_lmin(xi_0: float, xi_target: float, k_c: float) -> float:
    return k_c * np.log(xi_0 / xi_target)
def _synthetic_lmin_data(
    xi_values: list,
    k_c: float = 8.0,
    xi_target: float = 1.0,
    noise_std: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    lmin = np.array([_compute_lmin(xi, xi_target, k_c) for xi in xi_values])
    lmin += rng.standard_normal(len(lmin)) * noise_std
    return np.clip(lmin, 1.0, None)
class TestHypothesisH2:
    def test_lmin_log_linearity(self):
        xi_values = [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
        lmin      = _synthetic_lmin_data(xi_values, k_c=8.0, noise_std=0.3)
        log_xi    = np.log(xi_values)
        r, pval = pearsonr(log_xi, lmin)
        assert r > 0.90, f"L_min vs log(xi) Pearson r = {r:.4f} < 0.90."
    def test_fitted_slope_recovers_k_c(self):
        k_c_true  = 10.0
        xi_target = 1.0
        xi_values = np.array([3.0, 5.0, 8.0, 12.0, 18.0, 25.0])
        lmin      = np.array([_compute_lmin(xi, xi_target, k_c_true) for xi in xi_values])
        log_xi    = np.log(xi_values)
        A      = np.vstack([log_xi, np.ones(len(log_xi))]).T
        coef,  = np.linalg.lstsq(A, lmin, rcond=None)[:1]
        k_c_fit = coef[0]
        rel_err = abs(k_c_fit - k_c_true) / k_c_true
        assert rel_err < 0.20, (
            f"Fitted k_c = {k_c_fit:.2f} deviates {rel_err:.1%} from true {k_c_true}."
        )
    def test_lmin_increases_with_xi(self):
        xi_values = sorted([2.0, 5.0, 10.0, 20.0, 40.0])
        k_c       = 8.0
        xi_target = 1.0
        lmin      = np.array([_compute_lmin(xi, xi_target, k_c) for xi in xi_values])
        assert (np.diff(lmin) > 0).all(), "L_min not monotonically increasing with xi."
    @pytest.mark.parametrize("k_c", [5.0, 10.0, 15.0])
    def test_aic_prefers_log_model(self, k_c: float):
        xi_values = np.array([3.0, 5.0, 8.0, 12.0, 18.0, 25.0])
        lmin      = np.array([_compute_lmin(xi, 1.0, k_c) for xi in xi_values])
        n         = len(xi_values)
        log_xi    = np.log(xi_values)
        A_log     = np.vstack([log_xi, np.ones(n)]).T
        coef_log, res_log = np.linalg.lstsq(A_log, lmin, rcond=None)[:2]
        if len(res_log) == 0:
            res_log = [((lmin - A_log @ coef_log) ** 2).sum()]
        aic_log   = n * np.log(res_log[0] / n) + 2 * 2
        A_lin     = np.vstack([xi_values, np.ones(n)]).T
        coef_lin, res_lin = np.linalg.lstsq(A_lin, lmin, rcond=None)[:2]
        if len(res_lin) == 0:
            res_lin = [((lmin - A_lin @ coef_lin) ** 2).sum()]
        aic_lin   = n * np.log(res_lin[0] / n) + 2 * 2
        assert aic_log < aic_lin, (
            f"AIC_log={aic_log:.2f} not < AIC_lin={aic_lin:.2f} at k_c={k_c}."
        )
    def test_lmin_positive_for_xi_above_target(self):
        xi_values = [2.0, 5.0, 10.0, 20.0]
        lmin      = [_compute_lmin(xi, xi_target=1.0, k_c=8.0) for xi in xi_values]
        assert all(l > 0 for l in lmin), "L_min is non-positive for xi > xi_target."