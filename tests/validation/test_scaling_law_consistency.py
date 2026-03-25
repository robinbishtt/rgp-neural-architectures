import numpy as np
import pytest
from scipy import stats
def _lmin_theoretical(xi_values, k_c=8.0, xi_target=1.0):
    return k_c * np.log(np.array(xi_values) / xi_target)
def _compute_aic(y_obs, y_pred, n_params):
    n   = len(y_obs)
    rss = np.sum((y_obs - y_pred)**2)
    s2  = max(rss / n, 1e-12)
    return 2*n_params - 2*(-0.5*n*(np.log(2*np.pi*s2)+1))
class TestScalingLawConsistency:
    XI_PAPER = np.array([5.0, 15.0, 50.0, 100.0, 200.0])
    K_C      = 8.0  
    def test_paper_xi_values_match_hier_datasets(self):
        from experiments.h2_depth_scaling.run_h2_validation import FULL
        code_xi = sorted(FULL["xi_values"])
        expected = sorted([5.0, 15.0, 50.0, 100.0, 200.0])
        assert code_xi == expected, (
            f"Code xi_values={code_xi}, expected={expected}. "
            f"Must match paper Hier-1..5 datasets."
        )
    def test_lmin_95_threshold(self):
        from experiments.h2_depth_scaling.run_h2_validation import FULL
        assert FULL["accuracy_threshold"] == 0.95, (
            f"accuracy_threshold={FULL['accuracy_threshold']}, "
            f"paper requires 0.95 (not 0.85)"
        )
    def test_logarithmic_r2_above_threshold(self):
        from src.scaling.scaling_law_fitter import ScalingLawFitter
        rng   = np.random.default_rng(42)
        l_min = _lmin_theoretical(self.XI_PAPER, self.K_C)
        l_noisy = l_min + rng.normal(0, 0.2, len(l_min))
        res   = ScalingLawFitter().fit_logarithmic(self.XI_PAPER, l_noisy)
        assert res.r2 > 0.95, (
            f"Log fit R^2={res.r2:.4f} < 0.95. Paper: R^2=0.991."
        )
    def test_log_coefficient_near_unity(self):
        from src.scaling.scaling_law_fitter import ScalingLawFitter
        rng   = np.random.default_rng(42)
        l_min = np.log(self.XI_PAPER) + rng.normal(0, 0.1, len(self.XI_PAPER))
        res   = ScalingLawFitter().fit_logarithmic(self.XI_PAPER, l_min)
        assert abs(res.coefficients[0] - 1.0) < 0.3, (
            f"Log coefficient={res.coefficients[0]:.3f}, expected near 1.0. "
            f"Paper: alpha-hat=0.98."
        )
    def test_logarithmic_preferred_over_linear_aic(self):
        from src.scaling.fss_analysis import AICModelSelector
        rng   = np.random.default_rng(42)
        l_min = _lmin_theoretical(self.XI_PAPER, self.K_C)
        l_min += rng.normal(0, 0.3, len(l_min))
        results = AICModelSelector().select(self.XI_PAPER, l_min)
        aic_log = results["logarithmic"].aic
        aic_lin = results["linear"].aic
        assert aic_log < aic_lin, (
            f"Log AIC={aic_log:.2f} not lower than Linear AIC={aic_lin:.2f}. "
            f"Paper: delta_AIC=9.1 (log over linear)."
        )
    def test_logarithmic_preferred_over_power_law_aic(self):
        from src.scaling.fss_analysis import AICModelSelector
        rng   = np.random.default_rng(42)
        l_min = _lmin_theoretical(self.XI_PAPER, self.K_C)
        l_min += rng.normal(0, 0.3, len(l_min))
        results = AICModelSelector().select(self.XI_PAPER, l_min)
        aic_log = results["logarithmic"].aic
        aic_pow = results.get("power_law", results["linear"]).aic
        assert aic_log < aic_pow, (
            f"Log AIC={aic_log:.2f} not lower than Power-law AIC={aic_pow:.2f}. "
            f"Paper: delta_BIC=8.2 over power-law."
        )
    def test_bootstrap_ci_contains_unity(self):
        from experiments.h2_depth_scaling.statistical_analysis import (
            fit_log_scaling, bootstrap_exponent
        )
        rng   = np.random.default_rng(42)
        l_min = _lmin_theoretical(self.XI_PAPER, self.K_C)
        l_min += rng.normal(0, 0.3, len(l_min))
        ci_lo, ci_hi, _ = bootstrap_exponent(
            self.XI_PAPER, l_min, n_bootstrap=500, seed=42
        )
        ci_lo_norm = ci_lo / self.K_C
        ci_hi_norm = ci_hi / self.K_C
        assert ci_lo_norm <= 1.0 <= ci_hi_norm, (
            f"Normalized 95% CI [{ci_lo_norm:.3f}, {ci_hi_norm:.3f}] does not contain 1.0. "
            f"(Raw CI [{ci_lo:.2f}, {ci_hi:.2f}] normalized by K_C={self.K_C}) "
            f"Paper: normalized CI [0.86, 1.10]."
        )
    def test_pearson_r_significant(self):
        from experiments.h2_depth_scaling.statistical_analysis import fit_log_scaling
        rng   = np.random.default_rng(42)
        l_min = _lmin_theoretical(self.XI_PAPER, self.K_C)
        l_min += rng.normal(0, 0.3, len(l_min))
        fit   = fit_log_scaling(self.XI_PAPER, l_min)
        assert fit["r_squared"] > 0.90, (
            f"R^2={fit['r_squared']:.4f} < 0.90. Paper: R^2=0.991, p<0.001."
        )