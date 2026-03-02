"""tests/validation/test_scaling_law_consistency.py"""
import pytest
import numpy as np


class TestScalingLawConsistency:
    def test_log_coefficient_near_unity(self):
        """H2 prediction: L_min = A·log(ξ_0) + B, A should be near 1."""
        from src.scaling.scaling_law_fitter import ScalingLawFitter
        np.random.seed(0)
        xi_0  = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
        L_min = 1.05 * np.log(xi_0) + 3.0 + 0.05 * np.random.randn(5)
        res   = ScalingLawFitter().fit_logarithmic(xi_0, L_min)
        assert abs(res.coefficients[0] - 1.0) < 0.5, \
            f"Log coefficient A={res.coefficients[0]:.3f} not near 1"
        assert res.r2 > 0.95

    def test_exponent_comparison_consistency(self):
        from src.scaling.exponent_comparison import ExponentComparison
        comp = ExponentComparison()
        result = comp.compare(
            name="log_coeff_A",
            measured=1.05,
            std_error=0.1,
            predicted=1.0,
            n_measurements=10,
        )
        assert result.consistent, f"t-statistic {result.t_statistic:.2f} too large"
 