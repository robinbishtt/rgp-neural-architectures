"""tests/scaling/test_scaling_law_fit.py"""
import numpy as np


class TestScalingLawFit:
    def test_log_fit_perfect_data(self):
        from src.scaling.scaling_law_fitter import ScalingLawFitter
        xi = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        L_min = 1.0 * np.log(xi) + 2.0
        res = ScalingLawFitter().fit_logarithmic(xi, L_min)
        assert res.r2 > 0.99
        assert abs(res.coefficients[0] - 1.0) < 0.1  # A ≈ 1

    def test_power_law_fit(self):
        from src.scaling.scaling_law_fitter import ScalingLawFitter
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        y = 3.0 * x ** 0.5
        res = ScalingLawFitter().fit_power_law(x, y)
        assert res.r2 > 0.99
 