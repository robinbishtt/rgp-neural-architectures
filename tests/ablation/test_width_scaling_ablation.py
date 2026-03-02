"""tests/ablation/test_width_scaling_ablation.py"""
import numpy as np


class TestWidthScalingAblation:
    def test_power_law_fit_r2(self):
        """Power-law fit should achieve R² > 0.9 on synthetic data."""
        from src.scaling.width_scaling import WidthScalingAnalyzer
        widths = np.array([32, 64, 128, 256, 512], dtype=float)
        obs    = 1.0 + 5.0 * widths ** (-0.5) + 0.01 * np.random.default_rng(0).standard_normal(5)
        result = WidthScalingAnalyzer().fit_power_law(widths, obs)
        assert result.r2 > 0.9

    def test_extrapolation_larger_than_max_obs(self):
        """Infinite-width extrapolation should converge between min and max observable."""
        from src.scaling.width_scaling import WidthScalingAnalyzer
        widths = np.array([32.0, 64.0, 128.0, 256.0])
        obs    = 2.0 + 10.0 / widths
        inf_val = WidthScalingAnalyzer().infinite_width_extrapolation(widths, obs)
        assert obs.min() - 1.0 < inf_val < obs.max() + 1.0
 