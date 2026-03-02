"""tests/scaling/test_collapse_quality.py - Additional collapse quality tests."""
import pytest
import numpy as np


class TestCollapseQualityExtended:
    def test_perfect_collapse_high_q(self):
        """Perfectly overlapping curves should give high Q-value."""
        from src.scaling.collapse_quality import CollapseQualityMetrics
        x = np.linspace(0, 5, 20)
        y = np.sin(x)
        result = CollapseQualityMetrics().evaluate(
            [x, x], [y, y + 1e-6 * np.random.default_rng(0).standard_normal(20)]
        )
        assert result.q_value > 0.05

    def test_poor_collapse_low_q(self):
        """Misaligned curves should give low Q-value."""
        from src.scaling.collapse_quality import CollapseQualityMetrics
        x = np.linspace(0, 5, 20)
        y1 = np.sin(x)
        y2 = np.cos(x) * 3.0
        result = CollapseQualityMetrics(q_threshold=0.1).evaluate([x, x], [y1, y2])
        assert result.mean_residual > 0.1
