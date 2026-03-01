"""tests/spectral/test_tracy_widom_edge.py"""
import pytest
import numpy as np


class TestTracyWidomEdge:
    def test_cdf_monotone(self):
        from src.core.spectral.tracy_widom import TracyWidomDistribution
        tw = TracyWidomDistribution(beta=1)
        s  = np.linspace(-5, 3, 50)
        cdf = tw.cdf(s)
        assert np.all(np.diff(cdf) >= -1e-6), "TW CDF must be non-decreasing"

    def test_cdf_boundaries(self):
        from src.core.spectral.tracy_widom import TracyWidomDistribution
        tw = TracyWidomDistribution(beta=1)
        assert tw.cdf(np.array([-8.0]))[0] < 0.05
        assert tw.cdf(np.array([4.0]))[0] > 0.95

    def test_scaled_statistic_finite(self):
        from src.core.spectral.tracy_widom import TracyWidomDistribution
        tw = TracyWidomDistribution(beta=2)
        t  = tw.scaled_statistic(lambda_max=2.1, R=2.0, N=100)
        assert np.isfinite(t)
