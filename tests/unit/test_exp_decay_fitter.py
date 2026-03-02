"""tests/unit/test_exp_decay_fitter.py"""
import pytest
import numpy as np


class TestExponentialDecayFitter:
    def test_perfect_fit(self):
        from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter
        k   = np.arange(0, 30, dtype=float)
        xi  = 20.0 * np.exp(-k / 10.0)
        res = ExponentialDecayFitter().fit(k, xi)
        assert res.r2 > 0.99
        assert abs(res.xi_0 - 20.0) < 1.0
        assert abs(res.k_c - 10.0) < 1.0

    def test_chi1_in_unit_interval(self):
        from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter
        k   = np.arange(0, 20, dtype=float)
        xi  = 15.0 * np.exp(-k / 8.0)
        res = ExponentialDecayFitter().fit(k, xi)
        assert 0.0 < res.chi1 < 1.0
 