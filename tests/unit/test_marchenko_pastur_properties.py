"""tests/unit/test_marchenko_pastur_properties.py"""
import pytest
import numpy as np


class TestMarchenkoPasturProperties:
    def test_pdf_nonnegative(self):
        from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution
        mp = MarchenkoPasturDistribution(beta=0.5)
        lam = np.linspace(-1, 5, 200)
        assert np.all(mp.pdf(lam) >= 0)

    def test_pdf_zero_outside_support(self):
        from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution
        mp = MarchenkoPasturDistribution(beta=0.5)
        assert mp.pdf(np.array([mp.lam_plus + 1.0]))[0] == 0.0
        assert mp.pdf(np.array([-0.5]))[0] == 0.0

    def test_cdf_endpoints(self):
        from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution
        mp = MarchenkoPasturDistribution(beta=0.5)
        assert mp.cdf(np.array([mp.lam_minus - 0.1]))[0] < 0.01
        assert mp.cdf(np.array([mp.lam_plus + 0.1]))[0] > 0.99

    def test_ks_test_on_wishart_sample(self):
        from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution
        mp = MarchenkoPasturDistribution(beta=0.5)
        evs = mp.sample_wishart(200, 400, rng=np.random.default_rng(0))
        stat, pval = mp.ks_test(evs)
        assert pval > 0.01, f"KS test p-value too low: {pval:.4f}"
 