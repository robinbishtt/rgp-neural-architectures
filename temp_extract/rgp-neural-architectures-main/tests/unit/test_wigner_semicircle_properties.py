"""tests/unit/test_wigner_semicircle_properties.py"""
import pytest
import numpy as np


class TestWignerSemicircle:
    def test_pdf_nonneg_in_support(self):
        from src.core.spectral.wigner_semicircle import WignerSemicircleDistribution
        wsc = WignerSemicircleDistribution(R=2.0)
        lam = np.linspace(-2, 2, 100)
        assert np.all(wsc.pdf(lam) >= 0)

    def test_pdf_zero_outside_support(self):
        from src.core.spectral.wigner_semicircle import WignerSemicircleDistribution
        wsc = WignerSemicircleDistribution(R=2.0)
        assert wsc.pdf(np.array([3.0]))[0] == 0.0
        assert wsc.pdf(np.array([-3.0]))[0] == 0.0

    def test_cdf_monotone(self):
        from src.core.spectral.wigner_semicircle import WignerSemicircleDistribution
        wsc = WignerSemicircleDistribution(R=2.0)
        x   = np.linspace(-2.0, 2.0, 50)
        cdf = wsc.cdf(x)
        assert np.all(np.diff(cdf) >= -1e-8)
