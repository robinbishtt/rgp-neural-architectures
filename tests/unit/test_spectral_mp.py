"""
tests/unit/test_spectral_mp.py

Validates Marchenko-Pastur RMT predictions for wide network Jacobians.
"""
import numpy as np
import pytest
from src.core.spectral.spectral import MarchenkoPasturDistribution


def test_mp_pdf_integrates_to_one():
    mp   = MarchenkoPasturDistribution(beta=0.5, sigma2=1.0)
    x    = np.linspace(mp.lam_minus * 0.99, mp.lam_plus * 1.01, 5000)
    rho  = mp.pdf(x)
    area = np.trapz(rho, x)
    assert abs(area - 1.0) < 0.01, f"MP PDF area={area:.4f} != 1.0"


def test_mp_lam_pm_correct():
    sigma2 = 1.0
    for beta in [0.25, 0.5, 1.0, 2.0]:
        mp  = MarchenkoPasturDistribution(beta=beta, sigma2=sigma2)
        lp  = sigma2 * (1 + np.sqrt(beta)) ** 2
        lm  = sigma2 * (1 - np.sqrt(beta)) ** 2
        assert abs(mp.lam_plus  - lp) < 1e-8
        assert abs(mp.lam_minus - lm) < 1e-8


def test_mp_ks_test_on_wishart_sample():
    mp  = MarchenkoPasturDistribution(beta=0.5)
    ev  = mp.sample_wishart(n=200, m=400, rng=np.random.default_rng(0))
    stat, pval = mp.ks_test(ev)
    assert pval > 0.01, f"KS test failed: p={pval:.4f}"
 