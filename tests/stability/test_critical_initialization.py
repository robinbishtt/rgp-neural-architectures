"""tests/stability/test_critical_initialization.py"""
import numpy as np
from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2


def test_chi1_equals_one_at_critical():
    sigma_w_star = critical_sigma_w2("tanh")
    chi1 = chi1_gauss_hermite(sigma_w_star, "tanh")
    assert abs(chi1 - 1.0) < 0.01, f"chi1={chi1:.6f} not near 1.0 at critical init"


def test_lemma_critical_init_passes():
    from src.proofs.lemma_critical_init import run_all_verifications
    results = run_all_verifications()
    assert results["infinite_correlation"]
 