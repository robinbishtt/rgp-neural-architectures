import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2
import numpy as np
def test_chi1_equals_one_at_bisection_critical():
    sigma_w2_star = critical_sigma_w2("tanh")
    chi1 = chi1_gauss_hermite(sigma_w2_star, "tanh")
    assert abs(chi1 - 1.0) < 0.01, f"chi1={chi1:.6f} not near 1.0 at critical init"
def test_paper_init_near_critical():
    sigma_w, sigma_b = 1.4, 0.3
    chi1 = chi1_gauss_hermite(sigma_w**2, "tanh")
    assert 0.80 <= chi1 <= 1.05, (
        f"chi1={chi1:.4f} at paper init (sw={sigma_w}, sb={sigma_b}) "
        f"not in near-critical range [0.80, 1.05]"
    )
    assert chi1 <= 1.05, (
        f"Paper init gives chi1={chi1:.4f} > 1 (chaotic phase) - unexpected"
    )
def test_paper_sigma_w_default_in_rg_net():
    import inspect
    from src.architectures.rg_net.rg_net import RGLayer
    sig = inspect.signature(RGLayer.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults.get('sigma_w') == 1.4, (
        f"RGLayer default sigma_w={defaults.get('sigma_w')}, expected 1.4 (paper)"
    )
    assert defaults.get('sigma_b') == 0.3, (
        f"RGLayer default sigma_b={defaults.get('sigma_b')}, expected 0.3 (paper)"
    )
def test_xi_depth_positive_at_paper_init():
    chi1 = chi1_gauss_hermite(1.4**2, "tanh")
    assert chi1 < 1.0, f"chi1={chi1:.4f} >= 1 at paper init - cannot compute xi_depth"
    xi_depth = -1.0 / np.log(chi1)
    assert xi_depth > 0, f"xi_depth={xi_depth:.2f} must be positive"
    assert xi_depth > 1, f"xi_depth={xi_depth:.2f} should be > 1 layer"
def test_relu_critical_at_sqrt2():
    sigma_w2_star = critical_sigma_w2("relu")
    sigma_w_star  = sigma_w2_star ** 0.5
    assert abs(sigma_w_star - 2.0**0.5) < 0.05, (
        f"ReLU critical sigma_w*={sigma_w_star:.4f}, expected sqrt(2)={2**0.5:.4f}"
    )
def test_lemma_critical_init_passes():
    from src.proofs.lemma_critical_init import run_all_verifications
    results = run_all_verifications()
    assert results["infinite_correlation"]
    assert results["all_pass"], f"Lemma verification failed: {results}"