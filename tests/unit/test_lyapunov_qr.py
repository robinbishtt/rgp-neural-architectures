"""
tests/unit/test_lyapunov_qr.py

Verifies QR algorithm convergence and Lyapunov exponent accuracy.
"""
import numpy as np
from src.core.lyapunov.lyapunov import StandardQRAlgorithm, detect_regime


def _identity_jacobians(n: int = 4, depth: int = 50):
    return [np.eye(n) for _ in range(depth)]


def _stable_jacobians(n: int = 4, scale: float = 0.9, depth: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    Js = []
    for _ in range(depth):
        A = rng.standard_normal((n, n)) * scale / np.sqrt(n)
        Js.append(A)
    return Js


def test_identity_jacobians_zero_exponents():
    algo = StandardQRAlgorithm()
    exps = algo.compute(_identity_jacobians(), n_exponents=4)
    # For identity maps, all exponents should be ~0
    assert np.allclose(exps, 0.0, atol=0.5)


def test_stable_network_negative_mle():
    algo = StandardQRAlgorithm()
    exps = algo.compute(_stable_jacobians(scale=0.5), n_exponents=4)
    assert exps[0] < 0, "MLE should be negative for stable (ordered) network"


def test_regime_detection():
    assert detect_regime(np.array([-0.5, -1.0, -2.0])) == "ordered"
    assert detect_regime(np.array([0.5, -0.3, -1.0])) == "chaotic"
    assert detect_regime(np.array([0.01, -0.01, -0.1])) == "critical"
 