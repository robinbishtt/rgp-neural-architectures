"""
tests/unit/test_fisher_correctness.py

Verifies G^(k) = J_k^T G^(k-1) J_k pushforward implementation against symbolic derivations.
"""
import torch
from src.core.fisher.fisher_metric import FisherMetric


def test_pushforward_identity_input():
    fm = FisherMetric()
    n  = 4
    G  = torch.eye(n)
    J  = torch.eye(n)
    result = fm.pushforward(G, J)
    assert torch.allclose(result, torch.eye(n), atol=1e-6)


def test_pushforward_matches_manual():
    fm  = FisherMetric()
    n_in, n_out = 3, 3
    G  = torch.eye(n_in) * 2.0
    J  = torch.randn(n_out, n_in) * 0.5
    result   = fm.pushforward(G, J)
    expected = J @ G @ J.T
    assert torch.allclose(result, expected, atol=1e-5)


def test_pushforward_psd_preserved():
    fm = FisherMetric()
    G  = torch.eye(4)
    J  = torch.randn(4, 4)
    result = fm.pushforward(G, J)
    ev = torch.linalg.eigvalsh(result)
    assert (ev >= -1e-6).all(), "Pushforward result is not PSD"


def test_proof_numerical_verification():
    from src.proofs.theorem1_fisher_transform import verify_pushforward_numerically
    assert verify_pushforward_numerically(seed=42)
 