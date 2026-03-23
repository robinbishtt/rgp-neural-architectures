"""
tests/unit/test_fisher_correctness.py

Verifies G^(k) = J_k^T G^(k-1) J_k pushforward implementation against symbolic derivations.
"""
import pytest
torch = pytest.importorskip("torch", reason="torch not installed")  # noqa: F811
import torch
from src.core.fisher.fisher_metric import FisherMetric


def test_pullback_identity_input():
    """Verify pullback g^(k) = J^T g^(k-1) J with identity maps."""
    fm = FisherMetric()
    n  = 4
    G  = torch.eye(n)
    J  = torch.eye(n)
    result = fm.pullback(G, J)
    assert torch.allclose(result, torch.eye(n), atol=1e-6)


def test_pullback_matches_manual():
    """Verify pullback J^T G J matches manual computation."""
    fm  = FisherMetric()
    n_in, n_out = 3, 3
    G  = torch.eye(n_in) * 2.0
    J  = torch.randn(n_out, n_in) * 0.5
    result   = fm.pullback(G, J)         # g^(k) = J^T g^(k-1) J
    expected = J.T @ G @ J
    assert torch.allclose(result, expected, atol=1e-5)


def test_pullback_psd_preserved():
    """Pullback J^T G J preserves PSD when G is PSD."""
    fm = FisherMetric()
    G  = torch.eye(4)
    J  = torch.randn(4, 4)
    result = fm.pullback(G, J)   # J^T G J is PSD when G is PSD
    ev = torch.linalg.eigvalsh(result)
    assert (ev >= -1e-6).all(), "Pullback result is not PSD"


def test_pullback_contracts_metric():
    """Verify metric contraction: max eigenvalue of g^(k) ≤ max eigenvalue of g^(k-1).
    This is the empirical test of Theorem 1 (metric contraction).
    """
    fm     = FisherMetric()
    n      = 8
    G_prev = torch.eye(n)
    # Use a contractive Jacobian (singular values < 1)
    J      = torch.randn(n, n) * 0.5 / n**0.5
    G_k    = fm.pullback(G_prev, J)
    eta_prev = torch.linalg.eigvalsh(G_prev).max().item()
    eta_k    = torch.linalg.eigvalsh(G_k).max().item()
    assert eta_k <= eta_prev + 1e-6, (
        f"Metric not contracted: η_prev={eta_prev:.4f}, η_k={eta_k:.4f}"
    )


def test_proof_numerical_verification():
    from src.proofs.theorem1_fisher_transform import verify_pushforward_numerically
    assert verify_pushforward_numerically(seed=42)
 