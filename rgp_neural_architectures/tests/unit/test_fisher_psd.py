"""
tests/unit/test_fisher_psd.py

Verifies positive semi-definite property of Fisher matrices.
"""
import torch
from src.core.fisher.fisher_metric import FisherMetric


def test_pushforward_psd():
    fm = FisherMetric()
    G  = torch.eye(6)
    J  = torch.randn(6, 6) * 0.5
    G_k = fm.pushforward(G, J)
    ev  = torch.linalg.eigvalsh(G_k)
    assert (ev >= -1e-5).all()


def test_eigenvalue_clipping_maintains_psd():
    fm  = FisherMetric(clip_eigenvalues=True, min_eigenvalue=1e-8)
    G   = -torch.eye(4) * 0.1  # deliberately negative
    J   = torch.eye(4)
    G_k = fm.pushforward(G, J)
    ev  = torch.linalg.eigvalsh(G_k)
    assert (ev >= 0).all()
