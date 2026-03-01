"""
tests/unit/test_fisher_symmetry.py

Confirms Fisher matrices G^(k) are symmetric.
"""
import torch
from src.core.fisher.fisher_metric import FisherMetric


def test_pushforward_symmetric():
    fm = FisherMetric()
    G  = torch.eye(5) * 1.5
    J  = torch.randn(5, 5)
    G_k = fm.pushforward(G, J)
    assert torch.allclose(G_k, G_k.T, atol=1e-5), "Pushforward G^(k) is not symmetric"


def test_repeated_pushforward_symmetric():
    fm = FisherMetric()
    G  = torch.eye(4)
    for _ in range(10):
        J = torch.randn(4, 4) * 0.3
        G = fm.pushforward(G, J)
    assert torch.allclose(G, G.T, atol=1e-4)
