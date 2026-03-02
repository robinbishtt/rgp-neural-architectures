"""tests/integration/test_model_to_metrics.py"""
import torch
import torch.nn as nn


def test_model_produces_valid_logits():
    model = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 4))
    x     = torch.randn(8, 16)
    out   = model(x)
    assert out.shape == (8, 4)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
 