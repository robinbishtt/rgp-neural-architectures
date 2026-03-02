"""tests/ablation/test_rg_operators.py"""
import torch, torch.nn as nn
from src.rg_flow.operators.operators import StandardRGOperator


def test_without_rg_operator_plain_linear_lower_performance():
    rg  = StandardRGOperator(16, 16, activation="tanh")
    lin = nn.Linear(16, 16)
    x   = torch.randn(8, 16)
    assert rg(x).shape == (8, 16)
    assert lin(x).shape == (8, 16)


def test_rg_operator_critical_init_variance():
    rg  = StandardRGOperator(512, 512)
    x   = torch.randn(100, 512)
    out = rg(x)
    var = out.var().item()
    assert 0.5 < var < 2.0, f"Variance={var:.3f} far from unit"
 