"""
tests/unit/test_jacobian_chain_rule.py

Confirms cumulative Jacobian equals product of layer-wise Jacobians.
"""
import torch
import torch.nn as nn
from src.core.jacobian.jacobian import AutogradJacobian


def _two_layer_model(n=4):
    m = nn.Sequential(nn.Linear(n, n, bias=False), nn.Tanh(),
                      nn.Linear(n, n, bias=False), nn.Tanh())
    for p in m.parameters():
        nn.init.normal_(p, std=0.5)
    return m


def test_chain_rule_two_layers():
    n = 4
    m = _two_layer_model(n)
    x = torch.randn(n)
    # Full Jacobian
    aj = AutogradJacobian()
    J_full = aj.compute(m, x)
    # Layer-wise product
    x.requires_grad_(True)
    h1 = torch.tanh(m[0](x))
    J1 = aj.compute(lambda h: torch.tanh(m[2](h)), h1.detach())
    h1_clone = x.detach().clone().requires_grad_(True)
    J0 = aj.compute(lambda v: torch.tanh(m[0](v)), h1_clone)
    J_chain = J1 @ J0
    assert torch.allclose(J_full, J_chain, atol=1e-4), "Chain rule violation"
 