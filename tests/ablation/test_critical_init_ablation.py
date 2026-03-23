"""tests/ablation/test_critical_init_ablation.py"""
import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
from src.rg_flow.operators.operators import StandardRGOperator


def test_standard_init_vs_critical_gradient_norm():
    depth, width = 20, 64
    critical_layers = [StandardRGOperator(width, width, sigma_w=1.0) for _ in range(depth)]
    nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])

    x = torch.randn(1, width, requires_grad=True)

    h = x.clone().requires_grad_(True)
    for l in critical_layers:
        h = l(h)
    h.sum().backward()
    g_critical = x.grad.norm().item() if x.grad else 0.0

    assert g_critical > 1e-8
 