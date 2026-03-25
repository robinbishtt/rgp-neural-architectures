import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
import numpy as np
from src.rg_flow.operators.operators import StandardRGOperator
def test_gradient_norms_consistent_across_layers():
    depth, width = 20, 32
    layers = nn.ModuleList([StandardRGOperator(width, width) for _ in range(depth)])
    x = torch.randn(4, width)
    norms = []
    h = x
    for layer in layers:
        h = layer(h)
        norms.append(h.detach().norm(dim=-1).mean().item())
    cv = np.std(norms) / (np.mean(norms) + 1e-8)
    assert cv < 2.0, f"Activation norms vary too much: CV={cv:.3f}"