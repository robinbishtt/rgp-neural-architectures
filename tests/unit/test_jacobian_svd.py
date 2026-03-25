import pytest
torch = pytest.importorskip("torch", reason="torch not installed")  
import torch
import numpy as np
from src.core.jacobian.jacobian import AutogradJacobian
import torch.nn as nn
def test_svd_matches_numpy():
    n = 5
    m = nn.Linear(n, n, bias=False)
    nn.init.orthogonal_(m.weight)
    x = torch.randn(n)
    aj = AutogradJacobian()
    J  = aj.compute(m, x).detach().numpy()
    sv_torch = np.linalg.svd(J, compute_uv=False)
    sv_numpy = np.linalg.svd(J, compute_uv=False)
    assert np.allclose(sv_torch, sv_numpy, atol=1e-5)
def test_singular_values_positive():
    m = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    x = torch.randn(4)
    aj = AutogradJacobian()
    J  = aj.compute(m, x).detach().numpy()
    sv = np.linalg.svd(J, compute_uv=False)
    assert (sv >= 0).all()