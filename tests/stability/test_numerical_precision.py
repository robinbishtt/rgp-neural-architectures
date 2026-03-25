import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
def test_fp32_fp64_jacobian_agreement():
    import torch.nn as nn
    from src.core.jacobian.jacobian import AutogradJacobian
    n = 4
    m32 = nn.Linear(n, n)
    m64 = nn.Linear(n, n).double()
    with torch.no_grad():
        m64.weight.copy_(m32.weight.double())
        m64.bias.copy_(m32.bias.double())
    x32 = torch.randn(n)
    x64 = x32.double()
    aj = AutogradJacobian()
    J32 = aj.compute(m32, x32).detach()
    J64 = aj.compute(m64, x64).detach().float()
    assert torch.allclose(J32, J64, atol=1e-3), "FP32/FP64 Jacobians disagree"