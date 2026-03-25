import pytest
torch = pytest.importorskip("torch", reason="torch not installed")  
import torch
class TestVJPJacobian:
    def test_matches_autograd(self):
        from src.core.jacobian.vjp_jacobian import VJPJacobian
        from src.core.jacobian.autograd_jacobian import AutogradJacobian
        torch.manual_seed(1)
        W = torch.randn(3, 5)
        def fn(x):
            return W @ x
        x = torch.randn(5, requires_grad=True)
        assert torch.allclose(VJPJacobian().compute(fn, x),
                              AutogradJacobian().compute(fn, x), atol=1e-5)
    def test_shape(self):
        from src.core.jacobian.vjp_jacobian import VJPJacobian
        x = torch.randn(5, requires_grad=True)
        J = VJPJacobian().compute(lambda x: x * 2, x)
        assert J.shape == (5, 5)