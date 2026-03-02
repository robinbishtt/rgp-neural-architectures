"""tests/unit/test_jacobian_jvp.py"""
import torch


class TestJVPJacobian:
    def test_matches_autograd_linear(self):
        from src.core.jacobian.jvp_jacobian import JVPJacobian
        from src.core.jacobian.autograd_jacobian import AutogradJacobian
        torch.manual_seed(0)
        W = torch.randn(4, 6)
        def fn(x):
            return x @ W.t()
        x = torch.randn(6)
        assert torch.allclose(JVPJacobian().compute(fn, x),
                              AutogradJacobian().compute(fn, x), atol=1e-5)

    def test_identity_jacobian(self):
        from src.core.jacobian.jvp_jacobian import JVPJacobian
        def fn(x):
            return x
        x = torch.randn(4)
        J = JVPJacobian().compute(fn, x)
        assert torch.allclose(J, torch.eye(4), atol=1e-5)
 