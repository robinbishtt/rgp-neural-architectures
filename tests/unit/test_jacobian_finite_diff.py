"""tests/unit/test_jacobian_finite_diff.py"""
import pytest
import torch


class TestFiniteDiffJacobian:
    def test_matches_autograd(self):
        from src.core.jacobian.finite_difference_jacobian import FiniteDifferenceJacobian
        from src.core.jacobian.autograd_jacobian import AutogradJacobian
        torch.manual_seed(2)
        W = torch.randn(3, 4).double()
        fn = lambda x: W @ x
        x = torch.randn(4).double()
        J_fd = FiniteDifferenceJacobian(eps=1e-6).compute(fn, x)
        J_ad = AutogradJacobian().compute(fn, x)
        assert torch.allclose(J_fd.double(), J_ad.double(), atol=1e-4)

    def test_relative_error_small(self):
        from src.core.jacobian.finite_difference_jacobian import FiniteDifferenceJacobian
        from src.core.jacobian.autograd_jacobian import AutogradJacobian
        fn = lambda x: x ** 2
        x = torch.linspace(0.5, 2.0, 4)
        fdjac = FiniteDifferenceJacobian(eps=1e-5)
        err = fdjac.relative_error(AutogradJacobian().compute(fn, x), fn=fn, x=x)
        assert err < 0.01
 