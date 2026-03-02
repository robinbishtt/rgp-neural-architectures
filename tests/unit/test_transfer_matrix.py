"""tests/unit/test_transfer_matrix.py"""
import pytest
import torch
import numpy as np


class TestTransferMatrix:
    def test_returns_positive_xi(self):
        from src.core.correlation.transfer_matrix import TransferMatrixMethod
        tm = TransferMatrixMethod()
        W  = 0.5 * torch.eye(8)
        xi = tm.compute_from_jacobian(W)
        assert xi > 0.0

    def test_contracting_gives_finite_xi(self):
        from src.core.correlation.transfer_matrix import TransferMatrixMethod
        tm = TransferMatrixMethod()
        torch.manual_seed(0)
        W  = torch.randn(6, 6) * 0.3
        xi = tm.compute_from_jacobian(W)
        assert np.isfinite(xi)

    def test_gap_ratio_in_unit_interval(self):
        from src.core.correlation.transfer_matrix import TransferMatrixMethod
        tm = TransferMatrixMethod()
        W  = torch.randn(8, 8)
        gr = tm.gap_ratio(W, k=2)
        assert 0.0 <= gr <= 1.0
 