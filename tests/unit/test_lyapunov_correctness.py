"""tests/unit/test_lyapunov_correctness.py"""
import pytest
import numpy as np
import torch


class TestLyapunovCorrectness:
    def test_identity_map_zero_mle(self):
        from src.core.lyapunov.standard_qr import StandardQRAlgorithm
        J = [torch.eye(6) for _ in range(30)]
        result = StandardQRAlgorithm(reortho_interval=5, n_warmup=0).compute(J)
        assert abs(result.mle) < 0.2

    def test_contracting_map_ordered(self):
        from src.core.lyapunov.standard_qr import StandardQRAlgorithm
        J = [0.5 * torch.eye(6) for _ in range(40)]
        result = StandardQRAlgorithm(reortho_interval=5, n_warmup=0).compute(J)
        assert result.mle < 0

    def test_exponents_descending(self):
        from src.core.lyapunov.standard_qr import StandardQRAlgorithm
        torch.manual_seed(0)
        J = [0.8 * torch.randn(5, 5) for _ in range(25)]
        result = StandardQRAlgorithm().compute(J)
        exps = result.exponents
        assert np.all(exps[:-1] >= exps[1:] - 1e-8)

    def test_kaplan_yorke_nonnegative(self):
        from src.core.lyapunov.standard_qr import StandardQRAlgorithm
        torch.manual_seed(1)
        J = [0.9 * torch.randn(4, 4) for _ in range(20)]
        result = StandardQRAlgorithm().compute(J)
        assert result.kaplan_yorke_dim >= 0.0

    def test_adaptive_qr_finite_mle(self):
        from src.core.lyapunov.adaptive_qr import AdaptiveQRAlgorithm
        torch.manual_seed(2)
        J = [0.95 * torch.randn(4, 4) for _ in range(20)]
        result = AdaptiveQRAlgorithm().compute(J)
        assert np.isfinite(result.mle)
