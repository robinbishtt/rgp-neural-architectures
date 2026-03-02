"""tests/ablation/test_learned_operator_ablation.py"""
import torch


class TestLearnedOperatorAblation:
    def test_learned_adapts_to_different_inputs(self):
        """Learned operator should produce different outputs for different input statistics."""
        from src.rg_flow.operators.learned_rg_operator import LearnedRGOperator
        torch.manual_seed(2)
        op  = LearnedRGOperator(in_dim=16, out_dim=16)
        x1  = torch.randn(4, 16) * 0.1   # low variance input
        x2  = torch.randn(4, 16) * 5.0   # high variance input
        with torch.no_grad():
            y1 = op(x1)
            y2 = op(x2)
        # Outputs should differ (hyper-network adapts to statistics)
        assert not torch.allclose(y1, y2, atol=1e-3)

    def test_gradient_through_hypernetwork(self):
        from src.rg_flow.operators.learned_rg_operator import LearnedRGOperator
        op = LearnedRGOperator(in_dim=8, out_dim=8)
        x  = torch.randn(2, 8, requires_grad=True)
        op(x).sum().backward()
        assert x.grad is not None
 