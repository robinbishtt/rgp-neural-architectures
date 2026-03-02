"""tests/ablation/test_attention_operator_ablation.py"""
import pytest
import torch
import torch.nn as nn


class TestAttentionOperatorAblation:
    def test_attention_vs_standard_output_differs(self):
        """Attention and standard RG operators should produce different outputs."""
        from src.rg_flow.operators.operators import StandardRGOperator
        from src.rg_flow.operators.attention_rg_operator import AttentionRGOperator
        torch.manual_seed(0)
        std_op  = StandardRGOperator(in_dim=16, out_dim=16)
        attn_op = AttentionRGOperator(d_model=16, n_heads=4)
        x = torch.randn(4, 16)
        with torch.no_grad():
            y_std  = std_op(x)
            y_attn = attn_op(x)
        assert not torch.allclose(y_std, y_attn, atol=1e-3), \
            "Attention and standard operators should produce different outputs"

    def test_attention_gradient_propagates(self):
        from src.rg_flow.operators.attention_rg_operator import AttentionRGOperator
        op = AttentionRGOperator(d_model=16, n_heads=4)
        x  = torch.randn(4, 16, requires_grad=True)
        op(x).sum().backward()
        assert x.grad is not None and x.grad.norm() > 0
 