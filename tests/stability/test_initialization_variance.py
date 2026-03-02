"""tests/stability/test_initialization_variance.py"""
import pytest
import torch
import torch.nn as nn


class TestInitializationVariance:
    def test_sigma_w_squared_equals_one_over_N(self):
        """Critical init: each row of W should have variance ~1/N."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(7)
        model = RGNetStandard(input_dim=64, n_classes=8)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                empirical_var = m.weight.data.var().item()
                expected_var  = 1.0 / fan_in
                ratio = empirical_var / expected_var
                assert 0.5 < ratio < 2.0, \
                    f"Weight variance ratio: {ratio:.3f} (expected ~1.0)"

    def test_bias_initialization_near_zero(self):
        """Bias terms should be initialized near zero for critical init."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(8)
        model = RGNetStandard(input_dim=32, n_classes=4)
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                max_bias = m.bias.data.abs().max().item()
                assert max_bias < 1.0, f"Bias too large at init: {max_bias:.4f}"
 