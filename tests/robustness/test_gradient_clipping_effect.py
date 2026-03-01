"""tests/robustness/test_gradient_clipping_effect.py"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim


class TestGradientClippingEffect:
    def test_clipped_gradients_bounded(self):
        """After gradient clipping, max gradient norm should not exceed clip value."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        from src.training.training_utils import clip_gradients
        torch.manual_seed(4)
        model = RGNetStandard(input_dim=32, n_classes=4)
        x = torch.randn(16, 32)
        y = torch.randint(0, 4, (16,))
        nn.CrossEntropyLoss()(model(x), y).backward()
        clip_val = 1.0
        clip_gradients(model, max_norm=clip_val)
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        total_norm = total ** 0.5
        assert total_norm <= clip_val + 1e-5, \
            f"Gradient norm {total_norm:.4f} exceeds clip value {clip_val}"

    def test_training_stable_with_clipping(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        from src.training.training_utils import clip_gradients
        torch.manual_seed(5)
        model = RGNetStandard(input_dim=32, n_classes=4)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for step in range(20):
            x, y = torch.randn(16, 32), torch.randint(0, 4, (16,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            opt.zero_grad()
            loss.backward()
            clip_gradients(model, max_norm=1.0)
            opt.step()
            assert not torch.isnan(loss)
