"""tests/stability/test_loss_monotonicity.py"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim


class TestLossMonotonicity:
    def test_loss_decreases_on_fixed_batch(self):
        """With a fixed mini-batch, SGD should decrease loss over enough steps."""
        from src.architectures.rg_net.rg_net_shallow import RGNetShallow
        torch.manual_seed(5)
        model = RGNetShallow(input_dim=16, n_classes=2)
        opt = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(32, 16)
        y = torch.randint(0, 2, (32,))
        losses = []
        for _ in range(50):
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Loss should decrease at least somewhat over 50 steps
        assert losses[-1] < losses[0] * 1.5, \
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"

    def test_no_nan_loss_throughout_training(self):
        """Loss must not produce NaN values during standard training."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(6)
        model = RGNetStandard(input_dim=32, n_classes=4)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for _ in range(30):
            x = torch.randn(16, 32)
            y = torch.randint(0, 4, (16,))
            loss = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert not torch.isnan(loss), f"NaN loss detected: {loss.item()}"
