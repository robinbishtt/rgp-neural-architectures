"""tests/stability/test_weight_norms.py"""
import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
import torch.optim as optim


class TestWeightNorms:
    def test_weight_norms_stable_after_steps(self):
        """Weight norms should not diverge during the first training steps."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(3)
        model = RGNetStandard(input_dim=32, n_classes=4)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for step in range(20):
            x = torch.randn(16, 32)
            y = torch.randint(0, 4, (16,))
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for name, p in model.named_parameters():
            norm = p.data.norm().item()
            assert norm < 1e4, f"Weight norm diverged for {name}: {norm:.2e}"
            assert norm > 1e-10, f"Weight norm collapsed to zero for {name}"

    def test_initial_weight_norms_reasonable(self):
        """Initial weight norms should scale as 1/sqrt(fan_in) (critical init)."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(4)
        model = RGNetStandard(input_dim=64, n_classes=8)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                norm_per_unit = m.weight.data.norm() / (fan_in ** 0.5)
                assert 0.1 < norm_per_unit < 10.0, \
                    f"Unexpected weight norm per sqrt(fan_in): {norm_per_unit:.4f}"
 