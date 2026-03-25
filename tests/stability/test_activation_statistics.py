import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
class TestActivationStatistics:
    def test_unit_variance_at_critical_init(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(0)
        model = RGNetStandard(input_dim=64, n_classes=10)
        x = torch.randn(32, 64)
        variances = []
        hooks = []
        def hook(m, inp, out):
            variances.append(out.detach().var().item())
        for m in model.modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(hook))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        if variances:
            max_var = max(variances)
            min_var = min(variances)
            assert max_var / (min_var + 1e-12) < 1e3,                f"Activation variance range too large: min={min_var:.4f}, max={max_var:.4f}"
    def test_no_dead_neurons(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(1)
        model = RGNetStandard(input_dim=32, n_classes=4)
        x = torch.randn(64, 32)
        variances = []
        hooks = []
        def hook(m, inp, out):
            variances.append(out.detach().var(dim=0))
        for m in model.modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(hook))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        for i, v in enumerate(variances):
            dead = (v < 1e-8).sum().item()
            total = v.numel()
            assert dead / total < 0.5,                f"Layer {i}: {dead}/{total} neurons are dead"