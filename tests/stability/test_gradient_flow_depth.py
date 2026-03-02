"""tests/stability/test_gradient_flow_depth.py"""
import torch
import torch.nn as nn


class TestGradientFlowDepth:
    def test_gradient_reaches_first_layer(self):
        """Gradient must reach the first layer (no complete vanishing)."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(9)
        model = RGNetStandard(input_dim=32, n_classes=4)
        x = torch.randn(8, 32)
        y = torch.randint(0, 4, (8,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        # Find gradient of first linear layer
        first_grad_norm = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    first_grad_norm = m.weight.grad.norm().item()
                break
        assert first_grad_norm is not None and first_grad_norm > 1e-10, \
            f"Gradient vanished at first layer: {first_grad_norm}"

    def test_gradient_consistent_across_depth(self):
        """Gradient norms across depth should not differ by more than 3 orders."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(10)
        model = RGNetStandard(input_dim=32, n_classes=4)
        x = torch.randn(16, 32)
        y = torch.randint(0, 4, (16,))
        nn.CrossEntropyLoss()(model(x), y).backward()
        grad_norms = []
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.grad is not None:
                grad_norms.append(m.weight.grad.norm().item())
        if len(grad_norms) >= 2:
            ratio = max(grad_norms) / (min(grad_norms) + 1e-12)
            assert ratio < 1e3, f"Gradient norm ratio too large: {ratio:.2e}"
 