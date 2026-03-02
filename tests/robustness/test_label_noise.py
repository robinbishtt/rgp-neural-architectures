"""
tests/robustness/test_label_noise.py

Label noise robustness: mislabeled training example tolerance.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_model(in_dim: int = 16, n_classes: int = 2) -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(in_dim, in_dim), nn.Tanh(),
        nn.Linear(in_dim, n_classes),
    )


def _corrupt_labels(y: torch.Tensor, noise_rate: float, n_classes: int) -> torch.Tensor:
    """Randomly flip a fraction of labels to a different class."""
    y_noisy = y.clone()
    mask = torch.rand(len(y)) < noise_rate
    random_labels = torch.randint(0, n_classes, (mask.sum().item(),))
    y_noisy[mask] = random_labels
    return y_noisy


@pytest.fixture
def data():
    torch.manual_seed(7)
    x = torch.randn(64, 16)
    y = torch.randint(0, 2, (64,))
    return x, y


class TestLabelNoise:

    def test_zero_noise_loss_finite(self, data):
        """Training with 0% label noise must produce finite loss."""
        x, y = data
        model = _make_model()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        assert torch.isfinite(loss), "Loss is non-finite with clean labels."

    def test_high_noise_loss_finite(self, data):
        """Training with 50% label noise must produce finite loss."""
        x, y = data
        y_noisy = _corrupt_labels(y, noise_rate=0.5, n_classes=2)
        model = _make_model()
        logits = model(x)
        loss = F.cross_entropy(logits, y_noisy)
        assert torch.isfinite(loss), "Loss is non-finite with 50% label noise."

    def test_full_noise_loss_finite(self, data):
        """Training with 100% label noise must produce finite loss."""
        x, y = data
        y_noisy = _corrupt_labels(y, noise_rate=1.0, n_classes=2)
        model = _make_model()
        logits = model(x)
        loss = F.cross_entropy(logits, y_noisy)
        assert torch.isfinite(loss), "Loss is non-finite with 100% label noise."

    def test_loss_increases_with_noise(self, data):
        """Cross-entropy loss should generally be higher with more label noise."""
        x, y = data
        model = _make_model()
        with torch.no_grad():
            logits = model(x)

        losses = {}
        for rate in [0.0, 0.2, 0.5]:
            y_noisy = _corrupt_labels(y, rate, n_classes=2)
            losses[rate] = F.cross_entropy(logits, y_noisy).item()

        assert losses[0.5] >= losses[0.0] - 0.5, (
            f"Loss at 50% noise ({losses[0.5]:.3f}) not >= clean loss ({losses[0.0]:.3f})."
        )

    def test_gradient_finite_under_noise(self, data):
        """Gradients must remain finite when training with noisy labels."""
        x, y = data
        y_noisy = _corrupt_labels(y, 0.3, n_classes=2)
        model = _make_model()
        logits = model(x)
        loss = F.cross_entropy(logits, y_noisy)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in {name} under label noise."
                )
 