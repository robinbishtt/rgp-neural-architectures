"""
tests/robustness/test_adversarial_robustness.py

Adversarial robustness: FGSM and PGD attack evaluation.
Validates that RG-Net degrades gracefully under adversarial perturbations.
"""

import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_model(width: int = 32, depth: int = 4) -> nn.Module:
    layers = [nn.Linear(width, width), nn.Tanh()]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015).
    x_adv = x + epsilon * sign(grad_x L(f(x), y))
    """
    x_adv = x.clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    n_steps: int = 10,
) -> torch.Tensor:
    """
    Projected Gradient Descent attack (Madry et al., 2018).
    Iterative FGSM with projection onto L-inf ball of radius epsilon.
    """
    x_adv = x.clone() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = x_adv.detach()

    for _ in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back into epsilon-ball
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = (x + delta).detach()

    return x_adv


@pytest.fixture
def setup():
    torch.manual_seed(42)
    model = _make_model()
    x = torch.randn(8, 32)
    y = torch.randint(0, 2, (8,))
    return model, x, y


class TestAdversarialRobustness:

    def test_fgsm_output_finite(self, setup):
        """FGSM adversarial examples must produce finite model outputs."""
        model, x, y = setup
        x_adv = fgsm_attack(model, x, y, epsilon=0.1)
        with torch.no_grad():
            out = model(x_adv)
        assert torch.isfinite(out).all(), "NaN/Inf in FGSM adversarial output."

    def test_fgsm_perturbation_bounded(self, setup):
        """FGSM perturbation L-inf norm must equal epsilon exactly."""
        model, x, y = setup
        epsilon = 0.1
        x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
        delta = (x_adv - x).abs().max().item()
        assert abs(delta - epsilon) < 1e-4, (
            f"FGSM perturbation delta={delta:.4f} != epsilon={epsilon}."
        )

    def test_pgd_output_finite(self, setup):
        """PGD adversarial examples must produce finite model outputs."""
        model, x, y = setup
        x_adv = pgd_attack(model, x, y, epsilon=0.1, n_steps=5)
        with torch.no_grad():
            out = model(x_adv)
        assert torch.isfinite(out).all(), "NaN/Inf in PGD adversarial output."

    def test_pgd_perturbation_within_epsilon_ball(self, setup):
        """PGD adversarial delta must stay within L-inf epsilon-ball."""
        model, x, y = setup
        epsilon = 0.1
        x_adv = pgd_attack(model, x, y, epsilon=epsilon, n_steps=10)
        delta = (x_adv - x).abs().max().item()
        assert delta <= epsilon + 1e-5, (
            f"PGD delta={delta:.4f} exceeds epsilon={epsilon}."
        )

    def test_adversarial_increases_loss(self, setup):
        """FGSM must increase cross-entropy loss compared to clean input."""
        model, x, y = setup
        x_adv = fgsm_attack(model, x, y, epsilon=0.1)
        with torch.no_grad():
            loss_clean = F.cross_entropy(model(x), y).item()
            loss_adv   = F.cross_entropy(model(x_adv), y).item()
        assert loss_adv >= loss_clean - 1e-4, (
            f"Adversarial loss {loss_adv:.4f} not >= clean loss {loss_clean:.4f}."
        )
 