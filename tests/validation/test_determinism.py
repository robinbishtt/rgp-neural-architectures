"""
tests/validation/test_determinism.py

Validates seed reproducibility: two runs with identical seeds must produce
bit-exact identical outputs across model forward passes, gradient computations,
and data loading.
"""

import pytest
import random
import torch
import torch.nn as nn
import numpy as np


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _make_model(seed: int, width: int = 32, depth: int = 4) -> nn.Module:
    _set_all_seeds(seed)
    layers = [nn.Linear(width, width), nn.Tanh()]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def _make_input(seed: int, batch: int = 16, width: int = 32) -> torch.Tensor:
    _set_all_seeds(seed)
    return torch.randn(batch, width)


class TestDeterminism:

    def test_forward_pass_bit_exact(self):
        """
        Two models initialised with the same seed and given the same input
        must produce bit-exact identical outputs.
        """
        for master_seed in [0, 42, 123]:
            model_a = _make_model(master_seed)
            model_b = _make_model(master_seed)
            x = _make_input(master_seed)

            with torch.no_grad():
                out_a = model_a(x)
                out_b = model_b(x)

            assert torch.equal(out_a, out_b), (
                f"Forward pass not bit-exact at seed={master_seed}."
            )

    def test_backward_pass_bit_exact(self):
        """
        Gradients computed in two independent runs with the same seed
        must be bit-exact identical.
        """
        for seed in [7, 42]:
            grad_a, grad_b = {}, {}

            for run_idx, store in enumerate([grad_a, grad_b]):
                _set_all_seeds(seed)
                model = _make_model(seed)
                x     = _make_input(seed)
                y     = torch.randint(0, 2, (x.shape[0],))
                logits = model(x)
                loss   = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        store[name] = param.grad.clone()

            for name in grad_a:
                assert torch.equal(grad_a[name], grad_b[name]), (
                    f"Gradient for '{name}' not bit-exact at seed={seed}."
                )

    def test_numpy_rng_reproducible(self):
        """NumPy RNG must produce identical sequences for the same seed."""
        for seed in [0, 99]:
            np.random.seed(seed)
            a = np.random.randn(50)
            np.random.seed(seed)
            b = np.random.randn(50)
            np.testing.assert_array_equal(a, b, err_msg=f"NumPy not reproducible at seed={seed}.")

    def test_python_rng_reproducible(self):
        """Python random module must produce identical sequences for the same seed."""
        for seed in [1, 55]:
            random.seed(seed)
            a = [random.random() for _ in range(20)]
            random.seed(seed)
            b = [random.random() for _ in range(20)]
            assert a == b, f"Python random not reproducible at seed={seed}."

    def test_different_seeds_different_outputs(self):
        """Different seeds must produce different model outputs."""
        x = _make_input(0)
        model_a = _make_model(seed=0)
        model_b = _make_model(seed=99)
        with torch.no_grad():
            out_a = model_a(x)
            out_b = model_b(x)
        assert not torch.equal(out_a, out_b), (
            "Models with different seeds produced identical outputs (improbable)."
        )
 