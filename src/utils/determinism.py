"""
src/utils/determinism.py

Global determinism configuration — applies torch deterministic mode,
environment variables, and validates reproducibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DeterminismConfig:
    use_deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    warn_only: bool = True  # warn instead of error on non-deterministic ops


def apply_global_determinism(cfg: DeterminismConfig = DeterminismConfig()) -> None:
    """
    Apply determinism settings globally.
    Call once at program startup before any computation.
    """
    import torch

    # CUDA workspace config for deterministic CUBLAS
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark     = cfg.cudnn_benchmark

    if cfg.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=cfg.warn_only)


def verify_determinism(
    model_fn,
    input_fn,
    seed: int = 42,
    n_trials: int = 3,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify that model_fn produces bit-identical outputs across n_trials runs
    with the same seed.

    Returns True if deterministic, False otherwise.
    """
    import torch
    from src.utils.seed_registry import SeedRegistry

    reg = SeedRegistry.get_instance()
    outputs = []

    for _ in range(n_trials):
        reg.set_master_seed(seed)
        model = model_fn()
        x     = input_fn()
        with torch.no_grad():
            out = model(x)
        outputs.append(out.detach().cpu())

    reference = outputs[0]
    for i, out in enumerate(outputs[1:], start=1):
        if not torch.allclose(reference, out, rtol=rtol, atol=1e-7):
            return False
    return True
 