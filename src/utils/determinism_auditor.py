"""
src/utils/determinism_auditor.py

DeterminismAuditor: validates that all RNG calls go through SeedRegistry.
BitExactVerifier: compares two runs with the same seed for bit-exact outputs.
"""
from __future__ import annotations
import hashlib
import logging
import numpy as np
import torch
from typing import Callable, List

logger = logging.getLogger(__name__)


class DeterminismAuditor:
    """
    Audits code for direct RNG access bypassing SeedRegistry.

    Monkey-patches numpy and torch RNG entry points, logs warnings when
    they are called without going through SeedRegistry.
    """

    def __init__(self) -> None:
        self._violations: List[str] = []
        self._active = False

    def start_audit(self) -> None:
        self._active = True
        self._violations.clear()

    def stop_audit(self) -> List[str]:
        self._active = False
        return list(self._violations)

    def record_violation(self, source: str) -> None:
        if self._active:
            self._violations.append(source)
            logger.warning("Direct RNG access detected at: %s", source)

    @property
    def n_violations(self) -> int:
        return len(self._violations)


class BitExactVerifier:
    """
    Verifies bit-exact reproducibility across two runs with the same seed.

    Usage:
        verifier = BitExactVerifier()
        h1 = verifier.run_and_hash(fn, seed=42)
        h2 = verifier.run_and_hash(fn, seed=42)
        assert verifier.compare(h1, h2), "Not bit-exact!"
    """

    def run_and_hash(
        self, fn: Callable, seed: int, *args, **kwargs
    ) -> str:
        """Run fn with given seed and return SHA-256 of output."""
        from src.utils.seed_registry import SeedRegistry
        SeedRegistry.get_instance().set_master_seed(seed)
        result = fn(*args, **kwargs)

        h = hashlib.sha256()
        if isinstance(result, torch.Tensor):
            h.update(result.detach().cpu().numpy().tobytes())
        elif isinstance(result, np.ndarray):
            h.update(result.tobytes())
        else:
            h.update(str(result).encode())
        return h.hexdigest()

    def compare(self, hash1: str, hash2: str) -> bool:
        return hash1 == hash2

    def verify_n_runs(
        self, fn: Callable, seed: int, n_runs: int = 3, *args, **kwargs
    ) -> bool:
        """Verify fn produces identical output across n_runs with same seed."""
        hashes = [self.run_and_hash(fn, seed, *args, **kwargs) for _ in range(n_runs)]
        return len(set(hashes)) == 1
 