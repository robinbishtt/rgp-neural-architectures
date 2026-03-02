"""
src/proofs/proof_utils.py

Shared utilities for symbolic and numerical proof verification routines.

Provides:
    - symbolic_approx_equal: tests symbolic equality under simplification
    - numerical_verify: checks theorem conditions on random instances
    - latex_export: exports proof steps to LaTeX
    - ProofLogger: structured logging of verification results
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import numpy as np


@dataclass
class VerificationResult:
    theorem_name:   str
    passed:         bool
    n_tests:        int
    n_passed:       int
    max_error:      float
    mean_error:     float
    elapsed_s:      float
    details:        Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_tests, 1)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.theorem_name}: "
            f"{self.n_passed}/{self.n_tests} tests, "
            f"max_err={self.max_error:.2e}, "
            f"elapsed={self.elapsed_s:.2f}s"
        )


class NumericalVerifier:
    """
    Runs randomized numerical verification of mathematical theorems.

    Strategy: generate many random instances satisfying the theorem's
    preconditions, evaluate both sides of the claimed equality/inequality,
    and report the maximum absolute error.
    """

    def __init__(
        self,
        n_trials:   int   = 100,
        tol:        float = 1e-5,
        seed:       int   = 42,
    ) -> None:
        self.n_trials = n_trials
        self.tol      = tol
        self.rng      = np.random.default_rng(seed)

    def verify(
        self,
        theorem_name:   str,
        lhs_fn:         Callable[..., np.ndarray],
        rhs_fn:         Callable[..., np.ndarray],
        sampler:        Callable[[], tuple],
    ) -> VerificationResult:
        """
        Verify theorem: lhs_fn(*sample) ≈ rhs_fn(*sample) for n_trials samples.

        Args:
            theorem_name: human-readable name for logging
            lhs_fn:       function computing left-hand side
            rhs_fn:       function computing right-hand side
            sampler:      zero-argument function returning tuple of inputs

        Returns:
            VerificationResult with statistics
        """
        errors   = []
        n_passed = 0
        t0       = time.time()

        for _ in range(self.n_trials):
            try:
                args  = sampler()
                lhs   = np.asarray(lhs_fn(*args))
                rhs   = np.asarray(rhs_fn(*args))
                err   = float(np.max(np.abs(lhs - rhs)))
                errors.append(err)
                if err <= self.tol:
                    n_passed += 1
            except Exception as e:
                errors.append(float("inf"))

        elapsed = time.time() - t0
        passed  = n_passed == self.n_trials

        return VerificationResult(
            theorem_name=theorem_name,
            passed=passed,
            n_tests=self.n_trials,
            n_passed=n_passed,
            max_error=float(np.max(errors)) if errors else float("inf"),
            mean_error=float(np.mean(errors)) if errors else float("inf"),
            elapsed_s=elapsed,
        )


class ProofLogger:
    """
    Structured logger for proof verification results.
    Collects results across all theorems and generates summary reports.
    """

    def __init__(self) -> None:
        self._results: List[VerificationResult] = []

    def record(self, result: VerificationResult) -> None:
        """Record a single verification result."""
        self._results.append(result)

    def summary(self) -> str:
        """Generate a human-readable summary of all recorded results."""
        lines = ["=" * 60, "PROOF VERIFICATION SUMMARY", "=" * 60]
        n_pass = sum(1 for r in self._results if r.passed)
        for r in self._results:
            lines.append(str(r))
        lines.append("-" * 60)
        lines.append(f"Total: {n_pass}/{len(self._results)} theorems verified")
        lines.append("=" * 60)
        return "\n".join(lines)

    def all_passed(self) -> bool:
        """Return True if all recorded proofs passed."""
        return all(r.passed for r in self._results)
