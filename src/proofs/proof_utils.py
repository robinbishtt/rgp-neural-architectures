from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
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
            except Exception:
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
    def __init__(self) -> None:
        self._results: List[VerificationResult] = []
    def record(self, result: VerificationResult) -> None:
        self._results.append(result)
    def summary(self) -> str:
        lines = ["=" * 60, "PROOF VERIFICATION SUMMARY", "=" * 60]
        n_pass = sum(1 for r in self._results if r.passed)
        for r in self._results:
            lines.append(str(r))
        lines.append("-" * 60)
        lines.append(f"Total: {n_pass}/{len(self._results)} theorems verified")
        lines.append("=" * 60)
        return "\n".join(lines)
    def all_passed(self) -> bool:
        return all(r.passed for r in self._results)