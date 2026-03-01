"""
src/proofs/verification_runner.py

VerificationRunner: orchestrates execution of all formal proof verifications
(Theorem 1, Theorem 2, Theorem 3, and critical initialization lemma) and
produces a consolidated verification report.

Intended usage:
    runner = VerificationRunner()
    report = runner.run_all()
    print(report)
"""
from __future__ import annotations
from typing import Optional
import time

from src.proofs.proof_utils import ProofLogger, VerificationResult
from src.proofs.theorem1_fisher_transform import verify as verify_t1
from src.proofs.theorem2_exponential_decay import verify as verify_t2
from src.proofs.theorem3_depth_scaling import verify as verify_t3
from src.proofs.lemma_critical_init import verify as verify_lemma


class VerificationRunner:
    """
    Orchestrates all proof verifications and produces a consolidated report.

    This class is the single entry point for running the complete formal
    verification suite, as called by ``make validate`` via the Makefile.
    Each proof is run with configurable trial counts and tolerance thresholds
    appropriate for its mathematical structure.
    """

    def __init__(
        self,
        n_trials: int   = 100,
        tol:      float = 1e-5,
        verbose:  bool  = True,
    ) -> None:
        self.n_trials = n_trials
        self.tol      = tol
        self.verbose  = verbose
        self._logger  = ProofLogger()

    def run_all(self) -> str:
        """
        Run all proof verifications and return a consolidated report string.

        Executes:
            1. Theorem 1: Fisher metric transformation law
            2. Theorem 2: Exponential correlation length decay
            3. Theorem 3: Logarithmic depth scaling law
            4. Lemma:     Critical initialization conditions

        Returns:
            Multi-line string report of all verification outcomes.
        """
        t_start = time.time()

        for verify_fn, name in [
            (verify_t1, "Theorem 1 (Fisher Transform)"),
            (verify_t2, "Theorem 2 (Exponential Decay)"),
            (verify_t3, "Theorem 3 (Depth Scaling)"),
            (verify_lemma, "Lemma (Critical Initialization)"),
        ]:
            try:
                result = verify_fn(n_trials=self.n_trials, tol=self.tol)
                if self.verbose:
                    print(str(result))
                self._logger.record(result)
            except Exception as e:
                dummy = VerificationResult(
                    theorem_name=name, passed=False,
                    n_tests=0, n_passed=0,
                    max_error=float("inf"), mean_error=float("inf"),
                    elapsed_s=0.0, details={"error": str(e)},
                )
                self._logger.record(dummy)

        elapsed = time.time() - t_start
        report  = self._logger.summary()
        report += f"\nTotal wall time: {elapsed:.2f}s\n"
        return report

    @property
    def all_passed(self) -> bool:
        return self._logger.all_passed()
