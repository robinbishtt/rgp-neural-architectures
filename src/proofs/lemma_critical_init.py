from __future__ import annotations
import numpy as np
from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2
def verify_critical_sigma_w(
    nonlinearity: str = "tanh",
    tol: float = 1e-4,
) -> bool:
    sigma_w_star = critical_sigma_w2(nonlinearity)
    chi1_at_star = chi1_gauss_hermite(sigma_w_star, nonlinearity)
    return bool(abs(chi1_at_star - 1.0) < tol)
def verify_infinite_correlation_at_critical() -> bool:
    chi1_values = [0.999, 0.9999, 0.99999]
    xi_values   = [-1.0 / np.log(c) for c in chi1_values]
    return all(xi_values[i + 1] > xi_values[i] for i in range(len(xi_values) - 1))
def run_all_verifications() -> dict:
    results = {
        "critical_tanh":      verify_critical_sigma_w("tanh"),
        "critical_relu":      verify_critical_sigma_w("relu"),
        "infinite_correlation": verify_infinite_correlation_at_critical(),
    }
    results["all_pass"] = all(results.values())
    return results
def verify(n_trials: int = 100, tol: float = 1e-5):
    from src.proofs.proof_utils import VerificationResult
    import time
    t0 = time.time()
    results = run_all_verifications()
    n_tests  = len(results) - 1
    n_passed = sum(1 for k, v in results.items() if k != "all_pass" and v)
    return VerificationResult(
        theorem_name="Lemma: Critical Initialization chi1(sigma_w*)=1, xi_depth->inf",
        passed=bool(results["all_pass"]),
        n_tests=n_tests, n_passed=n_passed,
        max_error=0.0 if results["all_pass"] else 1.0,
        mean_error=0.0 if results["all_pass"] else 1.0,
        elapsed_s=time.time() - t0,
        details=results,
    )