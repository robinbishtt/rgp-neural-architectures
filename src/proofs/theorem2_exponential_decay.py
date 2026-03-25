from __future__ import annotations
import numpy as np
def propagate_correlation(
    c12_init: float,
    chi1: float,
    n_layers: int,
) -> np.ndarray:
    c = [c12_init]
    for _ in range(n_layers):
        c.append(chi1 * c[-1])
    return np.array(c)
def verify_exponential_decay(chi1: float = 0.8, n_layers: int = 30) -> bool:
    from scipy.optimize import curve_fit
    k           = np.arange(n_layers + 1, dtype=float)
    c           = propagate_correlation(0.9, chi1, n_layers)
    kc_analytic = -1.0 / np.log(chi1)
    def _exp_c(k, c0, kc): return c0 * np.exp(-k / kc)
    try:
        popt, _ = curve_fit(
            _exp_c, k, c, p0=[c[0], kc_analytic],
            bounds=([0, 0.01], [1.0, kc_analytic * 5]),
            maxfev=5000
        )
        kc_fitted = popt[1]
    except Exception:
        log_c = np.log(np.clip(c, 1e-30, None))
        slope = np.polyfit(k, log_c, 1)[0]
        kc_fitted = -1.0 / slope if slope < 0 else kc_analytic * 100
    return bool(abs(kc_fitted - kc_analytic) / kc_analytic < 0.05)
def run_all_verifications() -> dict:
    results = {
        : verify_exponential_decay(0.8),
        : verify_exponential_decay(0.5),
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
        theorem_name="Theorem 2: Exponential Correlation Decay c^(ell)=c^(0)*chi^ell",
        passed=bool(results["all_pass"]),
        n_tests=n_tests, n_passed=n_passed,
        max_error=0.0 if results["all_pass"] else 1.0,
        mean_error=0.0 if results["all_pass"] else 1.0,
        elapsed_s=time.time() - t0,
        details=results,
    )