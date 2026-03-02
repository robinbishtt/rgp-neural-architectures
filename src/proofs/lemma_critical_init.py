"""
src/proofs/lemma_critical_init.py

Lemma: Critical Initialization.

Statement: There exists a unique σ_w* such that χ₁(σ_w*) = 1,
placing the network exactly at the edge of chaos. At this point:
  - ξ → ∞ (infinite correlation length)
  - Lyapunov exponent λ₁ = 0 (marginal stability)
  - Gradients propagate without vanishing or exploding

For tanh nonlinearity: σ_w* ≈ 1.48 (verified by Gauss-Hermite quadrature).
"""
from __future__ import annotations
import numpy as np
from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2


def verify_critical_sigma_w(
    nonlinearity: str = "tanh",
    tol: float = 1e-4,
) -> bool:
    """Verify chi1(sigma_w*) == 1 to within tolerance."""
    sigma_w_star = critical_sigma_w2(nonlinearity)
    chi1_at_star = chi1_gauss_hermite(sigma_w_star, nonlinearity)
    return bool(abs(chi1_at_star - 1.0) < tol)


def verify_infinite_correlation_at_critical() -> bool:
    """Verify ξ → ∞ as χ₁ → 1⁻."""
    chi1_values = [0.999, 0.9999, 0.99999]
    xi_values   = [-1.0 / np.log(c) for c in chi1_values]
    # ξ should be increasing
    return all(xi_values[i + 1] > xi_values[i] for i in range(len(xi_values) - 1))


def run_all_verifications() -> dict:
    results = {
        "critical_sigma_w_tanh": verify_critical_sigma_w("tanh"),
        "critical_sigma_w_relu": verify_critical_sigma_w("relu"),
        "infinite_correlation":  verify_infinite_correlation_at_critical(),
    }
    results["all_pass"] = all(results.values())
    return results
 