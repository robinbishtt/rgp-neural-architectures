"""
ablation/run_width_ablation.py

Ablation Study 3: Network Width Effect (1/N Corrections).

Finite-width networks deviate from mean-field prediction.
Tests N in {32, 64, 128, 256, 512, 1024}.

Paper: for N >= 256, corrections < 1%. For N < 50, corrections dominate.
Appendix F.1: epsilon_0(N) = epsilon_0(inf) + c/N + O(N^{-2}).
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from src.core.correlation.two_point import chi1_gauss_hermite


WIDTHS = [32, 64, 128, 256, 512, 1024]
SIGMA_W = 1.4  # Paper critical init
SIGMA_B = 0.3


def finite_width_chi1(N: int, sigma_w: float = SIGMA_W, n_samples: int = 500, seed: int = 42) -> float:
    """Empirical chi1 from finite-width random network Jacobians."""
    rng = np.random.default_rng(seed)
    chi1_samples = []
    for _ in range(n_samples):
        W = rng.standard_normal((N, N)) * (sigma_w / N**0.5)
        x = rng.standard_normal(N)
        h = np.tanh(W @ x + SIGMA_B * rng.standard_normal(N))
        dphi = 1.0 - h**2
        J = dphi[:, None] * W
        sv = np.linalg.svdvals(J)
        chi1_samples.append(float(sv.mean()**2))  # approximation
    return float(np.mean(chi1_samples))


def run_width_ablation(fast_track: bool = False) -> dict:
    n_samples = 50 if fast_track else 500
    chi1_mf  = chi1_gauss_hermite(SIGMA_W**2, "tanh")
    results  = {}
    print(f"  Mean-field chi1 (N->inf): {chi1_mf:.6f}")
    for N in (WIDTHS[:3] if fast_track else WIDTHS):
        chi1_emp = finite_width_chi1(N, n_samples=n_samples)
        correction = abs(chi1_emp - chi1_mf)
        results[f"N_{N}"] = {
            "N": N,
            "chi1_empirical": chi1_emp,
            "chi1_mf": chi1_mf,
            "correction": correction,
            "correction_pct": correction / max(chi1_mf, 1e-6) * 100,
            "mean_field_valid": bool(correction < 0.01),
        }
        print(f"  N={N:5d}: chi1_emp={chi1_emp:.4f}, correction={correction:.4f} ({correction/chi1_mf*100:.2f}%)")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output", default="results/ablation/width/")
    args = p.parse_args()
    print("=== Width Ablation (1/N Corrections) ===")
    results = run_width_ablation(fast_track=args.fast_track)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "width_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")
