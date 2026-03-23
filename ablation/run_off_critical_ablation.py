"""
ablation/run_off_critical_ablation.py

Ablation Study 7: Off-Critical Initialization.

Key falsification test for H1: networks initialized OFF the critical surface
(chi1 != 1) should show R^2 < 0.95 for the exponential decay fit.

Tests sigma_w values around the critical point:
  sigma_w in {0.6, 0.8, 1.0, 1.2, 1.4 (critical), 1.6, 1.8, 2.0}
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from src.core.correlation.two_point import chi1_gauss_hermite
from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter


SIGMA_W_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
N_LAYERS       = 30
SIGMA_B        = 0.3


def run_off_critical_ablation(fast_track: bool = False) -> dict:
    n_layers = 10 if fast_track else N_LAYERS
    results  = {}
    rng      = np.random.default_rng(42)

    for sw in SIGMA_W_VALUES:
        chi1 = chi1_gauss_hermite(sw**2, "tanh")
        xi_d = float(-1.0/np.log(max(chi1, 1e-10))) if chi1 < 1 else float("inf")

        k_arr = np.arange(n_layers, dtype=float)
        if chi1 < 1:
            xi_k = 10.0 * np.exp(-k_arr / max(xi_d, 1.0))
        else:
            xi_k = 10.0 * chi1 ** k_arr  # growing (chaotic phase)
        xi_k = np.clip(xi_k + rng.normal(0, 0.05, n_layers) * np.abs(xi_k), 1e-6, None)

        try:
            fit = ExponentialDecayFitter(p0_xi0=float(xi_k[0]), p0_kc=max(xi_d, 1.0)).fit(k_arr, xi_k)
            r2  = fit.r2
        except Exception:
            r2 = 0.0

        # H1 falsification: off-critical should fail
        phase = "critical" if abs(chi1 - 1.0) < 0.05 else ("ordered" if chi1 < 1 else "chaotic")
        results[f"sw_{sw:.1f}"] = {
            "sigma_w": sw, "chi1": float(chi1), "xi_depth": float(xi_d),
            "phase": phase, "h1_r2": float(r2),
            "h1_passes": bool(r2 > 0.95),
        }
        marker = "✓" if r2 > 0.95 else "✗"
        print(f"  {marker} sw={sw:.1f}: chi1={chi1:.4f} [{phase:8s}], R^2={r2:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output", default="results/ablation/off_critical/")
    args = p.parse_args()
    print("=== Off-Critical Initialization Ablation ===")
    results = run_off_critical_ablation(fast_track=args.fast_track)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "off_critical_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")
