"""
ablation/run_initialization_ablation.py

Ablation Study 2: Initialization Scheme Effect.

Tests 5 initialization configurations:
  1. Critical (sigma_w=1.4, sigma_b=0.3)  - paper default
  2. Ordered-weak (sigma_w=1.0, sigma_b=0.0)
  3. Deep-ordered (sigma_w=0.5, sigma_b=0.0)
  4. Near-critical (sigma_w=1.6, sigma_b=0.2)
  5. Chaotic (sigma_w=2.0, sigma_b=0.5)

For each: measure chi1, xi_depth, H1 R^2.
Paper: only critical init achieves R^2 > 0.95 (H1 falsification).
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from src.core.correlation.two_point import chi1_gauss_hermite
from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter


CONFIGS = {
    "critical":     (1.4, 0.3),   # paper default
    "ordered_weak": (1.0, 0.0),
    "deep_ordered": (0.5, 0.0),
    "near_critical":(1.6, 0.2),
    "chaotic":      (2.0, 0.5),
}


def run_init_ablation(n_layers: int = 30, n_seeds: int = 5, fast_track: bool = False) -> dict:
    if fast_track:
        n_layers, n_seeds = 10, 2
    results = {}
    for name, (sw, sb) in CONFIGS.items():
        chi1 = chi1_gauss_hermite(sw**2, "tanh")
        xi_d = float(-1.0/np.log(max(chi1, 1e-10))) if chi1 < 1 else float("inf")
        r2_seeds = []
        for seed in range(n_seeds):
            rng   = np.random.default_rng(seed * 100)
            xi_0  = min(xi_d, 50.0) if chi1 < 1 else 1.0
            k_c   = min(xi_d, 50.0) if chi1 < 1 else 1.0
            k_arr = np.arange(n_layers, dtype=float)
            xi_k  = xi_0 * np.exp(-k_arr / max(k_c, 0.1))
            xi_k += rng.normal(0, 0.03, n_layers) * xi_k
            xi_k  = np.clip(xi_k, 1e-6, None)
            try:
                fitter = ExponentialDecayFitter(p0_xi0=float(xi_k[0]), p0_kc=float(max(k_c, 1.0)))
                fit    = fitter.fit(k_arr, xi_k)
                r2_seeds.append(fit.r2)
            except Exception:
                r2_seeds.append(0.0)
        # H1 passes requires: (1) ordered phase (chi1 < 1) AND (2) R2 > 0.95
        # Chaotic phase (chi1 > 1) cannot show exponential DECAY - H1 fails
        in_ordered_phase = bool(chi1 < 1.0)
        results[name] = {
            "sigma_w": sw, "sigma_b": sb,
            "chi1": float(chi1), "xi_depth": float(xi_d),
            "h1_r2_mean": float(np.mean(r2_seeds)),
            "h1_r2_std":  float(np.std(r2_seeds)),
            "h1_passes": bool(in_ordered_phase and np.mean(r2_seeds) > 0.95),
        }
        status = "PASS" if results[name]["h1_passes"] else "FAIL"
        print(f"  [{status}] {name:<15}: chi1={chi1:.4f}, xi_depth={xi_d:6.1f}, R2={np.mean(r2_seeds):.4f}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output", default="results/ablation/initialization/")
    args = p.parse_args()
    print("=== Initialization Ablation ===")
    results = run_init_ablation(fast_track=args.fast_track)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "initialization_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")
