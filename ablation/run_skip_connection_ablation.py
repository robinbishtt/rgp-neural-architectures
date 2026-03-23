"""
ablation/run_skip_connection_ablation.py

Ablation Study 6: Skip Connection Effect on Deep Networks.

Compares: no skip, skip every 5, skip every 10, skip every 20.
For depth=100, xi_data=50.0.

Hypothesis: skip connections help stabilize gradient flow but do not
change the fundamental logarithmic depth scaling.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path


SKIP_INTERVALS = [None, 5, 10, 20, 50]  # None = no skip
DEPTH = 100
SIGMA_W = 1.4


def simulate_gradient_norm(depth: int, skip_interval: None | int, sigma_w: float,
                            seed: int = 42) -> float:
    """Simulate gradient norm at layer 0 for a random network."""
    rng = np.random.default_rng(seed)
    grad = np.ones(32) / 32**0.5

    for ell in range(depth, 0, -1):
        W = rng.standard_normal((32, 32)) * (sigma_w / 32**0.5)
        h = np.tanh(W @ rng.standard_normal(32))
        dphi = 1.0 - h**2
        J = dphi[:, None] * W
        grad = J.T @ grad
        if skip_interval and ell % skip_interval == 0:
            grad = grad * 0.5 + np.ones(32) * 0.5 / 32**0.5

    return float(np.linalg.norm(grad))


def run_skip_ablation(fast_track: bool = False) -> dict:
    n_seeds = 2 if fast_track else 10
    depth   = 20 if fast_track else DEPTH
    results = {}
    for skip in SKIP_INTERVALS:
        norms = [simulate_gradient_norm(depth, skip, SIGMA_W, seed=s) for s in range(n_seeds)]
        name = f"skip_{skip}" if skip else "no_skip"
        results[name] = {
            "skip_interval": skip,
            "grad_norm_mean": float(np.mean(norms)),
            "grad_norm_std":  float(np.std(norms)),
            "vanishing": bool(np.mean(norms) < 1e-4),
            "exploding": bool(np.mean(norms) > 100),
        }
        status = "VANISHING" if results[name]["vanishing"] else ("EXPLODING" if results[name]["exploding"] else "STABLE")
        print(f"  {name:<15}: grad_norm={np.mean(norms):.6f}±{np.std(norms):.4f}  [{status}]")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output", default="results/ablation/skip/")
    args = p.parse_args()
    print("=== Skip Connection Ablation ===")
    results = run_skip_ablation(fast_track=args.fast_track)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "skip_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")
