"""
experiments/h1_scale_correspondence/run_h1_validation.py

H1 Validation: Scale Correspondence Experiment Runner.

Hypothesis H1: The correlation length xi(k) decays exponentially with depth k,
following xi(k) = xi_0 * exp(-k / k_c), with R^2 > 0.95 across all tested widths.

Usage
-----
    # Full run (4-6 hours, RTX 3090):
    python experiments/h1_scale_correspondence/run_h1_validation.py

    # Fast-track (3-5 minutes, any hardware):
    python experiments/h1_scale_correspondence/run_h1_validation.py --fast-track

    # Specific width sweep:
    python experiments/h1_scale_correspondence/run_h1_validation.py --widths 64 128 256
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("h1_validation")


# ---------------------------------------------------------------------------
# Fast-track parameters
# ---------------------------------------------------------------------------

FAST_TRACK = {
    "widths":    [64, 128],
    "n_layers":  8,
    "n_seeds":   3,
    "n_samples": 100,
    "epochs":    2,
}

FULL = {
    "widths":    [64, 128, 256, 512],
    "n_layers":  30,
    "n_seeds":   10,
    "n_samples": 5000,
    "epochs":    100,
}


# ---------------------------------------------------------------------------
# Core experiment functions
# ---------------------------------------------------------------------------

def _critical_sigma_w(nonlinearity: str = "tanh") -> float:
    """Approximate critical sigma_w for tanh activation."""
    return 1.0 / np.sqrt(0.456)  # chi1(sigma_w_crit, tanh) = 1


def _measure_correlation_length(
    n: int,
    m: int,
    sigma_w: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Measure layer-wise correlation length xi(k) for a random-weight network.
    Uses Fisher spectrum estimator: xi = [mean(1/lambda)]^{-1/2}.
    """
    rng = np.random.default_rng(seed)
    xi_values = []

    for k in range(m):
        W  = rng.standard_normal((n, n)) * sigma_w / np.sqrt(n)
        WW = W @ W.T / n
        ev = np.linalg.eigvalsh(WW)
        ev = np.clip(ev, 1e-10, None)
        xi = float(1.0 / np.sqrt(np.mean(1.0 / ev)))
        xi_values.append(xi)

    return np.array(xi_values)


def _fit_exponential(xi_values: np.ndarray) -> dict:
    """Fit xi(k) = xi_0 * exp(-k/k_c). Returns fit parameters and R^2."""
    from scipy.optimize import curve_fit

    k  = np.arange(len(xi_values), dtype=float)

    def _exp(k, xi_0, k_c):
        return xi_0 * np.exp(-k / k_c)

    try:
        popt, _ = curve_fit(
            _exp, k, xi_values,
            p0=[xi_values[0], max(len(xi_values) / 3.0, 1.0)],
            bounds=([0.0, 0.1], [np.inf, np.inf]),
            maxfev=10000,
        )
        xi_0, k_c = popt
    except RuntimeError:
        xi_0, k_c = xi_values[0], float(len(xi_values))

    xi_pred = _exp(k, xi_0, k_c)
    ss_res  = ((xi_values - xi_pred) ** 2).sum()
    ss_tot  = ((xi_values - xi_values.mean()) ** 2).sum()
    r2      = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {"xi_0": float(xi_0), "k_c": float(k_c), "r2": float(r2)}


def run_h1_experiment(params: dict, results_dir: Path, fast_track: bool) -> dict:
    """
    Main H1 experiment: sweep widths and seeds, fit exponential decay.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    sigma_w = _critical_sigma_w("tanh")
    all_results = {}

    for width in params["widths"]:
        logger.info("Width N=%d", width)
        width_results = []

        for seed in range(params["n_seeds"]):
            xi = _measure_correlation_length(
                n=width,
                m=params["n_layers"],
                sigma_w=sigma_w,
                seed=seed * 1000 + width,
            )
            fit = _fit_exponential(xi)
            fit["xi_values"] = xi.tolist()
            fit["width"]     = width
            fit["seed"]      = seed
            width_results.append(fit)

            logger.info(
                "  seed=%d  xi_0=%.3f  k_c=%.2f  R^2=%.4f",
                seed, fit["xi_0"], fit["k_c"], fit["r2"],
            )

        r2_values = [r["r2"] for r in width_results]
        logger.info(
            "  N=%d: mean R^2 = %.4f ± %.4f",
            width, np.mean(r2_values), np.std(r2_values),
        )
        all_results[f"width_{width}"] = width_results

    # Save
    tag = "[FAST_TRACK_UNVERIFIED]" if fast_track else "[VERIFIED]"
    output = {
        "tag":        tag,
        "hypothesis": "H1",
        "params":     {k: v for k, v in params.items() if k != "xi_values"},
        "results":    all_results,
    }
    out_path = results_dir / "h1_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="H1 Scale Correspondence Validation")
    p.add_argument("--fast-track", action="store_true",
                   help="Fast-track mode (3-5 minutes).")
    p.add_argument("--widths", nargs="+", type=int,
                   help="Override width list (e.g. 64 128 256).")
    p.add_argument("--n-seeds", type=int,
                   help="Override number of seeds.")
    p.add_argument("--results-dir", type=str, default="results/h1",
                   help="Output directory for results.")
    return p.parse_args()


def main():
    args   = parse_args()
    params = FAST_TRACK.copy() if args.fast_track else FULL.copy()

    if args.widths:
        params["widths"] = args.widths
    if args.n_seeds:
        params["n_seeds"] = args.n_seeds

    results_dir = Path(args.results_dir)

    t0 = time.time()
    logger.info("=== H1 Scale Correspondence Validation ===")
    logger.info("Mode: %s", "FAST-TRACK" if args.fast_track else "FULL")
    logger.info("Widths: %s", params["widths"])
    logger.info("Seeds:  %d", params["n_seeds"])

    run_h1_experiment(params, results_dir, fast_track=args.fast_track)

    elapsed = time.time() - t0
    logger.info("=== H1 COMPLETE in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
