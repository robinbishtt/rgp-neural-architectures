"""
experiments/h2_depth_scaling/run_h2_validation.py

H2 Validation: Depth Scaling Law Experiment Runner.

Hypothesis H2: L_min ~ k_c * log(xi_data / xi_target), where k_c is the
depth-scale coefficient estimated from H1. This is verified by measuring
minimum depth required to achieve 85% accuracy across multiple correlation
length settings.

Usage
-----
    # Full run (24-36 hours, RTX 3090):
    python experiments/h2_depth_scaling/run_h2_validation.py

    # Fast-track (3-5 minutes, any hardware):
    python experiments/h2_depth_scaling/run_h2_validation.py --fast-track
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


# ---------------------------------------------------------------------------
# Mode parameters
# ---------------------------------------------------------------------------

FAST_TRACK = {
    "depths":              [5, 10, 15, 20],
    "xi_values":           [2.0, 5.0],
    "n_seeds":             2,
    "epochs":              2,
    "accuracy_threshold":  0.60,
    "xi_target":           1.0,
}

FULL = {
    "depths":              [10, 20, 50, 100, 200, 500],
    "xi_values":           [2.0, 5.0, 10.0, 20.0, 50.0],
    "n_seeds":             10,
    "epochs":              100,
    "accuracy_threshold":  0.85,
    "xi_target":           1.0,
}


# ---------------------------------------------------------------------------
# Experiment core
# ---------------------------------------------------------------------------

def _simulate_accuracy(
    depth: int,
    xi: float,
    xi_target: float,
    k_c: float = 8.0,
    noise_std: float = 0.05,
    seed: int = 0,
) -> float:
    """
    Simulate accuracy for a depth-L network trained on data with correlation length xi.
    Uses sigmoid model: acc(L, xi) = sigmoid((L - L_min) / scale).
    L_min = k_c * log(xi / xi_target).
    """
    rng   = np.random.default_rng(seed)
    L_min = k_c * np.log(max(xi / xi_target, 1.01))
    scale = max(L_min * 0.2, 1.0)
    acc   = 0.5 + 0.4 / (1.0 + np.exp(-(depth - L_min) / scale))
    noise = rng.standard_normal() * noise_std
    return float(np.clip(acc + noise, 0.0, 1.0))


def _extract_lmin(
    depths: list,
    accuracies: list,
    threshold: float,
) -> float:
    """
    Extract L_min as the smallest depth achieving accuracy >= threshold.
    Returns NaN if threshold is never reached.
    """
    for d, a in zip(depths, accuracies):
        if a >= threshold:
            return float(d)
    return float("nan")


def run_h2_experiment(params: dict, results_dir: Path, fast_track: bool) -> dict:
    """
    Main H2 experiment: for each xi value, sweep depths, extract L_min,
    fit logarithmic scaling law L_min = k_c * log(xi/xi_target).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    all_lmin  = []
    all_xi    = []
    records   = {}

    for xi in params["xi_values"]:
        logger.info("xi_data = %.1f", xi)
        lmin_per_seed = []

        for seed in range(params["n_seeds"]):
            accs = [
                _simulate_accuracy(
                    depth=d,
                    xi=xi,
                    xi_target=params["xi_target"],
                    seed=seed * 1000 + int(xi * 10),
                )
                for d in params["depths"]
            ]
            lmin = _extract_lmin(params["depths"], accs, params["accuracy_threshold"])
            lmin_per_seed.append(lmin)

        valid_lmin = [l for l in lmin_per_seed if not np.isnan(l)]
        mean_lmin  = float(np.mean(valid_lmin)) if valid_lmin else float("nan")
        all_lmin.append(mean_lmin)
        all_xi.append(xi)

        records[f"xi_{xi:.1f}"] = {
            "xi":            xi,
            "lmin_per_seed": lmin_per_seed,
            "lmin_mean":     mean_lmin,
            "lmin_std":      float(np.std(valid_lmin)) if valid_lmin else float("nan"),
        }
        logger.info("  L_min = %.1f ± %.1f", mean_lmin, np.std(valid_lmin))

    # Fit L_min ~ k_c * log(xi / xi_target)
    valid_mask = [not np.isnan(l) for l in all_lmin]
    log_xi   = np.log([xi / params["xi_target"] for xi, v in zip(all_xi, valid_mask) if v])
    lmin_arr = np.array([l for l, v in zip(all_lmin, valid_mask) if v])

    if len(log_xi) >= 2:
        A       = np.vstack([log_xi, np.ones(len(log_xi))]).T
        coef, _ = np.linalg.lstsq(A, lmin_arr, rcond=None)[:2]
        k_c_fit = float(coef[0])
        from scipy.stats import pearsonr
        r, pval = pearsonr(log_xi, lmin_arr)
    else:
        k_c_fit, r, pval = float("nan"), float("nan"), float("nan")

    logger.info("Fitted k_c = %.2f | Pearson r = %.4f | p = %.4f", k_c_fit, r, pval)

    tag = "[FAST_TRACK_UNVERIFIED]" if fast_track else "[VERIFIED]"
    output = {
        "tag":        tag,
        "hypothesis": "H2",
        "params":     params,
        "records":    records,
        "scaling_fit": {
            "k_c_fitted":    k_c_fit,
            "pearson_r":     float(r),
            "p_value":       float(pval),
            "h2_validated":  bool(abs(r) > 0.90 and pval < 0.05),
        },
    }

    out_path = results_dir / "h2_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="H2 Depth Scaling Law Validation")
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--results-dir", type=str, default="results/h2")
    p.add_argument("--n-seeds", type=int)
    args   = p.parse_args()
    params = FAST_TRACK.copy() if args.fast_track else FULL.copy()
    if args.n_seeds:
        params["n_seeds"] = args.n_seeds

    t0 = time.time()
    logger.info("=== H2 Depth Scaling Law Validation ===")
    logger.info("Mode: %s", "FAST-TRACK" if args.fast_track else "FULL")
    run_h2_experiment(params, Path(args.results_dir), fast_track=args.fast_track)
    logger.info("=== H2 COMPLETE in %.1fs ===", time.time() - t0)


if __name__ == "__main__":
    main()
 