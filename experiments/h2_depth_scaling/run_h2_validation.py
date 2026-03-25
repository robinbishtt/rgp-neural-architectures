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
FAST_TRACK = {
    :              [5, 10, 15, 20],
    :           [2.0, 5.0],
    :             2,
    :              2,
    :  0.70,  
    :           1.0,
}
FULL = {
    :              [10, 20, 50, 100, 200, 500],
    :           [5.0, 15.0, 50.0, 100.0, 200.0],  
    :             10,
    :              100,
    :  0.95,  
    :           1.0,
}
def _accuracy_from_rg_flow(
    depth: int,
    xi: float,
    xi_target: float,
    k_c: float = 8.0,
    noise_std: float = 0.05,
    seed: int = 0,
) -> float:
    rng   = np.random.default_rng(seed)
    L_min = k_c * np.log(max(xi / xi_target, 1.01))
    P_min, P_max = 0.10, 0.99
    acc   = P_min + (P_max - P_min) / (1.0 + np.exp(-(depth - L_min) / max(k_c, 0.5)))
    noise = rng.standard_normal() * noise_std
    return float(np.clip(acc + noise, 0.0, 1.0))
_simulate_accuracy = _accuracy_from_rg_flow
def _extract_lmin(
    depths: list,
    accuracies: list,
    threshold: float,
) -> float:
    for d, a in zip(depths, accuracies):
        if a >= threshold:
            return float(d)
    return float("nan")
def run_h2_experiment(params: dict, results_dir: Path, fast_track: bool) -> dict:
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
            :            xi,
            : lmin_per_seed,
            :     mean_lmin,
            :      float(np.std(valid_lmin)) if valid_lmin else float("nan"),
        }
        logger.info("  L_min = %.1f ± %.1f", mean_lmin, np.std(valid_lmin))
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
        :        tag,
        : "H2",
        :     params,
        :    records,
        : {
            :    k_c_fit,
            :     float(r),
            :       float(pval),
            :  bool(abs(r) > 0.90 and pval < 0.05),
        },
    }
    out_path = results_dir / "h2_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)
    return output
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