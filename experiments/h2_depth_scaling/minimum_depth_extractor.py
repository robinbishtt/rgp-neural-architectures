from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
def extract_lmin_interpolated(
    depths: List[int],
    accuracies: List[float],
    threshold: float = 0.85,
) -> Optional[float]:
    depths_arr = np.array(depths, dtype=float)
    accs_arr   = np.array(accuracies, dtype=float)
    if accs_arr.max() < threshold:
        return None  
    idx = np.searchsorted(accs_arr, threshold)
    if idx == 0:
        return float(depths_arr[0])  
    d0, d1 = depths_arr[idx - 1], depths_arr[idx]
    a0, a1 = accs_arr[idx - 1], accs_arr[idx]
    if abs(a1 - a0) < 1e-12:
        return float(d0)
    lmin_interp = d0 + (threshold - a0) / (a1 - a0) * (d1 - d0)
    return float(lmin_interp)
def bootstrap_lmin(
    depths: List[int],
    accuracies_per_seed: List[List[float]],
    threshold: float = 0.85,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict:
    rng     = np.random.default_rng(seed)
    n_seeds = len(accuracies_per_seed)
    lmin_bootstrap = []
    for _ in range(n_resamples):
        idx   = rng.integers(0, n_seeds, size=n_seeds)
        accs  = np.mean([accuracies_per_seed[i] for i in idx], axis=0)
        lmin  = extract_lmin_interpolated(depths, accs.tolist(), threshold)
        if lmin is not None:
            lmin_bootstrap.append(lmin)
    if not lmin_bootstrap:
        return {"lmin_mean": None, "ci_lo": None, "ci_hi": None, "n_valid": 0}
    arr   = np.array(lmin_bootstrap)
    alpha = 1.0 - confidence
    return {
        :  float(arr.mean()),
        :   float(arr.std()),
        :      float(np.percentile(arr, 100 * alpha / 2)),
        :      float(np.percentile(arr, 100 * (1 - alpha / 2))),
        :    len(arr),
        :    n_resamples,
    }
def run_lmin_extraction(
    results_dir: Path,
    output_path: Path,
    threshold: float = 0.85,
) -> Dict:
    results_file = results_dir / "h2_results.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file} - run run_h2_validation.py first."
        )
    with open(results_file) as fh:
        raw = json.load(fh)
    params  = raw.get("params", {})
    params.get("depths", [10, 20, 50, 100, 200])
    records = raw.get("records", {})
    output  = {"threshold": threshold, "per_xi": {}}
    for key, rec in records.items():
        xi = rec["xi"]
        logger.info("Extracting L_min for xi=%.1f", xi)
        if "lmin_mean" in rec and rec["lmin_mean"]:
            lmin_mean = rec["lmin_mean"]
            lmin_std  = rec.get("lmin_std", 0.0)
        else:
            lmin_mean = None
            lmin_std  = None
        output["per_xi"][key] = {
            :         xi,
            :  lmin_mean,
            :   lmin_std,
            :  threshold,
            :     float(np.log(xi / params.get("xi_target", 1.0))),
        }
        logger.info("  L_min = %.1f ± %.1f", lmin_mean or 0, lmin_std or 0)
    valid = [(v["log_xi"], v["lmin_mean"]) for v in output["per_xi"].values()
             if v["lmin_mean"] is not None]
    if len(valid) >= 2:
        log_xi  = np.array([v[0] for v in valid])
        lmin_arr = np.array([v[1] for v in valid])
        A = np.vstack([log_xi, np.ones(len(log_xi))]).T
        coef, _, _, _ = np.linalg.lstsq(A, lmin_arr, rcond=None)
        output["scaling_fit"] = {
            :   float(coef[0]),
            :    float(coef[1]),
        }
        logger.info("Scaling fit: k_c = %.3f", coef[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("L_min extraction saved to %s", output_path)
    return output
def main():
    p = argparse.ArgumentParser(description="L_min Extraction from H2 Results")
    p.add_argument("--results-dir", type=str, default="results/h2")
    p.add_argument("--output",      type=str, default="results/h2/lmin_extraction.json")
    p.add_argument("--threshold",   type=float, default=0.85)
    args = p.parse_args()
    run_lmin_extraction(Path(args.results_dir), Path(args.output), args.threshold)
if __name__ == "__main__":
    main()