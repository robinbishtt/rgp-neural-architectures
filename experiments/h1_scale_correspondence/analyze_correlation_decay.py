from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
def _exp_decay(k: np.ndarray, xi_0: float, k_c: float) -> np.ndarray:
    return xi_0 * np.exp(-k / k_c)
def fit_single_decay(xi_values: np.ndarray) -> Dict:
    k = np.arange(len(xi_values), dtype=float)
    try:
        popt, pcov = curve_fit(
            _exp_decay, k, xi_values,
            p0=[xi_values[0], max(len(xi_values) / 3.0, 1.0)],
            bounds=([0.0, 0.1], [np.inf, np.inf]),
            maxfev=10000,
        )
        xi_0, k_c = popt
        perr       = np.sqrt(np.diag(pcov))
        df     = len(xi_values) - 2
        t_crit = float(t_dist.ppf(0.975, df)) if df > 0 else 2.0
        xi_0_ci = (xi_0 - t_crit * perr[0], xi_0 + t_crit * perr[0])
        k_c_ci  = (k_c  - t_crit * perr[1], k_c  + t_crit * perr[1])
    except RuntimeError:
        xi_0, k_c  = xi_values[0], float(len(xi_values))
        xi_0_ci    = (0.0, 0.0)
        k_c_ci     = (0.0, 0.0)
    xi_pred = _exp_decay(k, xi_0, k_c)
    ss_res  = ((xi_values - xi_pred) ** 2).sum()
    ss_tot  = ((xi_values - xi_values.mean()) ** 2).sum()
    r2      = float(1.0 - ss_res / max(ss_tot, 1e-12))
    chi1    = float(np.exp(-1.0 / k_c)) if k_c > 0 else 0.0
    return {
        :     float(xi_0),
        :  [float(xi_0_ci[0]), float(xi_0_ci[1])],
        :      float(k_c),
        :   [float(k_c_ci[0]), float(k_c_ci[1])],
        :       r2,
        :     chi1,
    }
def analyze_width_group(records: List[Dict]) -> Dict:
    per_seed_fits = []
    for rec in records:
        xi_values = np.array(rec["xi_values"])
        fit       = fit_single_decay(xi_values)
        fit["seed"] = rec["seed"]
        per_seed_fits.append(fit)
    r2_vals  = np.array([f["r2"]  for f in per_seed_fits])
    kc_vals  = np.array([f["k_c"] for f in per_seed_fits])
    chi1_vals= np.array([f["chi1"] for f in per_seed_fits])
    return {
        :       per_seed_fits,
        :        float(r2_vals.mean()),
        :         float(r2_vals.std()),
        :       float(kc_vals.mean()),
        :        float(kc_vals.std()),
        :      float(chi1_vals.mean()),
        :       float(chi1_vals.std()),
        :    bool((r2_vals > 0.95).all()),
        :        len(per_seed_fits),
    }
def run_analysis(results_dir: Path, output_path: Path) -> Dict:
    results_file = results_dir / "h1_results.json"
    if not results_file.exists():
        logger.error("Results file not found: %s", results_file)
        logger.info("Run run_h1_validation.py first.")
        raise FileNotFoundError(str(results_file))
    with open(results_file) as fh:
        raw = json.load(fh)
    analysis = {
        :       raw.get("tag", ""),
        : {},
    }
    all_r2 = []
    for key, records in raw["results"].items():
        width = records[0]["width"] if records else 0
        logger.info("Analysing width N=%d (%d seeds)", width, len(records))
        summary = analyze_width_group(records)
        analysis["per_width"][key] = summary
        all_r2.extend([f["r2"] for f in summary["per_seed"]])
        logger.info(
            ,
            width,
            summary["r2_mean"], summary["r2_std"],
            summary["k_c_mean"], summary["k_c_std"],
            summary["r2_above_95"],
        )
    all_r2 = np.array(all_r2)
    analysis["summary"] = {
        :   float(all_r2.mean()),
        :    float(all_r2.std()),
        :      bool((all_r2 > 0.90).mean() >= 0.90),
    }
    logger.info("H1 overall validated: %s", analysis["summary"]["h1_validated"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2))
    logger.info("Analysis saved to %s", output_path)
    return analysis
def main():
    p = argparse.ArgumentParser(description="Analyse H1 correlation decay results.")
    p.add_argument("--results-dir", type=str, default="results/h1")
    p.add_argument("--output",      type=str, default="results/h1/correlation_analysis.json")
    args = p.parse_args()
    run_analysis(Path(args.results_dir), Path(args.output))
if __name__ == "__main__":
    main()