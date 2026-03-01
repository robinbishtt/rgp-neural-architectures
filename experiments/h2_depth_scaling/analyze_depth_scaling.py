"""
experiments/h2_depth_scaling/analyze_depth_scaling.py

Logarithmic fitting and exponent extraction for H2 depth scaling data.
Compares log, linear, and power-law model fits using AIC.

Usage
-----
    python experiments/h2_depth_scaling/analyze_depth_scaling.py \
        --results-dir results/h2 \
        --output results/h2/depth_scaling_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _log_model(log_xi: np.ndarray, k_c: float, intercept: float) -> np.ndarray:
    """L_min = k_c * log(xi) + intercept."""
    return k_c * log_xi + intercept


def _linear_model(xi: np.ndarray, a: float, b: float) -> np.ndarray:
    """L_min = a * xi + b."""
    return a * xi + b


def _power_model(xi: np.ndarray, a: float, nu: float) -> np.ndarray:
    """L_min = a * xi^nu."""
    return a * np.power(np.maximum(xi, 1e-6), nu)


def _compute_aic(n: int, k_params: int, rss: float) -> float:
    """AIC = n * log(RSS/n) + 2k."""
    return n * np.log(rss / n + 1e-12) + 2.0 * k_params


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compare_models(xi_values: np.ndarray, lmin_values: np.ndarray) -> Dict:
    """
    Fit logarithmic, linear, and power-law models to L_min vs xi_data.
    Returns AIC, fitted parameters, and R^2 for each model.
    """
    log_xi = np.log(xi_values)
    n      = len(xi_values)
    results = {}

    # --- Logarithmic ---
    A_log   = np.vstack([log_xi, np.ones(n)]).T
    coef_log, _, _, _ = np.linalg.lstsq(A_log, lmin_values, rcond=None)
    rss_log = ((lmin_values - A_log @ coef_log) ** 2).sum()
    results["logarithmic"] = {
        "k_c":       float(coef_log[0]),
        "intercept": float(coef_log[1]),
        "aic":       _compute_aic(n, 2, rss_log),
        "rss":       float(rss_log),
    }

    # --- Linear ---
    A_lin   = np.vstack([xi_values, np.ones(n)]).T
    coef_lin, _, _, _ = np.linalg.lstsq(A_lin, lmin_values, rcond=None)
    rss_lin = ((lmin_values - A_lin @ coef_lin) ** 2).sum()
    results["linear"] = {
        "slope":     float(coef_lin[0]),
        "intercept": float(coef_lin[1]),
        "aic":       _compute_aic(n, 2, rss_lin),
        "rss":       float(rss_lin),
    }

    # --- Power-law ---
    try:
        popt_pow, _ = curve_fit(
            _power_model, xi_values, lmin_values,
            p0=[1.0, 1.0], bounds=([0.0, 0.0], [np.inf, 5.0]),
        )
        rss_pow = ((lmin_values - _power_model(xi_values, *popt_pow)) ** 2).sum()
        results["power_law"] = {
            "a":   float(popt_pow[0]),
            "nu":  float(popt_pow[1]),
            "aic": _compute_aic(n, 2, rss_pow),
            "rss": float(rss_pow),
        }
    except RuntimeError:
        results["power_law"] = {"aic": float("inf"), "rss": float("inf")}

    # Identify best model
    best_model = min(results, key=lambda m: results[m]["aic"])
    delta_aic  = {m: results[m]["aic"] - results[best_model]["aic"] for m in results}
    r, pval    = pearsonr(log_xi, lmin_values)

    return {
        "models":     results,
        "best_model": best_model,
        "delta_aic":  delta_aic,
        "pearson_r":  float(r),
        "p_value":    float(pval),
        "log_model_wins": best_model == "logarithmic",
    }


def run_analysis(results_dir: Path, output_path: Path) -> Dict:
    """Load H2 results and perform depth scaling analysis."""
    results_file = results_dir / "h2_results.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file} — run run_h2_validation.py first."
        )

    with open(results_file) as fh:
        raw = json.load(fh)

    xi_vals  = []
    lmin_vals = []
    for rec in raw["records"].values():
        if not np.isnan(rec["lmin_mean"]):
            xi_vals.append(rec["xi"])
            lmin_vals.append(rec["lmin_mean"])

    xi_arr   = np.array(xi_vals)
    lmin_arr = np.array(lmin_vals)

    logger.info("Fitting %d (xi, L_min) pairs", len(xi_arr))
    comparison = compare_models(xi_arr, lmin_arr)

    logger.info("Best model: %s", comparison["best_model"])
    logger.info("Fitted k_c = %.3f", comparison["models"]["logarithmic"]["k_c"])
    logger.info("Pearson r = %.4f (p = %.4f)", comparison["pearson_r"], comparison["p_value"])
    logger.info("H2 log model wins: %s", comparison["log_model_wins"])

    output = {
        "tag":         raw.get("tag", ""),
        "hypothesis":  "H2",
        "n_data_points": len(xi_arr),
        "analysis":    comparison,
        "h2_validated": bool(
            comparison["log_model_wins"]
            and abs(comparison["pearson_r"]) > 0.90
            and comparison["p_value"] < 0.05
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Analysis saved to %s", output_path)
    return output


def main():
    p = argparse.ArgumentParser(description="H2 Depth Scaling Analysis")
    p.add_argument("--results-dir", type=str, default="results/h2")
    p.add_argument("--output",      type=str, default="results/h2/depth_scaling_analysis.json")
    args = p.parse_args()
    run_analysis(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
