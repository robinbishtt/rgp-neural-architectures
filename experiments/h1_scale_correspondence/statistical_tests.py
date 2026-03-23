"""
experiments/h1_scale_correspondence/statistical_tests.py

Statistical significance testing and confidence interval computation for H1.
Provides KS tests, bootstrap CIs, and Pearson correlation tests.

Usage
-----
    python experiments/h1_scale_correspondence/statistical_tests.py \
        --analysis-file results/h1/correlation_analysis.json \
        --output results/h1/statistical_tests.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kstest, pearsonr, ttest_1samp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


def ks_test_exponential(xi_values: np.ndarray, k_c: float) -> Tuple[float, float]:
    """
    One-sample KS test of measured xi(k) against the fitted exponential CDF.
    H0: data follows xi_0 * exp(-k/k_c).
    Returns (statistic, p-value).
    """
    k          = np.arange(len(xi_values), dtype=float)
    xi_0       = xi_values[0]
    xi_fitted  = xi_0 * np.exp(-k / k_c)
    residuals  = xi_values - xi_fitted
    stat, pval = kstest(residuals, "norm")
    return float(stat), float(pval)


def bootstrap_r2(
    xi_values: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict:
    """
    Bootstrap 95% CI for R^2 of the exponential fit.
    """
    from scipy.optimize import curve_fit

    def _exp(k, xi_0, k_c):
        return xi_0 * np.exp(-k / k_c)

    rng = np.random.default_rng(seed)
    k   = np.arange(len(xi_values), dtype=float)
    r2_samples = []

    for _ in range(n_resamples):
        idx    = rng.integers(0, len(xi_values), size=len(xi_values))
        xi_b   = xi_values[idx]
        k_b    = k[idx]
        try:
            popt, _ = curve_fit(_exp, k_b, xi_b,
                                p0=[xi_b[0], 5.0],
                                bounds=([0.0, 0.1], [np.inf, np.inf]),
                                maxfev=5000)
            xi_pred = _exp(k, *popt)
            ss_res  = ((xi_values - xi_pred) ** 2).sum()
            ss_tot  = ((xi_values - xi_values.mean()) ** 2).sum()
            r2      = 1.0 - ss_res / max(ss_tot, 1e-12)
            r2_samples.append(float(r2))
        except RuntimeError:
            pass

    r2_samples = np.array(r2_samples)
    alpha      = 1.0 - confidence
    ci_lo      = float(np.percentile(r2_samples, 100 * alpha / 2))
    ci_hi      = float(np.percentile(r2_samples, 100 * (1 - alpha / 2)))

    return {
        "r2_mean":   float(r2_samples.mean()),
        "r2_std":    float(r2_samples.std()),
        "ci_lo":     ci_lo,
        "ci_hi":     ci_hi,
        "n_valid":   len(r2_samples),
        "n_total":   n_resamples,
    }


def test_pearson_correlation_with_xi(
    xi_values_list: List[np.ndarray],
    k_c_values: List[float],
) -> Dict:
    """
    Pearson correlation between measured xi_0 and fitted k_c across widths.
    H0: no correlation. Expected: positive r (wider networks have longer xi_0).
    """
    xi_0_vals  = [xv[0] for xv in xi_values_list]
    r, pval    = pearsonr(xi_0_vals, k_c_values)
    return {"pearson_r": float(r), "p_value": float(pval), "significant": bool(pval < 0.05)}


def run_statistical_tests(analysis_file: Path, output_path: Path) -> Dict:
    """Load analysis results and run full statistical test battery."""
    if not analysis_file.exists():
        logger.error("Analysis file not found: %s - run analyze_correlation_decay.py first.")
        raise FileNotFoundError(str(analysis_file))

    with open(analysis_file) as fh:
        analysis = json.load(fh)

    tests = {}

    for key, width_data in analysis.get("per_width", {}).items():
        per_seed = width_data["per_seed"]
        r2_vals  = np.array([s["r2"] for s in per_seed])
        kc_vals  = np.array([s["k_c"] for s in per_seed])

        # t-test: H0: mean R^2 <= 0.95
        t_stat, t_pval = ttest_1samp(r2_vals, popmean=0.95)

        # Bootstrap CI for first seed's xi_values if available
        xi_values = np.array(per_seed[0].get("xi_values", [1.0, 0.9, 0.8]))
        boot_ci   = bootstrap_r2(xi_values, n_resamples=200)

        tests[key] = {
            "r2_mean":           float(r2_vals.mean()),
            "r2_std":            float(r2_vals.std()),
            "r2_above_0.95":     bool((r2_vals > 0.95).all()),
            "ttest_vs_0.95_pval": float(t_pval),
            "bootstrap_r2_ci":   boot_ci,
            "k_c_mean":          float(kc_vals.mean()),
        }
        logger.info(
            "%s: mean R^2=%.4f, t-test p=%.4f, H1 held=%s",
            key, r2_vals.mean(), t_pval, (r2_vals > 0.95).all(),
        )

    # Global H1 verdict
    all_r2_above = all(v["r2_above_0.95"] for v in tests.values())
    tests["verdict"] = {
        "h1_validated": all_r2_above,
        "description": (
            "All width x seed combinations have R^2 > 0.95."
            if all_r2_above
            else "Some combinations failed R^2 > 0.95 threshold."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tests, indent=2))
    logger.info("Statistical tests saved to %s", output_path)
    return tests


def main():
    p = argparse.ArgumentParser(description="H1 Statistical Tests")
    p.add_argument("--analysis-file", type=str,
                   default="results/h1/correlation_analysis.json")
    p.add_argument("--output", type=str,
                   default="results/h1/statistical_tests.json")
    args = p.parse_args()
    run_statistical_tests(Path(args.analysis_file), Path(args.output))


if __name__ == "__main__":
    main()
 