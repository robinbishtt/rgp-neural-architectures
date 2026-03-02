"""
experiments/h3_multiscale_generalization/compare_architectures.py

Statistical comparison of RG-Net vs baseline architectures.
Generates the full significance table (Table 1) and Wilcoxon heatmap data.

Usage
-----
    python experiments/h3_multiscale_generalization/compare_architectures.py \
        --results-dir results/h3 \
        --output results/h3/architecture_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

SIGNIFICANCE_STARS = {0.001: "***", 0.01: "**", 0.05: "*", 1.01: "n.s."}


def _stars(pval: float) -> str:
    for threshold, star in sorted(SIGNIFICANCE_STARS.items()):
        if pval < threshold:
            return star
    return "n.s."


def _confidence_interval(values: list, confidence: float = 0.95) -> tuple:
    arr   = np.array(values)
    n     = len(arr)
    sem   = arr.std(ddof=1) / np.sqrt(n)
    from scipy.stats import t
    t_crit = t.ppf((1 + confidence) / 2, df=n - 1)
    return (float(arr.mean() - t_crit * sem), float(arr.mean() + t_crit * sem))


def _effect_size_cohens_d(a: list, b: list) -> float:
    """Cohen's d effect size for paired comparison."""
    a_arr, b_arr = np.array(a), np.array(b)
    diff  = a_arr - b_arr
    pooled_std = np.sqrt((a_arr.var(ddof=1) + b_arr.var(ddof=1)) / 2)
    return float(diff.mean() / (pooled_std + 1e-12))


def compare_all_baselines(h3_results: Dict) -> Dict:
    """
    Perform pairwise statistical comparisons: RG-Net vs each baseline
    on IID, hierarchical, and mean accuracy.
    """
    records   = h3_results.get("records", {})
    baselines = [k for k in records if k != "rgnet"]
    datasets  = ["iid", "hier"]

    summary = {}
    rgnet   = records.get("rgnet", {})

    for baseline in baselines:
        base_data = records.get(baseline, {})
        baseline_summary = {}

        for ds in datasets:
            rgnet_vals = rgnet.get("accuracies", {}).get(ds, [])
            base_vals  = base_data.get("accuracies", {}).get(ds, [])

            if not rgnet_vals or not base_vals:
                continue

            rgnet_arr = np.array(rgnet_vals)
            base_arr  = np.array(base_vals)

            try:
                stat, pval = wilcoxon(rgnet_arr, base_arr, alternative="greater")
            except ValueError:
                from scipy.stats import ttest_rel
                stat, pval = ttest_rel(rgnet_arr, base_arr)

            ci_rgnet = _confidence_interval(rgnet_vals)
            ci_base  = _confidence_interval(base_vals)
            d        = _effect_size_cohens_d(rgnet_vals, base_vals)

            baseline_summary[ds] = {
                "rgnet_mean":   float(rgnet_arr.mean()),
                "rgnet_std":    float(rgnet_arr.std(ddof=1)),
                "rgnet_ci":     list(ci_rgnet),
                "baseline_mean": float(base_arr.mean()),
                "baseline_std":  float(base_arr.std(ddof=1)),
                "baseline_ci":   list(ci_base),
                "advantage":     float(rgnet_arr.mean() - base_arr.mean()),
                "wilcoxon_stat": float(stat),
                "p_value":       float(pval),
                "stars":         _stars(float(pval)),
                "cohens_d":      d,
            }
            logger.info(
                "  RGNet vs %s (%s): adv=%.4f, p=%.4f %s, d=%.2f",
                baseline, ds,
                baseline_summary[ds]["advantage"],
                pval, _stars(pval), d,
            )

        summary[baseline] = baseline_summary

    # Build Table 1 LaTeX fragment
    table_lines = [
        r"\begin{table}",
        r"\caption{Architecture comparison on IID and hierarchical datasets.}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & IID Accuracy & Hier. Accuracy & $\Delta$ (Hier) & Significance \\",
        r"\midrule",
    ]
    rgnet_hier = float(np.mean(rgnet.get("accuracies", {}).get("hier", [0])))
    table_lines.append(
        rf"RG-Net & {float(np.mean(rgnet.get('accuracies', {{}}).get('iid', [0]))):.3f} "
        rf"& {rgnet_hier:.3f} & --- & --- \\"
    )
    for baseline, data in summary.items():
        iid_data  = data.get("iid",  {})
        hier_data = data.get("hier", {})
        iid_acc   = iid_data.get("baseline_mean",  0.0)
        hier_acc  = hier_data.get("baseline_mean", 0.0)
        adv_hier  = hier_data.get("advantage",     0.0)
        stars     = hier_data.get("stars", "n.s.")
        table_lines.append(
            rf"{baseline.capitalize()} & {iid_acc:.3f} & {hier_acc:.3f} "
            rf"& {adv_hier:+.3f} & {stars} \\"
        )
    table_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return {"pairwise": summary, "table1_latex": "\n".join(table_lines)}


def run_comparison(results_dir: Path, output_path: Path) -> Dict:
    results_file = results_dir / "h3_results.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file} — run run_h3_validation.py first."
        )

    with open(results_file) as fh:
        h3_results = json.load(fh)

    logger.info("=== Architecture Comparison ===")
    comparison = compare_all_baselines(h3_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2))

    # Also write Table 1 LaTeX separately
    table_path = output_path.parent / "table1.tex"
    table_path.write_text(comparison.get("table1_latex", ""))
    logger.info("Architecture comparison saved to %s", output_path)
    logger.info("Table 1 LaTeX saved to %s", table_path)
    return comparison


def main():
    p = argparse.ArgumentParser(description="H3 Architecture Comparison")
    p.add_argument("--results-dir", type=str, default="results/h3")
    p.add_argument("--output",      type=str, default="results/h3/architecture_comparison.json")
    args = p.parse_args()
    run_comparison(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
 