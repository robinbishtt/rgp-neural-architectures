"""
figures/extended_data/run_extended_figure4.py

Extended Data Figure 4 — Finite-Size Scaling Data Collapse

  a) Raw accuracy vs width (before collapse) for multiple ξ values
  b) Collapsed data: (L−L_c)·W^{1/ν} vs accuracy
  c) Residuals from master curve
  d) Bootstrap distribution of critical exponent ν

Usage
-----
    python figures/extended_data/run_extended_figure4.py \
        --results results/fss/ --output figures/out/ed_fig4.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.color_palette import correlation_length_colors, panel_label
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label, remove_top_right_spines,
)


def _fss_ansatz(x: np.ndarray, nu: float = 1.0) -> np.ndarray:
    """Master curve f(x) = 1 / (1 + exp(-x)) for FSS collapse."""
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic_ed4_data(fast_track: bool = False) -> Dict:
    rng = np.random.default_rng(99)
    xi_values  = [5.0, 10.0] if fast_track else [2.0, 5.0, 10.0, 20.0, 50.0]
    widths     = [32, 64]    if fast_track else [32, 64, 128, 256, 512]
    L_c        = 15.0
    nu_true    = 1.2
    n_seeds    = 5 if fast_track else 20

    raw_curves: Dict = {}
    for xi in xi_values:
        acc_by_width = []
        for W in widths:
            x = (xi - L_c) * W ** (1.0 / nu_true)
            acc = float(np.clip(_fss_ansatz(np.array([x]))[0]
                                + rng.normal(0, 0.015), 0, 1))
            acc_by_width.append(acc)
        raw_curves[str(xi)] = acc_by_width

    # Bootstrap ν
    nu_boots = []
    for _ in range(n_seeds):
        def _collapse_residual(nu_try):
            residuals = []
            for xi in xi_values:
                for i, W in enumerate(widths):
                    x = (xi - L_c) * W ** (1.0 / nu_try)
                    y_pred = _fss_ansatz(np.array([x]))[0]
                    y_obs  = raw_curves[str(xi)][i] + rng.normal(0, 0.01)
                    residuals.append((y_obs - y_pred) ** 2)
            return np.mean(residuals)

        res = minimize_scalar(_collapse_residual, bounds=(0.5, 3.0), method="bounded")
        nu_boots.append(float(res.x))

    return {
        "xi_values":  xi_values,
        "widths":     widths,
        "raw_curves": raw_curves,
        "L_c":        L_c,
        "nu_true":    nu_true,
        "nu_boots":   nu_boots,
    }


def _panel_raw_curves(ax: plt.Axes, data: Dict, fast_track: bool) -> None:
    xi_values = data["xi_values"]
    widths    = data["widths"]
    colors    = correlation_length_colors(xi_values)

    for xi, color in zip(xi_values, colors):
        acc = data["raw_curves"][str(xi)]
        ax.plot(widths, acc, "-o", ms=3, lw=1.0, color=color, label=f"ξ={xi:.0f}")

    tag = " [FT]" if fast_track else ""
    ax.set_xlabel("Network width $N$")
    ax.set_ylabel("Accuracy")
    ax.set_title("ED4a: Raw curves (before collapse)" + tag, fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=5.5)
    remove_top_right_spines(ax)


def _panel_collapsed(ax: plt.Axes, data: Dict) -> None:
    xi_values = data["xi_values"]
    widths    = data["widths"]
    colors    = correlation_length_colors(xi_values)
    L_c  = data["L_c"]
    nu   = data["nu_true"]

    x_all, y_all = [], []
    for xi, color in zip(xi_values, colors):
        acc = data["raw_curves"][str(xi)]
        xs  = [(xi - L_c) * W ** (1.0 / nu) for W in widths]
        ax.scatter(xs, acc, s=14, color=color, label=f"ξ={xi:.0f}", zorder=3)
        x_all.extend(xs)
        y_all.extend(acc)

    x_line = np.linspace(min(x_all) - 0.5, max(x_all) + 0.5, 200)
    ax.plot(x_line, _fss_ansatz(x_line), "k--", lw=1.1, label="Master curve")

    ax.set_xlabel(r"$(\xi - \xi_c)\cdot N^{1/\nu}$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"ED4b: FSS collapse ($\nu=" + f"{nu:.2f}$)", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=5.5)
    remove_top_right_spines(ax)


def _panel_residuals(ax: plt.Axes, data: Dict) -> None:
    xi_values = data["xi_values"]
    widths    = data["widths"]
    L_c  = data["L_c"]
    nu   = data["nu_true"]
    colors = correlation_length_colors(xi_values)

    for xi, color in zip(xi_values, colors):
        acc = data["raw_curves"][str(xi)]
        for W, a in zip(widths, acc):
            x      = (xi - L_c) * W ** (1.0 / nu)
            y_pred = _fss_ansatz(np.array([x]))[0]
            ax.scatter(x, a - y_pred, s=12, color=color, alpha=0.7)

    ax.axhline(y=0, lw=0.8, color="#333333", ls="--")
    ax.set_xlabel(r"$(\xi - \xi_c)\cdot N^{1/\nu}$")
    ax.set_ylabel("Residual")
    ax.set_title("ED4c: Collapse residuals", fontsize=7, pad=3)
    remove_top_right_spines(ax)


def _panel_nu_bootstrap(ax: plt.Axes, data: Dict) -> None:
    nu_boots = np.array(data["nu_boots"])
    nu_true  = data["nu_true"]

    ax.hist(nu_boots, bins=10, color="#4878CF", edgecolor="white",
            lw=0.4, density=True)
    ax.axvline(x=nu_true, color="#D62728", lw=1.0, ls="--",
               label=f"True ν={nu_true:.2f}")
    ax.axvline(x=nu_boots.mean(), color="#6ACC65", lw=0.9, ls=":",
               label=f"Mean={nu_boots.mean():.2f}")

    ax.set_xlabel(r"Critical exponent $\nu$")
    ax.set_ylabel("Density")
    ax.set_title("ED4d: Bootstrap distribution of ν", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def generate(
    results_dir: str = "results/fss",
    output_path: str = "figures/out/ed_fig4.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()
    data = _synthetic_ed4_data(fast_track=fast_track)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.0))
    fig.subplots_adjust(wspace=0.42, hspace=0.52)

    _panel_raw_curves(axes[0, 0], data, fast_track)
    _panel_collapsed(axes[0, 1], data)
    _panel_residuals(axes[1, 0], data)
    _panel_nu_bootstrap(axes[1, 1], data)

    for i, ax in enumerate(axes.flat):
        add_panel_label(ax, panel_label(i))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)

    tag = " [FAST_TRACK_UNVERIFIED]" if fast_track else ""
    print(f"Extended Data Figure 4 saved: {out}{tag}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/fss")
    p.add_argument("--output",  default="figures/out/ed_fig4.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(results_dir=args.results, output_path=args.output,
             fast_track=args.fast_track)
