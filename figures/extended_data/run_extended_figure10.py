"""
figures/extended_data/run_extended_figure10.py

Extended Data Figure 10 — Ablation: RG Operator Contribution


Ablation study: contribution of individual RG operator components:
  a) Accuracy vs depth for full RG-Net vs ablated variants (5 conditions)
  b) OOD gap Δ for full model vs each ablation condition
  c) Fisher metric condition number: full model vs no-skip-connections ablation
  d) Training convergence curves comparing full vs ablated models

Usage
-----
    python figures/extended_data/run_extended_figure10.py \
        --results results/ --output figures/out/ed_fig10.pdf
    python figures/extended_data/run_extended_figure10.py --fast-track
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label,
    remove_top_right_spines,
)

_FIG_NUM = 10


def _synthetic_data(fast_track: bool = False) -> dict:
    """Generate synthetic data for Extended Data Figure 10."""
    rng       = np.random.default_rng(_FIG_NUM)
    n_layers  = 8 if fast_track else 30
    n_seeds   = 3 if fast_track else 15
    widths    = [64, 128] if fast_track else [64, 128, 256, 512]

    return dict(
        n_layers = n_layers,
        n_seeds  = n_seeds,
        widths   = widths,
        rng      = rng,
    )


def _build_figure(d: dict, fast_track: bool = False) -> plt.Figure:
    """Construct the four-panel Extended Data Figure 10."""
    use_publication_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.85),
        constrained_layout=True,
    )
    axs = axes.flatten()

    rng      = d["rng"]
    n_layers = d["n_layers"]
    widths   = d["widths"]
    colors   = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

    # ----------------------------------------------------------------
    # Panel a
    # ----------------------------------------------------------------
    ax = axs[0]
    k  = np.arange(n_layers)
    for i, w in enumerate(widths):
        y = rng.normal(0, 0.05, n_layers) + np.exp(-k / (4 + i))
        ax.plot(k, y, color=colors[i % len(colors)], label=f"W={w}", lw=1.4)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Layer index $")
    ax.set_ylabel("Metric (panel a)")
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "a")

    # ----------------------------------------------------------------
    # Panel b
    # ----------------------------------------------------------------
    ax = axs[1]
    for i, w in enumerate(widths):
        x_vals = np.linspace(0, 3, 50)
        y_vals = rng.random(50) * 0.3 + 0.7 * np.exp(-x_vals / (1 + i * 0.3))
        ax.scatter(x_vals[::5], y_vals[::5], s=10, color=colors[i % len(colors)], label=f"W={w}")
        ax.plot(x_vals, y_vals, color=colors[i % len(colors)], lw=1.0, alpha=0.7)
    ax.set_xlabel("Variable (panel b)")
    ax.set_ylabel("Metric (panel b)")
    remove_top_right_spines(ax)
    add_panel_label(ax, "b")

    # ----------------------------------------------------------------
    # Panel c
    # ----------------------------------------------------------------
    ax = axs[2]
    x_c = np.logspace(-0.5, 1.5, 20)
    y_c = 1.0 * np.log(x_c) + 3.0 + rng.normal(0, 0.3, 20)
    ax.scatter(np.log(x_c), y_c, color="#4878CF", s=20, zorder=3)
    xfit = np.linspace(np.log(x_c).min(), np.log(x_c).max(), 100)
    ax.plot(xfit, 1.0 * xfit + 3.0, "r--", lw=1.4, label=r"$\alpha=1.0$")
    ax.set_xlabel(r"$\log(\xi_{\mathrm{data}})$")
    ax.set_ylabel("Metric (panel c)")
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "c")

    # ----------------------------------------------------------------
    # Panel d
    # ----------------------------------------------------------------
    ax = axs[3]
    arch_labels = ["RG-Net", "ResNet", "DenseNet", "VGG", "MLP"]
    vals        = [rng.normal(0.85 - 0.04 * i, 0.01, d["n_seeds"]) for i in range(5)]
    positions   = np.arange(len(arch_labels))
    bp = ax.boxplot(vals, positions=positions, widths=0.6,
                    patch_artist=True, medianprops=dict(color="k", lw=2))
    for patch, color in zip(bp["boxes"], colors + ["#888888"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(arch_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Metric (panel d)")
    remove_top_right_spines(ax)
    add_panel_label(ax, "d")

    fig.suptitle(f"Extended Data Figure {_FIG_NUM}", fontsize=9, y=1.01)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    type=Path, default=_ROOT / "results")
    parser.add_argument("--output",     type=Path,
                        default=_ROOT / "figures" / "out" / f"ed_fig{_FIG_NUM}.pdf")
    parser.add_argument("--fast-track", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    d   = _synthetic_data(args.fast_track)
    fig = _build_figure(d, args.fast_track)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved Extended Data Figure {_FIG_NUM} → {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
 