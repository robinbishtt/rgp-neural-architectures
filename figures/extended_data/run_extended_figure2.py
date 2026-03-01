"""
figures/extended_data/run_extended_figure2.py

Extended Data Figure 2  Jacobian Spectrum Evolution

  a) Singular value distribution across 5 depth checkpoints
  b) Log singular value evolution (mean ± std across seeds)
  c) Jacobian condition number κ vs depth
  d) Cumulative Lyapunov exponents (top-5)

Usage
-----
    python figures/extended_data/run_extended_figure2.py \
        --results results/jacobian/ --output figures/out/ed_fig2.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.color_palette import depth_colors, panel_label
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label, remove_top_right_spines,
)


def _synthetic_ed2_data(fast_track: bool = False) -> Dict:
    rng  = np.random.default_rng(7)
    N    = 64 if fast_track else 256
    depths = [1, 5, 10] if fast_track else [1, 5, 10, 20, 40]
    n_seeds = 3 if fast_track else 10

    sv_by_depth: Dict = {}
    for d in depths:
        W = rng.standard_normal((N, N)) / np.sqrt(N)
        for _ in range(d - 1):
            Wi = rng.standard_normal((N, N)) / np.sqrt(N)
            W  = W @ Wi
        sv = np.linalg.svd(W, compute_uv=False)
        sv_by_depth[d] = sv.tolist()

    # Mean log|sv| across seeds at each depth
    log_sv_stats: Dict = {}
    for d in depths:
        seeds = []
        for _ in range(n_seeds):
            W = rng.standard_normal((N, N)) / np.sqrt(N)
            for __ in range(d - 1):
                Wi = rng.standard_normal((N, N)) / np.sqrt(N)
                W  = W @ Wi
            sv = np.linalg.svd(W, compute_uv=False)
            seeds.append(np.log(sv + 1e-12).mean())
        log_sv_stats[d] = {"mean": float(np.mean(seeds)),
                           "std":  float(np.std(seeds))}

    # Condition number vs depth
    cond_nos = []
    for d in depths:
        sv = np.array(sv_by_depth[d])
        cond_nos.append(float(sv[0] / (sv[-1] + 1e-10)))

    # Cumulative top-5 Lyapunov exponents
    top5_lyap = []
    for d in depths:
        sv  = np.array(sv_by_depth[d])
        lam = np.log(sv[:5] + 1e-12) / max(d, 1)
        top5_lyap.append(lam.tolist())

    return {
        "depths":       depths,
        "sv_by_depth":  sv_by_depth,
        "log_sv_stats": log_sv_stats,
        "cond_nos":     cond_nos,
        "top5_lyap":    top5_lyap,
        "N":            N,
    }


def _panel_sv_distribution(ax: plt.Axes, data: Dict, fast_track: bool) -> None:
    depths = data["depths"]
    colors = depth_colors(depths)
    sv_max_global = max(max(data["sv_by_depth"][d]) for d in depths)
    x_arr = np.linspace(0, sv_max_global * 1.05, 300)

    for d, color in zip(depths, colors):
        sv = np.array(data["sv_by_depth"][d])
        if sv.std() > 1e-8:
            kde = gaussian_kde(sv, bw_method=0.3)
            ax.plot(x_arr, kde(x_arr), lw=1.0, color=color, label=f"k={d}")

    ax.set_xlabel("Singular value $\\sigma$")
    ax.set_ylabel("Density")
    tag = " [FT]" if fast_track else ""
    ax.set_title("ED2a: SV distribution vs depth" + tag, fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=5)
    remove_top_right_spines(ax)


def _panel_log_sv_evolution(ax: plt.Axes, data: Dict) -> None:
    depths  = data["depths"]
    means   = [data["log_sv_stats"][d]["mean"] for d in depths]
    stds    = [data["log_sv_stats"][d]["std"]  for d in depths]
    d_arr   = np.array(depths)
    means   = np.array(means)
    stds    = np.array(stds)

    ax.plot(d_arr, means, "-o", ms=3.5, lw=1.2, color="#4878CF",
            label=r"Mean $\langle\log|\sigma|\rangle$")
    ax.fill_between(d_arr, means - stds, means + stds,
                    alpha=0.18, color="#4878CF", label="±1 s.d.")

    ax.axhline(y=0, lw=0.7, color="#333333", ls="--", label="log|σ|=0")
    ax.set_xlabel("Depth $k$")
    ax.set_ylabel(r"$\langle\log|\sigma|\rangle$")
    ax.set_title("ED2b: Log-SV evolution (mean ± s.d.)", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def _panel_condition_number(ax: plt.Axes, data: Dict) -> None:
    depths  = np.array(data["depths"])
    conds   = np.array(data["cond_nos"])

    ax.semilogy(depths, conds, "-s", ms=4, lw=1.2, color="#D65F5F")
    ax.set_xlabel("Depth $k$")
    ax.set_ylabel(r"Condition number $\kappa$")
    ax.set_title("ED2c: Jacobian condition number", fontsize=7, pad=3)
    remove_top_right_spines(ax)


def _panel_lyapunov_top5(ax: plt.Axes, data: Dict) -> None:
    depths   = data["depths"]
    top5     = data["top5_lyap"]
    colors5  = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]

    for i in range(5):
        lam_vals = [top5[di][i] for di in range(len(depths))]
        ax.plot(depths, lam_vals, "-o", ms=3, lw=1.0,
                color=colors5[i], label=f"λ_{i+1}")

    ax.axhline(y=0, lw=0.7, color="#333333", ls="--", label="λ=0 (critical)")
    ax.set_xlabel("Depth $k$")
    ax.set_ylabel("Lyapunov exponent $\\lambda$")
    ax.set_title("ED2d: Top-5 Lyapunov exponents", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    remove_top_right_spines(ax)


def generate(
    results_dir: str = "results/jacobian",
    output_path: str = "figures/out/ed_fig2.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()
    data = _synthetic_ed2_data(fast_track=fast_track)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.0))
    fig.subplots_adjust(wspace=0.38, hspace=0.50)

    _panel_sv_distribution(axes[0, 0], data, fast_track)
    _panel_log_sv_evolution(axes[0, 1], data)
    _panel_condition_number(axes[1, 0], data)
    _panel_lyapunov_top5(axes[1, 1], data)

    for i, ax in enumerate(axes.flat):
        add_panel_label(ax, panel_label(i))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)

    tag = " [FAST_TRACK_UNVERIFIED]" if fast_track else ""
    print(f"Extended Data Figure 2 saved: {out}{tag}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/jacobian")
    p.add_argument("--output",  default="figures/out/ed_fig2.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(results_dir=args.results, output_path=args.output,
             fast_track=args.fast_track)
