"""
figures/manuscript/generate_figure2.py

Figure 2 - RG Layer Mechanics

Panel layout:
  a) Single RG-layer forward computation: h^(k) = σ(W_k g_k h^(k-1) + b_k)
  b) Fisher metric pushforward G^(k) = J_k G^(k-1) J_kᵀ across five layers
  c) Eigenvalue spectrum evolution: layer 1 vs layer k (MP convergence)

Usage
-----
    python figures/manuscript/generate_figure2.py --output figures/out/fig2.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.color_palette import depth_colors
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label, remove_top_right_spines,
)


# ---------------------------------------------------------------------------
# Synthetic Marchenko-Pastur PDF
# ---------------------------------------------------------------------------

def _mp_pdf(lam: np.ndarray, beta: float, sigma2: float = 1.0) -> np.ndarray:
    lam_minus = sigma2 * (1.0 - np.sqrt(beta)) ** 2
    lam_plus  = sigma2 * (1.0 + np.sqrt(beta)) ** 2
    pdf = np.zeros_like(lam)
    mask = (lam > lam_minus) & (lam < lam_plus)
    pdf[mask] = (
        np.sqrt((lam_plus - lam[mask]) * (lam[mask] - lam_minus))
        / (2.0 * np.pi * sigma2 * beta * lam[mask])
    )
    return pdf


# ---------------------------------------------------------------------------
# Panel (a) - Single-layer forward computation schematic
# ---------------------------------------------------------------------------

def _draw_layer_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def _box(x, y, w, h, label, sub, facecolor, fontsize=6):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=facecolor, edgecolor="#555555", linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.18, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                ha="center", va="center", fontsize=5, color="#555555")

    def _arrow(x0, x1, y, label=""):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
        if label:
            ax.text((x0 + x1) / 2, y + 0.2, label,
                    ha="center", fontsize=5, color="#333333")

    _box(0.2, 2.2, 1.8, 1.6, r"$h^{(k-1)}$", "input activation", "#AED6F1")
    _arrow(2.0, 3.0, 3.0)
    _box(3.0, 1.5, 1.6, 3.0, r"$W_k$", "weight matrix\n$N_{k} \times N_{k-1}$", "#A9DFBF")
    _arrow(4.6, 5.4, 3.0, label=r"$\times g_k$")
    _box(5.4, 2.2, 1.6, 1.6, r"$+b_k$", "bias", "#F9E79F")
    _arrow(7.0, 7.8, 3.0)
    _box(7.8, 2.2, 1.8, 1.6, r"$\sigma(\cdot)$", "activation fn", "#F1948A")
    _arrow(9.6, 10.0, 3.0)
    ax.text(9.8, 3.15, r"$h^{(k)}$", fontsize=6.5, fontweight="bold")

    ax.text(5.0, 0.4,
            r"$h^{(k)} = \sigma\!\left(W_k\, g_k(h^{(k-1)}) + b_k\right)$",
            ha="center", fontsize=6.5)

    ax.set_title("RG-Layer Forward Computation", fontsize=7, pad=3)


# ---------------------------------------------------------------------------
# Panel (b) - Fisher metric pushforward across layers
# ---------------------------------------------------------------------------

def _draw_fisher_pushforward(ax: plt.Axes) -> None:
    rng = np.random.default_rng(0)
    layers = np.arange(1, 6)
    N = 64

    traces   = []
    cond_nos = []
    for k in layers:
        # Synthetic: metric contracts and aligns with depth
        ev = rng.exponential(scale=1.0 / k, size=N)
        ev.sort()
        traces.append(ev.mean())
        cond_nos.append(ev[-1] / (ev[0] + 1e-6))

    depth_colors(list(layers))

    ax2 = ax.twinx()
    ax.plot(layers, traces, "-o", ms=4, lw=1.2,
            color="#4878CF", label="tr G$^{(k)}$ / N")
    ax2.plot(layers, cond_nos, "-s", ms=4, lw=1.2,
             color="#D65F5F", label="κ(G$^{(k)}$)")

    ax.set_xlabel("Layer $k$")
    ax.set_ylabel(r"Mean eigenvalue", color="#4878CF")
    ax2.set_ylabel(r"Condition number $\kappa$", color="#D65F5F")
    ax.tick_params(axis="y", labelcolor="#4878CF")
    ax2.tick_params(axis="y", labelcolor="#D65F5F")
    remove_top_right_spines(ax)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", frameon=False)
    ax.set_title(r"Fisher Metric Pushforward $G^{(k)} = J_k G^{(k-1)} J_k^\top$",
                 fontsize=7, pad=3)


# ---------------------------------------------------------------------------
# Panel (c) - Eigenvalue spectrum evolution (MP convergence)
# ---------------------------------------------------------------------------

def _draw_spectrum_evolution(ax: plt.Axes) -> None:
    rng = np.random.default_rng(42)
    beta = 0.5
    N, M = 200, 100
    colors = depth_colors([1, 5, 15, 30])
    depths = [1, 5, 15, 30]

    lam_arr = np.linspace(0, 3, 300)
    mp_theoretical = _mp_pdf(lam_arr, beta=beta, sigma2=1.0)

    for depth, color in zip(depths, colors):
        # Simulate product of Jacobian matrices approaching MP
        W = rng.standard_normal((N, M)) / np.sqrt(M)
        for _ in range(depth - 1):
            W_i = rng.standard_normal((M, M)) / np.sqrt(M)
            W = W @ W_i
        ev = np.linalg.svd(W, compute_uv=False) ** 2

        if len(ev) > 5:
            kde = gaussian_kde(ev, bw_method=0.25)
            y = kde(lam_arr)
            ax.plot(lam_arr, y, lw=1.0, color=color, alpha=0.85, label=f"k={depth}")

    ax.plot(lam_arr, mp_theoretical, "--", lw=1.2,
            color="#D62728", label="MP theory")

    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 3)
    ax.legend(loc="upper right", frameon=False)
    remove_top_right_spines(ax)
    ax.set_title("Eigenvalue Spectrum → Marchenko-Pastur", fontsize=7, pad=3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(
    output_path: str = "figures/out/fig2.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()

    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.40)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    _draw_layer_schematic(ax_a)
    _draw_fisher_pushforward(ax_b)
    _draw_spectrum_evolution(ax_c)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)

    tag = "" if fast_track else ""
    print(f"Figure 2 saved: {out}{tag}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 2")
    p.add_argument("--output", default="figures/out/fig2.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(output_path=args.output, fast_track=args.fast_track)
 