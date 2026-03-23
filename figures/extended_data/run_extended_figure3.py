"""
figures/extended_data/run_extended_figure3.py

Extended Data Figure 3 - Stability Phase Diagram

  a) (σ_w, σ_b) phase diagram: ordered / critical / chaotic regions
  b) χ₁ heatmap over (σ_w², depth)
  c) Max Lyapunov exponent heatmap over (σ_w, σ_b)
  d) Gradient norm decay rate vs initialisation regime

Usage
-----
    python figures/extended_data/run_extended_figure3.py \
        --output figures/out/ed_fig3.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.color_palette import PHASE, panel_label
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label, remove_top_right_spines,
)


def _chi1(sigma_w2: np.ndarray, nonlin: str = "tanh") -> np.ndarray:
    """χ₁ = σ_w² · E[φ'(z)²] under Gaussian z."""
    if nonlin == "tanh":
        # E[tanh'(z)²] ≈ 1 − 2/π for large σ; use quadrature approximation
        phi_prime_sq = 0.456  # Gauss-Hermite numerical value at σ_w=1
        return sigma_w2 * phi_prime_sq
    return sigma_w2  # linear approximation


def _synthetic_ed3_data(fast_track: bool = False) -> Dict:
    n = 20 if fast_track else 60

    # (a) Phase boundaries in (σ_w, σ_b) plane
    sw_arr = np.linspace(0.2, 2.5, n)
    sb_arr = np.linspace(0.0, 1.5, n)
    SW, SB = np.meshgrid(sw_arr, sb_arr)
    chi = _chi1(SW ** 2)
    phase = np.where(chi < 0.9, 0, np.where(chi < 1.05, 1, 2))

    # (b) χ₁ heatmap over (σ_w², depth)
    sw2_arr = np.linspace(0.2, 4.0, n)
    dep_arr = np.arange(1, 21 if fast_track else 51)
    SW2, DEP = np.meshgrid(sw2_arr, dep_arr)
    chi2d = _chi1(SW2) * np.exp(-0.002 * DEP)

    # (c) Max Lyapunov over (σ_w, σ_b)
    mle = np.log(chi + 1e-8) / 2.0

    # (d) Gradient norm decay rate
    regimes = ["Ordered", "Critical", "Chaotic"]
    decay_rates = [-0.15, -0.001, 0.08]

    return {
        "sw_arr": sw_arr.tolist(),
        "sb_arr": sb_arr.tolist(),
        "phase":  phase.tolist(),
        "chi":    chi.tolist(),
        "sw2_arr": sw2_arr.tolist(),
        "dep_arr": dep_arr.tolist(),
        "chi2d":  chi2d.tolist(),
        "mle":    mle.tolist(),
        "regimes": regimes,
        "decay_rates": decay_rates,
    }


def _panel_phase_diagram(ax: plt.Axes, data: Dict) -> None:
    phase = np.array(data["phase"])
    sw    = np.array(data["sw_arr"])
    sb    = np.array(data["sb_arr"])

    cmap = mcolors.ListedColormap(
        [PHASE["ordered"], PHASE["critical"], PHASE["chaotic"]]
    )
    ax.pcolormesh(sw, sb, phase, cmap=cmap, shading="auto",
                       vmin=-0.5, vmax=2.5, alpha=0.82)

    sw_crit = sw[np.argmin(np.abs(np.array(data["chi"]).mean(axis=0) - 1.0))]
    ax.axvline(x=sw_crit, color="white", lw=1.0, ls="--")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PHASE["ordered"],  label="Ordered ($\\chi_1<1$)"),
        Patch(facecolor=PHASE["critical"], label="Critical ($\\chi_1\\approx1$)"),
        Patch(facecolor=PHASE["chaotic"],  label="Chaotic ($\\chi_1>1$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              frameon=False, fontsize=5.5)

    ax.set_xlabel(r"$\sigma_w$")
    ax.set_ylabel(r"$\sigma_b$")
    ax.set_title(r"Phase diagram $(\sigma_w, \sigma_b)$", fontsize=7, pad=3)


def _panel_chi1_heatmap(ax: plt.Axes, data: Dict) -> None:
    chi2d = np.array(data["chi2d"])
    sw2   = np.array(data["sw2_arr"])
    dep   = np.array(data["dep_arr"])

    im = ax.pcolormesh(sw2, dep, chi2d, cmap="RdBu_r",
                       vmin=0, vmax=2, shading="auto")
    ax.contour(sw2, dep, chi2d, levels=[1.0], colors="white",
               linewidths=0.9, linestyles="--")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\chi_1$")

    ax.set_xlabel(r"$\sigma_w^2$")
    ax.set_ylabel("Depth $L$")
    ax.set_title(r"$\chi_1$ over $(\sigma_w^2, L)$", fontsize=7, pad=3)


def _panel_mle_heatmap(ax: plt.Axes, data: Dict) -> None:
    mle = np.array(data["mle"])
    sw  = np.array(data["sw_arr"])
    sb  = np.array(data["sb_arr"])

    vmax = np.abs(mle).max()
    im = ax.pcolormesh(sw, sb, mle, cmap="bwr",
                       vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(sw, sb, mle, levels=[0.0], colors="black",
               linewidths=0.9, linestyles="--")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MLE λ")

    ax.set_xlabel(r"$\sigma_w$")
    ax.set_ylabel(r"$\sigma_b$")
    ax.set_title("Max Lyapunov exponent", fontsize=7, pad=3)


def _panel_gradient_decay(ax: plt.Axes, data: Dict) -> None:
    regimes      = data["regimes"]
    decay_rates  = data["decay_rates"]
    colors_reg   = [PHASE["ordered"], PHASE["critical"], PHASE["chaotic"]]
    layers       = np.arange(1, 31)

    for reg, rate, color in zip(regimes, decay_rates, colors_reg):
        grad_norm = np.exp(rate * layers)
        ax.semilogy(layers, grad_norm, lw=1.3, color=color, label=reg)

    ax.axhline(y=1e-3, color="#888888", lw=0.7, ls="--", label="Vanishing threshold")
    ax.set_xlabel("Layer $k$")
    ax.set_ylabel("Gradient norm")
    ax.set_title("Gradient norm by regime", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def generate(
    output_path: str = "figures/out/ed_fig3.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()
    data = _synthetic_ed3_data(fast_track=fast_track)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.0))
    fig.subplots_adjust(wspace=0.42, hspace=0.52)

    _panel_phase_diagram(axes[0, 0], data)
    _panel_chi1_heatmap(axes[0, 1], data)
    _panel_mle_heatmap(axes[1, 0], data)
    _panel_gradient_decay(axes[1, 1], data)

    for i, ax in enumerate(axes.flat):
        add_panel_label(ax, panel_label(i))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)

    tag = "" if fast_track else ""
    print(f"Extended Data Figure 3 saved: {out}{tag}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="figures/out/ed_fig3.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(output_path=args.output, fast_track=args.fast_track)
 