"""
figures/extended_data/run_extended_figure6.py

Extended Data Figure 6  Perturbation Growth Analysis

  a) Input perturbation δh vs layer k (three regimes)
  b) Mean-field sensitivity dh^(k)/dh^(0) norm vs depth
  c) Cross-correlation of perturbed and unperturbed activations
  d) Perturbation spectrum (SVD of sensitivity matrix) at critical init

Usage
-----
    python figures/extended_data/run_extended_figure6.py \
        --output figures/out/ed_fig6.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.color_palette import PHASE, panel_label
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label, remove_top_right_spines,
)

_REGIMES = ["Ordered", "Critical", "Chaotic"]
_REGIME_COLORS = [PHASE["ordered"], PHASE["critical"], PHASE["chaotic"]]
_REGIME_RATES  = [-0.20, 0.0, 0.18]


def _synthetic_ed6_data(fast_track: bool = False) -> Dict:
    rng = np.random.default_rng(12)
    n_layers = 15 if fast_track else 50
    N        = 32  if fast_track else 128
    eps      = 0.01   # perturbation magnitude

    layers = np.arange(n_layers)

    # (a) Perturbation norm vs layer
    delta_by_regime: Dict = {}
    for regime, rate in zip(_REGIMES, _REGIME_RATES):
        delta = eps * np.exp(rate * layers) + 1e-5 * rng.random(n_layers)
        delta_by_regime[regime] = delta.tolist()

    # (b) Sensitivity norm (Frobenius of cumulative Jacobian)
    sens_by_regime: Dict = {}
    for regime, rate in zip(_REGIMES, _REGIME_RATES):
        norm = (N ** 0.5) * np.exp(rate * layers) + 0.01 * rng.random(n_layers)
        sens_by_regime[regime] = norm.tolist()

    # (c) Cross-correlation between perturbed and unperturbed
    xcorr_by_regime: Dict = {}
    for regime, rate in zip(_REGIMES, _REGIME_RATES):
        # correlation decays / grows with perturbation
        corr = np.exp(-0.5 * np.abs(rate) * layers)
        if rate > 0:   # chaotic: faster de-correlation
            corr = corr ** 2
        xcorr_by_regime[regime] = np.clip(corr, 0, 1).tolist()

    # (d) Perturbation spectrum SVD at criticality (rate ≈ 0)
    W_critical = rng.standard_normal((N, N)) / np.sqrt(N)
    for _ in range(5):
        Wi = rng.standard_normal((N, N)) / np.sqrt(N)
        W_critical = W_critical @ Wi
    sv_critical = np.linalg.svd(W_critical, compute_uv=False)

    return {
        "layers":           layers.tolist(),
        "N":                N,
        "eps":              eps,
        "delta_by_regime":  delta_by_regime,
        "sens_by_regime":   sens_by_regime,
        "xcorr_by_regime":  xcorr_by_regime,
        "sv_critical":      sv_critical.tolist(),
    }


def _panel_perturbation_norm(ax: plt.Axes, data: Dict, fast_track: bool) -> None:
    layers = np.array(data["layers"])
    eps    = data["eps"]

    for regime, color in zip(_REGIMES, _REGIME_COLORS):
        delta = np.array(data["delta_by_regime"][regime])
        ax.semilogy(layers, delta / eps, lw=1.2, color=color, label=regime)

    ax.axhline(y=1.0, lw=0.7, color="#333333", ls="--", label="|δh|/ε = 1")
    tag = " [FT]" if fast_track else ""
    ax.set_xlabel("Layer $k$")
    ax.set_ylabel(r"$\|\delta h^{(k)}\| / \varepsilon$")
    ax.set_title("ED6a: Perturbation growth" + tag, fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def _panel_sensitivity_norm(ax: plt.Axes, data: Dict) -> None:
    layers = np.array(data["layers"])

    for regime, color in zip(_REGIMES, _REGIME_COLORS):
        norm = np.array(data["sens_by_regime"][regime])
        ax.semilogy(layers, norm + 1e-8, lw=1.2, color=color, label=regime)

    ax.set_xlabel("Depth $k$")
    ax.set_ylabel(r"$\|\partial h^{(k)}/\partial h^{(0)}\|_F$")
    ax.set_title("ED6b: Cumulative Jacobian norm", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def _panel_xcorr(ax: plt.Axes, data: Dict) -> None:
    layers = np.array(data["layers"])

    for regime, color in zip(_REGIMES, _REGIME_COLORS):
        xcorr = np.array(data["xcorr_by_regime"][regime])
        ax.plot(layers, xcorr, lw=1.2, color=color, label=regime)

    ax.axhline(y=0.5, lw=0.7, color="#888888", ls="--", label="Corr=0.5")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer $k$")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("ED6c: Activation de-correlation", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def _panel_sv_spectrum(ax: plt.Axes, data: Dict) -> None:
    sv  = np.array(data["sv_critical"])
    idx = np.arange(1, len(sv) + 1)

    ax.semilogy(idx, sv, "-", lw=1.2, color=PHASE["critical"],
                label="Critical init")
    ax.semilogy(idx[[0, -1]], [sv[0], sv[-1]], "--", lw=0.8,
                color="#888888", label="Rank-1 envelope")

    ax.set_xlabel("Singular value index $i$")
    ax.set_ylabel("Singular value $\\sigma_i$")
    ax.set_title("ED6d: Sensitivity matrix spectrum (critical)", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)


def generate(
    output_path: str = "figures/out/ed_fig6.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()
    data = _synthetic_ed6_data(fast_track=fast_track)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.0))
    fig.subplots_adjust(wspace=0.42, hspace=0.52)

    _panel_perturbation_norm(axes[0, 0], data, fast_track)
    _panel_sensitivity_norm(axes[0, 1], data)
    _panel_xcorr(axes[1, 0], data)
    _panel_sv_spectrum(axes[1, 1], data)

    for i, ax in enumerate(axes.flat):
        add_panel_label(ax, panel_label(i))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)

    tag = " [FAST_TRACK_UNVERIFIED]" if fast_track else ""
    print(f"Extended Data Figure 6 saved: {out}{tag}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="figures/out/ed_fig6.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(output_path=args.output, fast_track=args.fast_track)
