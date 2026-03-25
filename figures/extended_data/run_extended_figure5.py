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
def _synthetic_ed5_data(fast_track: bool = False) -> Dict:
    rng = np.random.default_rng(5)
    N   = 32 if fast_track else 128
    n_iter = 20 if fast_track else 200
    spectra: Dict = {}
    for regime, mu in zip(_REGIMES, [-0.3, 0.0, 0.3]):
        lam = np.sort(mu + rng.standard_normal(N) * 0.15)[::-1]
        spectra[regime] = lam.tolist()
    depths = np.arange(1, 11 if fast_track else 51)
    ly_sums: Dict = {}
    for regime, mu in zip(_REGIMES, [-0.3, 0.0, 0.3]):
        s = mu * depths + rng.normal(0, 0.02, len(depths))
        ly_sums[regime] = s.tolist()
    qr_conv: Dict = {}
    for regime, mu in zip(_REGIMES, [-0.3, 0.0, 0.3]):
        iters = np.arange(1, n_iter + 1)
        err   = np.abs(mu) * np.exp(-0.08 * iters) + 1e-6 * rng.random(n_iter)
        qr_conv[regime] = err.tolist()
    sw_vals  = np.linspace(0.5, 2.0, 20 if fast_track else 60)
    ky_dims  = N * (1.0 - np.exp(-0.5 * (sw_vals - 1.0) ** 2))
    return {
        :        N,
        :  spectra,
        :   depths.tolist(),
        :  ly_sums,
        :   n_iter,
        :  qr_conv,
        :  sw_vals.tolist(),
        :  ky_dims.tolist(),
    }
def _panel_full_spectrum(ax: plt.Axes, data: Dict) -> None:
    N = data["N"]
    idx = np.arange(1, N + 1)
    for regime, color in zip(_REGIMES, _REGIME_COLORS):
        lam = np.array(data["spectra"][regime])
        ax.plot(idx, lam, lw=1.0, color=color, label=regime)
    ax.axhline(y=0, lw=0.7, color="#333333", ls="--")
    ax.set_xlabel("Exponent index $i$")
    ax.set_ylabel("Lyapunov exponent $\\lambda_i$")
    ax.set_title("Full Lyapunov spectrum", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)
def _panel_lyapunov_sum(ax: plt.Axes, data: Dict) -> None:
    depths = np.array(data["depths"])
    for regime, color in zip(_REGIMES, _REGIME_COLORS):
        s = np.array(data["ly_sums"][regime])
        ax.plot(depths, s, lw=1.2, color=color, label=regime)
    ax.axhline(y=0, lw=0.7, color="#333333", ls="--", label="∑λ=0")
    ax.set_xlabel("Depth $k$")
    ax.set_ylabel(r"$\sum_i \lambda_i$ (entropy production)")
    ax.set_title("Lyapunov sum vs depth", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)
def _panel_qr_convergence(ax: plt.Axes, data: Dict) -> None:
    n_iter = data["n_iter"]
    iters  = np.arange(1, n_iter + 1)
    for regime, color in zip(_REGIMES, _REGIME_COLORS):
        err = np.array(data["qr_conv"][regime])
        ax.semilogy(iters, err + 1e-9, lw=1.0, color=color, label=regime)
    ax.set_xlabel("QR iteration")
    ax.set_ylabel("Exponent error")
    ax.set_title("QR algorithm convergence", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)
def _panel_kaplan_yorke(ax: plt.Axes, data: Dict) -> None:
    sw   = np.array(data["sw_vals"])
    ky   = np.array(data["ky_dims"])
    N    = data["N"]
    ax.plot(sw, ky, lw=1.3, color="#4878CF", label="$D_{KY}$")
    ax.axhline(y=0, lw=0.7, color=PHASE["ordered"], ls="--", label="Ordered")
    ax.axhline(y=N, lw=0.7, color=PHASE["chaotic"],  ls="--", label="Max ($N$)")
    sw_crit = sw[np.argmin(np.abs(ky - N / 2))]
    ax.axvline(x=sw_crit, color=PHASE["critical"], lw=0.8, ls=":",
               label=f"$\\sigma_w^*={sw_crit:.2f}$")
    ax.set_xlabel(r"$\sigma_w$")
    ax.set_ylabel("Kaplan-Yorke dimension")
    ax.set_title("$D_{KY}$ vs $\\sigma_w$", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=6)
    remove_top_right_spines(ax)
def generate(
    output_path: str = "figures/out/ed_fig5.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()
    data = _synthetic_ed5_data(fast_track=fast_track)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.0))
    fig.subplots_adjust(wspace=0.42, hspace=0.52)
    _panel_full_spectrum(axes[0, 0], data)
    _panel_lyapunov_sum(axes[0, 1], data)
    _panel_qr_convergence(axes[1, 0], data)
    _panel_kaplan_yorke(axes[1, 1], data)
    for i, ax in enumerate(axes.flat):
        add_panel_label(ax, panel_label(i))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    tag = "" if fast_track else ""
    print(f"Extended Data Figure 5 saved: {out}{tag}")
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="figures/out/ed_fig5.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()
if __name__ == "__main__":
    args = _parse_args()
    generate(output_path=args.output, fast_track=args.fast_track)