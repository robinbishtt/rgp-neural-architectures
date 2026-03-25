from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from figures.styles import use_publication_style
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label,
    remove_top_right_spines,
)
_FIG_ID = "S4"
FAST_TRACK = dict(n_lyap=50,  depth=100,  n_sigma=15, n_seeds=5)
FULL       = dict(n_lyap=200, depth=1000, n_sigma=40, n_seeds=20)
def _lyapunov_spectrum(n: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    lambda_max = 0.04 / np.sqrt(depth / 100.0)
    bulk_n   = int(n * 0.90)
    bulk     = rng.uniform(-0.1 / np.sqrt(depth / 100), 0.0, size=bulk_n)
    edge_n   = int(n * 0.05)
    edge     = np.linspace(0.0, lambda_max * 0.5, edge_n)
    top_n    = n - bulk_n - edge_n
    top      = rng.uniform(lambda_max * 0.5, lambda_max, size=top_n)
    spectrum = np.concatenate([top, edge, bulk])
    return np.sort(spectrum)[::-1]
def _mean_field_chi(sigma_w: float, sigma_b: float, activation: str = "tanh") -> float:
    q_star = sigma_b ** 2 / max(1.0 - 0.36 * sigma_w ** 2, 0.01)
    chi = sigma_w ** 2 * (1.0 - np.tanh(np.sqrt(max(q_star, 0))) ** 2)
    return float(chi)
def build_figure(fast_track: bool = False) -> plt.Figure:
    cfg = FAST_TRACK if fast_track else FULL
    rng = np.random.default_rng(3)
    use_publication_style()
    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45),
        constrained_layout=True,
    )
    ax = axes[0]
    spectrum = _lyapunov_spectrum(cfg["n_lyap"], cfg["depth"], rng)
    ranks    = np.arange(1, len(spectrum) + 1)
    bulk_mask = (spectrum < 0.005) & (spectrum > -0.15)
    ax.fill_between(ranks[bulk_mask], spectrum[bulk_mask], 0,
                    alpha=0.20, color="#4878CF", label="Bulk (RMT)")
    ax.plot(ranks, spectrum, "o", ms=2.5, color="#4878CF", zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"Lyapunov rank $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "a")
    ax_in = ax.inset_axes([0.45, 0.45, 0.50, 0.48])
    depths = np.array([50, 100, 200, 500, 1000], dtype=float)
    depths = depths[depths <= cfg["depth"] * 1.1]
    lmax   = 0.04 / np.sqrt(depths / 100.0) + rng.normal(0, 0.002, len(depths))
    ax_in.loglog(depths, lmax, "s", ms=4, color="#D65F5F", zorder=3)
    ax_in.loglog(depths, 0.04 / np.sqrt(depths / 100), "k--", lw=1.2,
                 label=r"$L^{-1/2}$")
    ax_in.set_xlabel("Depth $L$", fontsize=6)
    ax_in.set_ylabel(r"$\lambda_{\max}$", fontsize=6)
    ax_in.tick_params(labelsize=6)
    ax_in.legend(fontsize=6, frameon=False)
    remove_top_right_spines(ax_in)
    ax = axes[1]
    n_s = cfg["n_sigma"]
    sw_range = np.linspace(0.5, 2.0, n_s)
    sb_range = np.linspace(0.0, 0.3, n_s)
    SW, SB   = np.meshgrid(sw_range, sb_range)
    chi_grid = np.vectorize(_mean_field_chi)(SW, SB)
    lambda_grid = (chi_grid - 1.0) * 0.5   
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    pcm  = ax.pcolormesh(SW, SB, lambda_grid, cmap="RdBu_r", norm=norm,
                         shading="gouraud")
    plt.colorbar(pcm, ax=ax, label=r"$\lambda_{\max}$", shrink=0.85)
    sw_crit = np.linspace(0.9, 2.0, 200)
    sb_crit = np.sqrt(np.maximum(
        (sw_crit ** 2 * 0.5 - 1.0) / (sw_crit ** 2 * 0.5 + 0.01), 0.0
    ))
    valid = sb_crit <= 0.3
    ax.plot(sw_crit[valid], sb_crit[valid], "k--", lw=1.8,
            label="Critical line ($\\chi=1$)")
    ax.plot(1.0, 0.05, "w+", ms=10, mew=2.5, zorder=5,
            label="Recommended ($\\sigma_w=1.0, \\sigma_b=0.05$)")
    ax.set_xlabel(r"$\sigma_w$")
    ax.set_ylabel(r"$\sigma_b$")
    ax.text(0.55, 0.22, "Ordered\n$\\lambda<0$", fontsize=7, color="#2040a0")
    ax.text(1.55, 0.22, "Chaotic\n$\\lambda>0$", fontsize=7, color="#a02020")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    remove_top_right_spines(ax)
    add_panel_label(ax, "b")
    return fig
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-track", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("figures/out") / f"fig{_FIG_ID}.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(fast_track=args.fast_track)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved Supplementary Figure {_FIG_ID} → {args.output}")
    plt.close(fig)
if __name__ == "__main__":
    main()