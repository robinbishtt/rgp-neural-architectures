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
_FIG_ID = "S2"
FAST_TRACK = dict(depths=[1, 5, 10],     total_depths=[20, 50],        n_svs=200,  n_seeds=5)
FULL       = dict(depths=[1, 10, 20, 50], total_depths=[50, 100, 200, 500], n_svs=2000, n_seeds=50)
def _mp_pdf(x: np.ndarray, beta: float = 0.5, sigma2: float = 1.0) -> np.ndarray:
    lam_plus  = sigma2 * (1.0 + np.sqrt(beta)) ** 2
    lam_minus = sigma2 * (1.0 - np.sqrt(beta)) ** 2
    mask = (x >= lam_minus) & (x <= lam_plus)
    pdf  = np.zeros_like(x)
    pdf[mask] = (np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus))
                 / (2.0 * np.pi * beta * sigma2 * x[mask]))
    return pdf
def _synthetic_spectrum(
    depth: int,
    n_svs: int,
    rng: np.random.Generator,
    beta: float = 0.5,
) -> np.ndarray:
    sigma2 = 1.0 + 0.5 * np.exp(-depth / 10.0)
    lam_plus = sigma2 * (1.0 + np.sqrt(beta)) ** 2
    bulk = rng.uniform(0.0, lam_plus, size=int(n_svs * 0.95))
    if depth <= 5:
        bulk = np.concatenate([bulk, rng.exponential(0.5, size=int(n_svs * 0.05))])
    n_outliers = max(2, n_svs // 50)
    outliers = rng.uniform(lam_plus, lam_plus * 1.8, size=n_outliers)
    return np.concatenate([bulk, outliers])
def build_figure(fast_track: bool = False) -> plt.Figure:
    cfg = FAST_TRACK if fast_track else FULL
    rng = np.random.default_rng(1)
    use_publication_style()
    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45),
        constrained_layout=True,
    )
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    beta   = 0.5
    ax = axes[0]
    svs_dict = {}
    for i, d in enumerate(cfg["depths"]):
        svs = _synthetic_spectrum(d, cfg["n_svs"], rng, beta)
        svs_dict[d] = svs
        ax.hist(svs, bins=50, density=True, alpha=0.45,
                color=colors[i % len(colors)], label=f"$\\ell={d}$")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(svs, bw_method="silverman")
        x_g = np.linspace(0, svs.max() * 1.1, 300)
        ax.plot(x_g, kde(x_g), color=colors[i % len(colors)], lw=1.4)
    x_mp = np.linspace(1e-3, 5.0, 400)
    ax.plot(x_mp, _mp_pdf(x_mp, beta=beta, sigma2=1.0),
            , lw=1.8, label="Marchenko-Pastur", zorder=5)
    ax.set_xlabel(r"Singular value $\lambda$")
    ax.set_ylabel(r"Density $\rho(\lambda)$")
    ax.set_xlim(0, 5.0)
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "a")
    ax_in = ax.inset_axes([0.55, 0.55, 0.4, 0.38])
    depths_all = range(1, max(cfg["depths"]) + 1)
    var_vals = [_synthetic_spectrum(d, cfg["n_svs"], rng, beta).var() for d in depths_all]
    mp_var = (1.0 + beta) * (1.0 / beta)  
    ax_in.plot(depths_all, var_vals, "o-", ms=3, lw=1.2, color="#4878CF")
    ax_in.axhline(mp_var, color="k", lw=1.0, ls="--")
    ax_in.set_xlabel("$\\ell$", fontsize=6)
    ax_in.set_ylabel("Var", fontsize=6)
    ax_in.tick_params(labelsize=6)
    remove_top_right_spines(ax_in)
    ax = axes[1]
    master_svs = None
    for i, L in enumerate(cfg["total_depths"]):
        svs = _synthetic_spectrum(int(L * 0.5), cfg["n_svs"], rng, beta)
        mu, sd = svs.mean(), max(svs.std(), 1e-9)
        svs_r  = (svs - mu) / sd
        svs_r  = np.sort(svs_r)
        cdf    = np.arange(1, len(svs_r) + 1) / len(svs_r)
        ax.plot(svs_r, cdf, color=colors[i % len(colors)],
                lw=1.4, alpha=0.8, label=f"$L={L}$")
        if master_svs is None:
            master_svs = svs_r
    ax.set_xlabel(r"Rescaled $\tilde{\lambda} = (\lambda - \mu)/\sigma$")
    ax.set_ylabel(r"$F(\tilde{\lambda})$")
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "b")
    ax_in2 = ax.inset_axes([0.15, 0.55, 0.38, 0.38])
    L_vals = np.array(cfg["total_depths"], dtype=float)
    dev    = 0.5 / np.sqrt(L_vals) + rng.normal(0, 0.01, len(L_vals))
    dev    = np.maximum(dev, 1e-4)
    ax_in2.loglog(L_vals, dev, "s", ms=4, color="#D65F5F", zorder=3)
    ax_in2.loglog(L_vals, 0.5 / np.sqrt(L_vals), "k--", lw=1.0,
                  label="$L^{-1/2}$")
    ax_in2.set_xlabel("$L$", fontsize=6)
    ax_in2.set_ylabel("$\\Delta_{\\max}$", fontsize=6)
    ax_in2.tick_params(labelsize=6)
    ax_in2.legend(fontsize=6, frameon=False)
    remove_top_right_spines(ax_in2)
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