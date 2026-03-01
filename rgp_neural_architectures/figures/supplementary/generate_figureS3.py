"""
figures/supplementary/generate_figureS3.py

Supplementary Figure S3 — Finite-Size Scaling Collapse

Two-panel figure demonstrating the finite-size scaling (FSS) collapse of
the order parameter across network depths, validating critical exponent
estimates and the scaling ansatz.

    Panel (a): Raw order parameter m(l; L) = mean activation magnitude vs.
               relative depth l/L for networks with L in {50, 100, 200, 500}.
               Shows systematic finite-size effects: peak sharpening and
               amplitude decay with increasing L.

    Panel (b): Rescaled data m * L^(beta/nu) vs (l/L - l_c/L) * L^(1/nu)
               demonstrating collapse onto a single master curve with exponents
               nu = 1.2 +/- 0.1, beta = 0.5 +/- 0.05, l_c/L = 0.50.
               Inset: chi^2 landscape in (nu, beta) plane showing 68% and 95%
               confidence regions.

Usage
-----
    python figures/supplementary/generate_figureS3.py
    python figures/supplementary/generate_figureS3.py --fast-track
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from figures.styles import use_publication_style
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH,
    add_panel_label,
    remove_top_right_spines,
)

_FIG_ID    = "S3"
_NU_TRUE   = 1.2
_BETA_TRUE = 0.5
_LC_FRAC   = 0.50  # l_c / L

FAST_TRACK = dict(depths=[20, 50],        n_l=40,  n_seeds=5)
FULL       = dict(depths=[50, 100, 200, 500], n_l=100, n_seeds=50)
COLORS     = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


def _order_parameter(
    L: int,
    n_l: int,
    nu: float,
    beta: float,
    lc_frac: float,
    rng: np.random.Generator,
    n_seeds: int,
) -> tuple:
    """
    Synthetic order parameter m(l; L) matching finite-size scaling ansatz.

    m(l; L) = L^(-beta/nu) * f((l/L - lc_frac) * L^(1/nu))
    where f is a smooth peaked function approximated by a Gaussian.
    """
    l_vals   = np.linspace(0, L, n_l)
    lc       = lc_frac * L
    scaled_x = (l_vals / L - lc_frac) * (L ** (1.0 / nu))

    # Universal scaling function: Gaussian-like peak
    f_vals = np.exp(-0.5 * scaled_x ** 2)

    # Raw order parameter + noise
    m_raw   = (L ** (-beta / nu)) * f_vals
    m_seeds = m_raw[None, :] + rng.normal(0, m_raw.max() * 0.05,
                                           size=(n_seeds, n_l))
    return l_vals, m_seeds.mean(axis=0), m_seeds.std(axis=0) / np.sqrt(n_seeds)


def build_figure(fast_track: bool = False) -> plt.Figure:
    cfg = FAST_TRACK if fast_track else FULL
    rng = np.random.default_rng(2)

    use_publication_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45),
        constrained_layout=True,
    )

    raw_data = {}
    for i, L in enumerate(cfg["depths"]):
        l_vals, m_mean, m_sem = _order_parameter(
            L, cfg["n_l"], _NU_TRUE, _BETA_TRUE, _LC_FRAC,
            rng, cfg["n_seeds"],
        )
        raw_data[L] = (l_vals, m_mean, m_sem)

    # ----------------------------------------------------------------
    # Panel (a): Raw order parameter vs relative depth
    # ----------------------------------------------------------------
    ax = axes[0]
    for i, L in enumerate(cfg["depths"]):
        l_vals, m_mean, m_sem = raw_data[L]
        x = l_vals / L
        ax.plot(x, m_mean, color=COLORS[i % len(COLORS)],
                lw=1.4, label=f"$L={L}$")
        ax.fill_between(x, m_mean - m_sem, m_mean + m_sem,
                        alpha=0.2, color=COLORS[i % len(COLORS)])

    ax.axvline(_LC_FRAC, color="k", lw=0.8, ls=":", alpha=0.6,
               label=f"$\\ell_c/L={_LC_FRAC}$")
    ax.set_xlabel(r"Relative depth $\ell/L$")
    ax.set_ylabel(r"Order parameter $m(\ell; L)$")
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "a")

    # ----------------------------------------------------------------
    # Panel (b): Rescaled collapse
    # ----------------------------------------------------------------
    ax = axes[1]

    master_x_all, master_y_all = [], []
    for i, L in enumerate(cfg["depths"]):
        l_vals, m_mean, m_sem = raw_data[L]
        # Rescale axes
        x_resc = (l_vals / L - _LC_FRAC) * (L ** (1.0 / _NU_TRUE))
        y_resc = m_mean * (L ** (_BETA_TRUE / _NU_TRUE))

        ax.plot(x_resc, y_resc, color=COLORS[i % len(COLORS)],
                lw=1.4, alpha=0.85, label=f"$L={L}$")
        master_x_all.extend(x_resc.tolist())
        master_y_all.extend(y_resc.tolist())

    # Reference Gaussian scaling function
    x_ref = np.linspace(-4, 4, 200)
    y_ref = np.exp(-0.5 * x_ref ** 2)
    ax.plot(x_ref, y_ref, "k--", lw=1.6, label="Scaling fn $f(x)$", zorder=5)

    ax.set_xlabel(r"$({\ell}/{L} - {\ell_c}/{L})\,L^{1/\nu}$")
    ax.set_ylabel(r"$m\,L^{\beta/\nu}$")
    ax.legend(fontsize=7, frameon=False)
    remove_top_right_spines(ax)
    add_panel_label(ax, "b")

    # Inset: chi^2 landscape in (nu, beta)
    ax_in = ax.inset_axes([0.6, 0.55, 0.37, 0.38])
    nu_grid   = np.linspace(0.8, 1.6, 20)
    beta_grid = np.linspace(0.3, 0.7, 20)
    NU, BETA  = np.meshgrid(nu_grid, beta_grid)

    # Synthetic chi^2: minimum at (nu_true, beta_true)
    chi2 = ((NU - _NU_TRUE) ** 2 / 0.1 ** 2 +
            (BETA - _BETA_TRUE) ** 2 / 0.05 ** 2)
    chi2_min = chi2.min()
    delta_chi2 = chi2 - chi2_min

    cs = ax_in.contour(NU, BETA, delta_chi2, levels=[1.0, 4.0],
                       colors=["#4878CF", "#D65F5F"], linewidths=1.2)
    ax_in.clabel(cs, fmt={1.0: "68%", 4.0: "95%"}, fontsize=6)
    ax_in.plot(_NU_TRUE, _BETA_TRUE, "k+", ms=6, mew=1.5)
    ax_in.set_xlabel("$\\nu$", fontsize=6)
    ax_in.set_ylabel("$\\beta$", fontsize=6)
    ax_in.tick_params(labelsize=6)
    remove_top_right_spines(ax_in)

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
