from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from figures.styles import use_publication_style
from figures.styles.font_config import (
    DOUBLE_COL_WIDTH, add_panel_label, remove_top_right_spines,
)
_TIER_LABELS = [
    ("Tier 1", "The Nervous System\n(Fisher · Jacobian · Lyapunov · RG Flow)"),
    ("Tier 2", "The Engine Room\n(Architectures · Training · Scaling)"),
    ("Tier 3", "The Audit Bureau\n(Unit · Integration · Stability · Ablation)"),
    ("Tier 4", "The Command Center\n(Config · Datasets · Containers)"),
    ("Tier 5", "The Publication Machine\n(Experiments · Figures · Scripts)"),
    ("ICL",    "Infrastructure Cross-Layer\n(Seed · Device · Telemetry · Checkpoint)"),
]
_TIER_COLORS = [
    ,   
    ,   
    ,   
    ,   
    ,   
    ,   
]
def _draw_tier_diagram(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(_TIER_LABELS)
    h = 0.13
    gap = 0.02
    total = n * h + (n - 1) * gap
    y0 = (1.0 - total) / 2.0
    for i, ((tier_id, desc), color) in enumerate(
        zip(_TIER_LABELS, _TIER_COLORS)
    ):
        y = y0 + i * (h + gap)
        rect = mpatches.FancyBboxPatch(
            (0.03, y), 0.94, h,
            boxstyle="round,pad=0.01",
            linewidth=0.6,
            edgecolor="white",
            facecolor=color,
            alpha=0.82,
        )
        ax.add_patch(rect)
        ax.text(
            0.10, y + h / 2, tier_id,
            va="center", ha="left",
            fontsize=6, fontweight="bold", color="white",
        )
        ax.text(
            0.28, y + h / 2, desc,
            va="center", ha="left",
            fontsize=5.5, color="white", linespacing=1.3,
        )
    ax.set_title("System Architecture", fontsize=7, pad=3)
def _draw_rg_cartoon(ax: plt.Axes) -> None:
    ax.set_xlim(0, 6)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    np.random.default_rng(42)
    x_fine = np.linspace(0, 6, 300)
    field = (
        0.6 * np.sin(2 * np.pi * x_fine / 0.4)
        + 0.3 * np.sin(2 * np.pi * x_fine / 1.5)
    )
    ax.plot(x_fine, field, lw=0.6, color="#4878CF", alpha=0.6)
    x_coarse = np.linspace(0, 6, 40)
    coarse = 0.3 * np.sin(2 * np.pi * x_coarse / 1.5)
    ax.plot(x_coarse, coarse, lw=1.4, color="#D65F5F")
    ax.annotate(
        , xy=(5.5, 0.9), xytext=(0.5, 0.9),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#636363"),
    )
    ax.text(3.0, 1.05, "RG coarse-graining", ha="center", fontsize=6, color="#636363")
    ax.text(0.5, -0.95, r"Fine scale $\xi_{\rm data}$",
            fontsize=6, color="#4878CF")
    ax.text(3.2, -0.95, r"Coarse scale $\xi(k)$",
            fontsize=6, color="#D65F5F")
    ax.set_title("Renormalization-Group Correspondence", fontsize=7, pad=3)
def _draw_xi_decay(ax: plt.Axes) -> None:
    k = np.arange(0, 25)
    xi_0 = 20.0
    k_c  = 8.0
    xi_k = xi_0 * np.exp(-k / k_c)
    ax.plot(k, xi_k, "-o", ms=3, lw=1.2,
            color="#4878CF", markerfacecolor="#4878CF", label=r"$\xi(k)$ measured")
    k_fit = np.linspace(0, 24, 200)
    xi_fit = xi_0 * np.exp(-k_fit / k_c)
    ax.plot(k_fit, xi_fit, "--", lw=0.8, color="#D65F5F",
            label=r"$\xi_0 e^{-k/k_c}$ fit")
    ax.axvline(x=k_c, color="#636363", lw=0.6, ls=":")
    ax.text(k_c + 0.4, 12, r"$k_c$", fontsize=6, color="#636363")
    xi_target = 1.0
    ax.axhline(y=xi_target, color="#6ACC65", lw=0.6, ls=":")
    ax.text(20, xi_target + 0.8, r"$\xi_{\rm target}$",
            fontsize=6, color="#6ACC65")
    ax.set_xlabel("Layer index $k$")
    ax.set_ylabel(r"Correlation length $\xi(k)$")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False)
    remove_top_right_spines(ax)
    ax.set_title(r"Exponential decay $\xi(k) = \xi_0 e^{-k/k_c}$", fontsize=7, pad=3)
def generate(
    output_path: str = "figures/out/fig1.pdf",
    fast_track: bool = False,
) -> None:
    use_publication_style()
    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.38)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    _draw_tier_diagram(ax_a)
    _draw_rg_cartoon(ax_b)
    _draw_xi_decay(ax_c)
    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    tag = "" if fast_track else ""
    print(f"Figure 1 saved: {out}{tag}")
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 1")
    p.add_argument("--output", default="figures/out/fig1.pdf")
    p.add_argument("--fast-track", action="store_true")
    return p.parse_args()
if __name__ == "__main__":
    args = _parse_args()
    generate(output_path=args.output, fast_track=args.fast_track)