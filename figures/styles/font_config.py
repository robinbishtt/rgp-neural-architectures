"""
figures/styles/font_config.py

Font configuration for publication-quality figures.
Arial / Helvetica for all text; math rendered via mathtext (no full LaTeX).
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Size constants (points)
# ---------------------------------------------------------------------------
AXIS_LABEL_SIZE    = 7
TICK_LABEL_SIZE    = 6
LEGEND_FONT_SIZE   = 6
TITLE_FONT_SIZE    = 7
ANNOTATION_SIZE    = 6
PANEL_LABEL_SIZE   = 8   # bold panel labels (a, b, c)
CAPTION_SIZE       = 7

# ---------------------------------------------------------------------------
# Figure dimensions (inches)
# ---------------------------------------------------------------------------
SINGLE_COL_WIDTH   = 3.46   # 88 mm
DOUBLE_COL_WIDTH   = 7.09   # 180 mm
MAX_HEIGHT         = 9.45   # 240 mm

# ---------------------------------------------------------------------------
# Apply font settings to rcParams
# ---------------------------------------------------------------------------

def apply_publication_fonts() -> None:
    """
    Apply publication-quality font settings globally.
    Call once at the start of each figure script.
    """
    mpl.rcParams.update({
        "font.family":               "sans-serif",
        "font.sans-serif":           ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":                 7,
        "axes.labelsize":            AXIS_LABEL_SIZE,
        "axes.titlesize":            TITLE_FONT_SIZE,
        "xtick.labelsize":           TICK_LABEL_SIZE,
        "ytick.labelsize":           TICK_LABEL_SIZE,
        "legend.fontsize":           LEGEND_FONT_SIZE,
        "text.usetex":               False,
        "mathtext.fontset":          "custom",
        "mathtext.rm":               "Arial",
        "mathtext.it":               "Arial:italic",
        "mathtext.bf":               "Arial:bold",
        "pdf.fonttype":              42,    # embed TrueType in PDF
        "ps.fonttype":               42,
    })


# ---------------------------------------------------------------------------
# Panel-label helper
# ---------------------------------------------------------------------------

def add_panel_label(
    ax: "mpl.axes.Axes",
    label: str,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = PANEL_LABEL_SIZE,
    bold: bool = True,
) -> None:
    """
    Add a bold panel label (a, b, c, …) to an axes in normalised coords.

    Parameters
    ----------
    ax     : target axes
    label  : string label, e.g. "a"
    x, y   : position in axes-fraction coordinates
    """
    weight = "bold" if bold else "normal"
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        va="top", ha="right",
        fontfamily="sans-serif",
    )


# ---------------------------------------------------------------------------
# Axis formatting helpers
# ---------------------------------------------------------------------------

def format_log_axis(ax: "mpl.axes.Axes", axis: str = "both") -> None:
    """Apply minor ticks and grid for logarithmic axes."""
    if axis in ("x", "both"):
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs="auto"))
    if axis in ("y", "both"):
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs="auto"))


def remove_top_right_spines(ax: "mpl.axes.Axes") -> None:
    """Remove top and right spines for clean publication style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_axis_linewidth(ax: "mpl.axes.Axes", lw: float = 0.75) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(width=lw)
 