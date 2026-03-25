from __future__ import annotations
import matplotlib as mpl
AXIS_LABEL_SIZE    = 7
TICK_LABEL_SIZE    = 6
LEGEND_FONT_SIZE   = 6
TITLE_FONT_SIZE    = 7
ANNOTATION_SIZE    = 6
PANEL_LABEL_SIZE   = 8   
CAPTION_SIZE       = 7
SINGLE_COL_WIDTH   = 3.46   
DOUBLE_COL_WIDTH   = 7.09   
MAX_HEIGHT         = 9.45   
def apply_publication_fonts() -> None:
    mpl.rcParams.update({
        :               "sans-serif",
        :           ["Arial", "Helvetica", "DejaVu Sans"],
        :                 7,
        :            AXIS_LABEL_SIZE,
        :            TITLE_FONT_SIZE,
        :           TICK_LABEL_SIZE,
        :           TICK_LABEL_SIZE,
        :           LEGEND_FONT_SIZE,
        :               False,
        :          "custom",
        :               "Arial",
        :               "Arial:italic",
        :               "Arial:bold",
        :              42,    
        :               42,
    })
def add_panel_label(
    ax: "mpl.axes.Axes",
    label: str,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = PANEL_LABEL_SIZE,
    bold: bool = True,
) -> None:
    weight = "bold" if bold else "normal"
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        va="top", ha="right",
        fontfamily="sans-serif",
    )
def format_log_axis(ax: "mpl.axes.Axes", axis: str = "both") -> None:
    if axis in ("x", "both"):
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs="auto"))
    if axis in ("y", "both"):
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs="auto"))
def remove_top_right_spines(ax: "mpl.axes.Axes") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
def set_axis_linewidth(ax: "mpl.axes.Axes", lw: float = 0.75) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(width=lw)