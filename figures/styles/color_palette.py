"""
figures/styles/color_palette.py

Consistent color scheme for all figures. Palette designed for:
  - Publication-quality (print + digital)
  - Colorblind accessibility (deuteranopia / protanopia safe)
  - Semantic consistency across all figures
"""

from __future__ import annotations
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib as _mpl_
import numpy as np


# ---------------------------------------------------------------------------
# Primary palette - used for main experimental series
# ---------------------------------------------------------------------------
PRIMARY: Dict[str, str] = {
    "rgp_tanh":   "#1f77b4",   # blue   - canonical RG-Net
    "rgp_relu":   "#ff7f0e",   # orange - ReLU variant
    "rgp_gelu":   "#2ca02c",   # green  - GELU variant
    "rgp_deep":   "#9467bd",   # purple - ultra-deep variant
    "baseline_resnet":   "#d62728",   # red    - ResNet baseline
    "baseline_densenet": "#8c564b",   # brown  - DenseNet baseline
    "baseline_mlp":      "#e377c2",   # pink   - MLP baseline
    "baseline_vgg":      "#7f7f7f",   # gray   - VGG baseline
}

# ---------------------------------------------------------------------------
# Correlation-length palette - spectral from short to long range
# ---------------------------------------------------------------------------
CORRELATION_LENGTH_CMAP = "viridis"

def correlation_length_colors(xi_values: List[float]) -> List[str]:
    """
    Return a list of hex colours, one per xi value, sampled from viridis.
    Shorter xi → dark purple; longer xi → yellow-green.
    """
    cmap = _mpl_.colormaps[CORRELATION_LENGTH_CMAP] if hasattr(_mpl_, "colormaps") else plt.get_cmap(CORRELATION_LENGTH_CMAP)
    xi_arr = np.array(xi_values, dtype=float)
    norm = (xi_arr - xi_arr.min()) / ((xi_arr.max() - xi_arr.min()) + 1e-12)
    return [mcolors.to_hex(cmap(v)) for v in norm]


# ---------------------------------------------------------------------------
# Depth palette - lighter to darker blue as L increases
# ---------------------------------------------------------------------------
DEPTH_CMAP = "Blues"

def depth_colors(depths: List[int]) -> List[str]:
    cmap = _mpl_.colormaps[DEPTH_CMAP] if hasattr(_mpl_, "colormaps") else plt.get_cmap(DEPTH_CMAP)
    d_arr = np.array(depths, dtype=float)
    norm = 0.3 + 0.65 * (d_arr - d_arr.min()) / ((d_arr.max() - d_arr.min()) + 1e-12)
    return [mcolors.to_hex(cmap(v)) for v in norm]


# ---------------------------------------------------------------------------
# Phase-diagram palette
# ---------------------------------------------------------------------------
PHASE: Dict[str, str] = {
    "ordered":   "#3182bd",   # blue
    "critical":  "#31a354",   # green
    "chaotic":   "#e6550d",   # red-orange
    "boundary":  "#636363",   # gray
}

# ---------------------------------------------------------------------------
# Spectral / RMT palette
# ---------------------------------------------------------------------------
SPECTRAL: Dict[str, str] = {
    "empirical": "#1f77b4",   # blue  - empirical eigenvalue histogram
    "mp_theory": "#d62728",   # red   - Marchenko-Pastur theoretical curve
    "wigner":    "#2ca02c",   # green - Wigner semi-circle
    "tracy_widom": "#9467bd", # purple - Tracy-Widom edge
    "bulk":      "#aec7e8",   # light blue - bulk region shading
    "edge":      "#ffbb78",   # light orange - edge region shading
}

# ---------------------------------------------------------------------------
# Lyapunov spectrum palette
# ---------------------------------------------------------------------------
LYAPUNOV: Dict[str, str] = {
    "positive":  "#d62728",   # red   - positive exponents (chaotic)
    "zero":      "#31a354",   # green - zero exponents (marginal)
    "negative":  "#3182bd",   # blue  - negative exponents (ordered)
    "sum":       "#636363",   # gray  - Lyapunov sum
}

# ---------------------------------------------------------------------------
# FSS / data-collapse palette
# ---------------------------------------------------------------------------
FSS: Dict[str, str] = {
    "collapsed":  "#1f77b4",  # blue  - post-collapse data
    "raw":        "#aec7e8",  # light - raw pre-collapse data
    "critical":   "#d62728",  # red   - critical point xi_c
    "fit":        "#000000",  # black - master curve fit
}

# ---------------------------------------------------------------------------
# Confidence-interval / error band style
# ---------------------------------------------------------------------------
CI_ALPHA = 0.20   # fill transparency for confidence bands

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def model_color(model_name: str) -> str:
    """
    Return colour for a named model. Raises KeyError for unknown models
    so callers notice missing palette entries immediately.
    """
    key = model_name.lower().replace("-", "_")
    if key in PRIMARY:
        return PRIMARY[key]
    raise KeyError(
        f"No colour defined for model {model_name!r}. "
        f"Available: {list(PRIMARY.keys())}"
    )


def phase_color(phase: str) -> str:
    """Return colour for a named dynamical phase."""
    key = phase.lower()
    if key in PHASE:
        return PHASE[key]
    raise KeyError(
        f"No colour defined for phase {phase!r}. "
        f"Available: {list(PHASE.keys())}"
    )


def make_colormap(
    colors: List[str], name: str = "custom"
) -> mcolors.LinearSegmentedColormap:
    """Create a linear segmented colormap from a list of hex colours."""
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


# ---------------------------------------------------------------------------
# Figure panel label helper
# ---------------------------------------------------------------------------

PANEL_LABELS = list("abcdefghijklmnopqrstuvwxyz")

def panel_label(idx: int) -> str:
    """Return lowercase panel label: a, b, c, … z."""
    return PANEL_LABELS[idx % len(PANEL_LABELS)]
 