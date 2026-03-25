from __future__ import annotations
from typing import Dict, List
import matplotlib.colors as mcolors
import matplotlib as _mpl_
import numpy as np
PRIMARY: Dict[str, str] = {
    :   "#1f77b4",   
    :   "#ff7f0e",   
    :   "#2ca02c",   
    :   "#9467bd",   
    :   "#d62728",   
    : "#8c564b",   
    :      "#e377c2",   
    :      "#7f7f7f",   
}
CORRELATION_LENGTH_CMAP = "viridis"
def correlation_length_colors(xi_values: List[float]) -> List[str]:
    cmap = _mpl_.colormaps[CORRELATION_LENGTH_CMAP] if hasattr(_mpl_, "colormaps") else plt.get_cmap(CORRELATION_LENGTH_CMAP)
    xi_arr = np.array(xi_values, dtype=float)
    norm = (xi_arr - xi_arr.min()) / ((xi_arr.max() - xi_arr.min()) + 1e-12)
    return [mcolors.to_hex(cmap(v)) for v in norm]
DEPTH_CMAP = "Blues"
def depth_colors(depths: List[int]) -> List[str]:
    cmap = _mpl_.colormaps[DEPTH_CMAP] if hasattr(_mpl_, "colormaps") else plt.get_cmap(DEPTH_CMAP)
    d_arr = np.array(depths, dtype=float)
    norm = 0.3 + 0.65 * (d_arr - d_arr.min()) / ((d_arr.max() - d_arr.min()) + 1e-12)
    return [mcolors.to_hex(cmap(v)) for v in norm]
PHASE: Dict[str, str] = {
    :   "#3182bd",   
    :  "#31a354",   
    :   "#e6550d",   
    :  "#636363",   
}
SPECTRAL: Dict[str, str] = {
    : "#1f77b4",   
    : "#d62728",   
    :    "#2ca02c",   
    : "#9467bd", 
    :      "#aec7e8",   
    :      "#ffbb78",   
}
LYAPUNOV: Dict[str, str] = {
    :  "#d62728",   
    :      "#31a354",   
    :  "#3182bd",   
    :       "#636363",   
}
FSS: Dict[str, str] = {
    :  "#1f77b4",  
    :        "#aec7e8",  
    :   "#d62728",  
    :        "#000000",  
}
CI_ALPHA = 0.20   
def model_color(model_name: str) -> str:
    key = model_name.lower().replace("-", "_")
    if key in PRIMARY:
        return PRIMARY[key]
    raise KeyError(
        f"No colour defined for model {model_name!r}. "
        f"Available: {list(PRIMARY.keys())}"
    )
def phase_color(phase: str) -> str:
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
    return mcolors.LinearSegmentedColormap.from_list(name, colors)
PANEL_LABELS = list("abcdefghijklmnopqrstuvwxyz")
def panel_label(idx: int) -> str:
    return PANEL_LABELS[idx % len(PANEL_LABELS)]