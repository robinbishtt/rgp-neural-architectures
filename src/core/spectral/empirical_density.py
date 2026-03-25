from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.stats import gaussian_kde
def empirical_spectral_density(
    eigenvalues: np.ndarray,
    bw_method: float = 0.25,
    n_points: int = 500,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ev = np.asarray(eigenvalues, dtype=float)
    if xlim is None:
        xlim = (max(0.0, ev.min() * 0.9), ev.max() * 1.1)
    x_grid = np.linspace(xlim[0], xlim[1], n_points)
    if len(ev) < 2:
        return x_grid, np.zeros(n_points)
    kde    = gaussian_kde(ev, bw_method=bw_method)
    return x_grid, kde(x_grid)