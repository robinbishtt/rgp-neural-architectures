"""
src/core/spectral/spectral.py

Random Matrix Theory spectral distributions.
Re-exports canonical implementations from submodules.
"""
import numpy as np
from typing import Optional, Tuple
from scipy.stats import gaussian_kde, kstest

# Canonical classes - single source of truth in submodules
from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution  # noqa: F401
from src.core.spectral.wigner_semicircle import WignerSemicircleDistribution  # noqa: F401
from src.core.spectral.tracy_widom import TracyWidomDistribution              # noqa: F401


def empirical_spectral_density(
    eigenvalues: np.ndarray,
    bw_method: float = 0.25,
    n_points: int = 500,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KDE-smoothed empirical spectral density.

    Parameters
    ----------
    eigenvalues : 1-D array of measured eigenvalues
    bw_method   : KDE bandwidth (default 0.25)
    n_points    : grid resolution (default 500)
    xlim        : (x_min, x_max) for the grid; auto-inferred if None

    Returns
    -------
    (x_grid, density) - both 1-D arrays of length n_points
    """
    ev = np.asarray(eigenvalues, dtype=float)
    if len(ev) < 2:
        return np.array([]), np.array([])
    if xlim is None:
        xlim = (ev.min() * 0.9, ev.max() * 1.1)
    x_grid = np.linspace(xlim[0], xlim[1], n_points)
    kde    = gaussian_kde(ev, bw_method=bw_method)
    return x_grid, kde(x_grid)
