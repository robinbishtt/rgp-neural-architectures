import numpy as np
from typing import Optional, Tuple
from scipy.stats import gaussian_kde, kstest
from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution  
from src.core.spectral.wigner_semicircle import WignerSemicircleDistribution  
from src.core.spectral.tracy_widom import TracyWidomDistribution              
def empirical_spectral_density(
    eigenvalues: np.ndarray,
    bw_method: float = 0.25,
    n_points: int = 500,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ev = np.asarray(eigenvalues, dtype=float)
    if len(ev) < 2:
        return np.array([]), np.array([])
    if xlim is None:
        xlim = (ev.min() * 0.9, ev.max() * 1.1)
    x_grid = np.linspace(xlim[0], xlim[1], n_points)
    kde    = gaussian_kde(ev, bw_method=bw_method)
    return x_grid, kde(x_grid)