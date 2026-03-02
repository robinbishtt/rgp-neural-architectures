"""
src/scaling/depth_width_analyzer.py

DepthWidthAnalyzer: jointly analyzes how observables scale with both
depth L and width N, computing the full 2D phase diagram of the
RGP system in the (L, N) parameter space.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class DepthWidthSurface:
    depths:      np.ndarray      # 1D array of L values
    widths:      np.ndarray      # 1D array of N values
    surface:     np.ndarray      # 2D array shape (n_depths, n_widths)
    critical_L:  Optional[float] # critical depth at target width
    critical_N:  Optional[float] # critical width at target depth


class DepthWidthAnalyzer:
    """
    Analyzes the joint (L, N) scaling surface of RGP observables.

    Maps the observable O(L, N) over a grid of depths and widths,
    identifies phase boundaries, and extracts scaling exponents in
    both directions simultaneously.

    Supports the H2 hypothesis validation by providing a 2D view
    of how the minimum network depth required to achieve a target
    generalization scales with both width and input correlation length.
    """

    def build_surface(
        self,
        depths:  np.ndarray,
        widths:  np.ndarray,
        values:  np.ndarray,
    ) -> DepthWidthSurface:
        """
        Build the 2D observable surface O(L, N).

        Args:
            depths: 1D sorted array of depth values L
            widths: 1D sorted array of width values N
            values: 2D array of shape (len(depths), len(widths))

        Returns:
            DepthWidthSurface with interpolator
        """
        depths = np.asarray(depths, float)
        widths = np.asarray(widths, float)
        values = np.asarray(values, float)
        assert values.shape == (len(depths), len(widths))

        return DepthWidthSurface(
            depths=depths,
            widths=widths,
            surface=values,
            critical_L=None,
            critical_N=None,
        )

    def extract_depth_slice(
        self,
        surface: DepthWidthSurface,
        target_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract O(L) at a fixed target width via linear interpolation.

        Returns:
            (depths, observable_values) arrays
        """
        interp = RegularGridInterpolator(
            (surface.depths, surface.widths),
            surface.surface,
            method="linear",
        )
        pts = np.column_stack([
            surface.depths,
            np.full_like(surface.depths, target_width),
        ])
        return surface.depths, interp(pts)

    def find_phase_boundary(
        self,
        surface:   DepthWidthSurface,
        threshold: float,
    ) -> np.ndarray:
        """
        Find the depth L*(N) at which O(L, N) crosses a threshold for each width.

        Returns:
            Array of critical depths L*(N) for each width in surface.widths.
        """
        critical_depths = []
        for j, N in enumerate(surface.widths):
            vals = surface.surface[:, j]
            idx  = np.searchsorted(vals, threshold)
            if idx == 0:
                critical_depths.append(float(surface.depths[0]))
            elif idx >= len(vals):
                critical_depths.append(float(surface.depths[-1]))
            else:
                # Linear interpolation
                x0, x1 = surface.depths[idx - 1], surface.depths[idx]
                y0, y1 = vals[idx - 1], vals[idx]
                frac   = (threshold - y0) / (y1 - y0 + 1e-12)
                critical_depths.append(float(x0 + frac * (x1 - x0)))
        return np.array(critical_depths)
 