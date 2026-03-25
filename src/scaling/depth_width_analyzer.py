from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import RegularGridInterpolator
@dataclass
class DepthWidthSurface:
    depths:      np.ndarray      
    widths:      np.ndarray      
    surface:     np.ndarray      
    critical_L:  Optional[float] 
    critical_N:  Optional[float] 
class DepthWidthAnalyzer:
    def build_surface(
        self,
        depths:  np.ndarray,
        widths:  np.ndarray,
        values:  np.ndarray,
    ) -> DepthWidthSurface:
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
        critical_depths = []
        for j, N in enumerate(surface.widths):
            vals = surface.surface[:, j]
            idx  = np.searchsorted(vals, threshold)
            if idx == 0:
                critical_depths.append(float(surface.depths[0]))
            elif idx >= len(vals):
                critical_depths.append(float(surface.depths[-1]))
            else:
                x0, x1 = surface.depths[idx - 1], surface.depths[idx]
                y0, y1 = vals[idx - 1], vals[idx]
                frac   = (threshold - y0) / (y1 - y0 + 1e-12)
                critical_depths.append(float(x0 + frac * (x1 - x0)))
        return np.array(critical_depths)