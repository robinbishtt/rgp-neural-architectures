from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
@dataclass
class PhasePoint:
    sigma_w: float
    sigma_b: float
    mle:     float
    regime:  str   
    xi:      Optional[float] = None
class PhaseDiagramMapper:
    def __init__(
        self,
        n_points:  int   = 50,
        n_gauss:   int   = 1000,
        sigma_w_range: Tuple[float, float] = (0.1, 3.0),
        sigma_b_range: Tuple[float, float] = (0.0, 2.0),
    ) -> None:
        self.n_points      = n_points
        self.sigma_w_range = sigma_w_range
        self.sigma_b_range = sigma_b_range
        z, w = np.polynomial.hermite.hermgauss(n_gauss)
        self._z = z * np.sqrt(2)
        self._w = w / np.sqrt(np.pi)
    def _chi1(self, sigma_w: float, sigma_b: float) -> float:
        q = 1.0
        for _ in range(100):
            h   = sigma_w * np.sqrt(max(q, 0)) * self._z + sigma_b
            q_new = float(np.dot(self._w, np.tanh(h) ** 2))
            if abs(q_new - q) < 1e-6:
                break
            q = q_new
        h = sigma_w * np.sqrt(max(q, 0)) * self._z + sigma_b
        chi1 = sigma_w ** 2 * float(np.dot(self._w, (1.0 - np.tanh(h) ** 2) ** 2))
        return chi1
    def compute_full_diagram(self) -> List[PhasePoint]:
        sigma_ws = np.linspace(*self.sigma_w_range, self.n_points)
        sigma_bs = np.linspace(*self.sigma_b_range, self.n_points)
        points   = []
        for sw in sigma_ws:
            for sb in sigma_bs:
                chi1 = self._chi1(sw, sb)
                mle  = float(np.log(max(chi1, 1e-12)))
                if chi1 < 0.95:
                    regime = "ordered"
                elif chi1 > 1.05:
                    regime = "chaotic"
                else:
                    regime = "critical"
                xi = -1.0 / np.log(max(chi1, 1e-12)) if chi1 < 1.0 and chi1 > 0 else float("inf")
                points.append(PhasePoint(
                    sigma_w=sw, sigma_b=sb,
                    mle=mle, regime=regime, xi=xi,
                ))
        return points
    def critical_line(
        self, sigma_b_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if sigma_b_values is None:
            sigma_b_values = np.linspace(*self.sigma_b_range, self.n_points)
        critical = []
        for sb in sigma_b_values:
            lo, hi = self.sigma_w_range
            for _ in range(60):
                mid  = (lo + hi) / 2.0
                chi1 = self._chi1(mid, sb)
                if chi1 < 1.0:
                    lo = mid
                else:
                    hi = mid
            critical.append((sb, (lo + hi) / 2.0))
        return np.array(critical)