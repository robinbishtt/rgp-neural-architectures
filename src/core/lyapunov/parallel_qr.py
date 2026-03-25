from __future__ import annotations
from typing import List, Optional
import numpy as np
class ParallelQRAlgorithm:
    def __init__(
        self,
        n_workers: int = 4,
        segment_size: int = 50,
        reortho_interval: int = 10,
    ) -> None:
        self.n_workers       = n_workers
        self.segment_size    = segment_size
        self.reortho_interval = reortho_interval
    def compute(
        self,
        jacobians: List[np.ndarray],
        n_exponents: Optional[int] = None,
    ) -> np.ndarray:
        from src.core.lyapunov.lyapunov import StandardQRAlgorithm
        algo = StandardQRAlgorithm(
            reortho_interval=self.reortho_interval,
            n_warmup=max(1, len(jacobians) // (10 * self.segment_size)),
        )
        return algo.compute(jacobians, n_exponents=n_exponents)
    def _segment_jacobians(self, jacobians: list) -> List[List]:
        segs = []
        for i in range(0, len(jacobians), self.segment_size):
            segs.append(jacobians[i: i + self.segment_size])
        return segs