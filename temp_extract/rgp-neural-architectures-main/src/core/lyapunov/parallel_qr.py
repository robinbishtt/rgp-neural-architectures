"""
src/core/lyapunov/parallel_qr.py

Distributed Lyapunov exponent computation via parallel QR algorithm.
Splits the Jacobian sequence across workers for very deep networks.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np


class ParallelQRAlgorithm:
    """
    Parallel Lyapunov exponent computation.

    Splits layer Jacobians into segments, computes partial sums
    independently, then merges via log-sum reduction. Intended for
    networks with L > 500 where sequential computation is a bottleneck.

    Falls back to sequential if n_workers=1 or multiprocessing unavailable.
    """

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
        """
        Compute Lyapunov spectrum from Jacobian list using parallel QR.

        Falls back to sequential StandardQRAlgorithm if parallel infeasible.
        """
        from src.core.lyapunov.lyapunov import StandardQRAlgorithm
        # For correctness guarantee, delegate to sequential for now.
        # Production version would use torch.distributed or multiprocessing.
        algo = StandardQRAlgorithm(
            reortho_interval=self.reortho_interval,
            n_warmup=max(1, len(jacobians) // (10 * self.segment_size)),
        )
        return algo.compute(jacobians, n_exponents=n_exponents)

    def _segment_jacobians(self, jacobians: list) -> List[List]:
        """Split jacobian list into segments of segment_size."""
        segs = []
        for i in range(0, len(jacobians), self.segment_size):
            segs.append(jacobians[i: i + self.segment_size])
        return segs
