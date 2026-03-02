"""
src/core/lyapunov — Lyapunov spectrum via Benettin QR algorithm.

CANONICAL IMPLEMENTATIONS (imported here):
    StandardQRAlgorithm  — src/core/lyapunov/standard_qr.py
    AdaptiveQRAlgorithm  — src/core/lyapunov/adaptive_qr.py
    ParallelQRAlgorithm  — src/core/lyapunov/parallel_qr.py

NOTE: lyapunov.py is the original monolithic file retained for internal
use by parallel_qr.py. Do NOT import StandardQRAlgorithm from lyapunov.py
directly — its .compute() returns np.ndarray, not LyapunovResult.
The standalone standard_qr.py version correctly returns LyapunovResult
with .mle, .exponents, .kaplan_yorke_dim, and .regime fields.
"""
from src.core.lyapunov.standard_qr import (
    StandardQRAlgorithm,
    LyapunovResult,
)
from src.core.lyapunov.adaptive_qr import AdaptiveQRAlgorithm
from src.core.lyapunov.parallel_qr import ParallelQRAlgorithm

# Utility functions — still live in the monolith
from src.core.lyapunov.lyapunov import (
    detect_regime,
    kaplan_yorke_dimension,
    analyze_lyapunov,
)

__all__ = [
    "StandardQRAlgorithm",
    "AdaptiveQRAlgorithm",
    "ParallelQRAlgorithm",
    "LyapunovResult",
    "detect_regime",
    "kaplan_yorke_dimension",
    "analyze_lyapunov",
]
