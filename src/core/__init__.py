"""
src/core — Mathematical and physics foundations (Tier 1: Nervous System).

Submodules
----------
fisher/         Fisher Information Geometry
jacobian/       Jacobian analysis and computation strategies
spectral/       Random Matrix Theory distributions
correlation/    Two-point correlation functions and chi_1
lyapunov/       Lyapunov spectrum via QR algorithms
"""

from src.core.fisher.fisher_metric import FisherMetric
from src.core.fisher.eigenvalue_analyzer import FisherEigenvalueAnalyzer
from src.core.fisher.condition_tracker import FisherConditionTracker
from src.core.fisher.effective_dimension import FisherEffectiveDimension
from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
from src.core.fisher.analytic import FisherAnalyticCalculator

from src.core.jacobian.jacobian import (
    AutogradJacobian, JVPJacobian, VJPJacobian,
    FiniteDifferenceJacobian, CumulativeJacobian,
)
from src.core.jacobian.symbolic_jacobian import SymbolicJacobian

from src.core.spectral.spectral import (
    MarchenkoPasturDistribution,
    WignerSemicircleDistribution,
    TracyWidomDistribution,
)
from src.core.spectral.level_spacing import LevelSpacingDistribution
from src.core.spectral.empirical_density import empirical_spectral_density

from src.core.correlation.two_point import (
    TwoPointCorrelation, chi1_gauss_hermite, critical_sigma_w2,
)
from src.core.correlation.estimators import (
    FisherSpectrumMethod, ExponentialDecayFitter,
    MaximumLikelihoodEstimator, TransferMatrixMethod,
    CorrelationLengthResult,
)

from src.core.lyapunov.lyapunov import (
    StandardQRAlgorithm, AdaptiveQRAlgorithm,
    detect_regime, kaplan_yorke_dimension, analyze_lyapunov, LyapunovResult,
)
from src.core.lyapunov.parallel_qr import ParallelQRAlgorithm

__all__ = [
    "FisherMetric", "FisherEigenvalueAnalyzer", "FisherConditionTracker",
    "FisherEffectiveDimension", "FisherMonteCarloEstimator", "FisherAnalyticCalculator",
    "AutogradJacobian", "JVPJacobian", "VJPJacobian",
    "FiniteDifferenceJacobian", "CumulativeJacobian", "SymbolicJacobian",
    "MarchenkoPasturDistribution", "WignerSemicircleDistribution",
    "TracyWidomDistribution", "LevelSpacingDistribution", "empirical_spectral_density",
    "TwoPointCorrelation", "chi1_gauss_hermite", "critical_sigma_w2",
    "FisherSpectrumMethod", "ExponentialDecayFitter",
    "MaximumLikelihoodEstimator", "TransferMatrixMethod", "CorrelationLengthResult",
    "StandardQRAlgorithm", "AdaptiveQRAlgorithm", "ParallelQRAlgorithm",
    "detect_regime", "kaplan_yorke_dimension", "analyze_lyapunov", "LyapunovResult",
]
 