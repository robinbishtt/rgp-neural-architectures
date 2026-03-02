"""
src/core/correlation — Correlation length dynamics and two-point functions.

Two ExponentialDecayFitter implementations exist intentionally:

  estimators.ExponentialDecayFitter:
      Simple interface: .fit(xi_values) — infers layer indices automatically.
      Returns CorrelationLengthResult.

  exponential_decay_fitter.ExponentialDecayFitter:
      Full interface: .fit(layers, xi_values, weights=None) — explicit layers.
      Returns ExponentialDecayFitResult with bootstrap CI and R² goodness-of-fit.
      This is the richer, production-grade version used by the H1 experiment.

Import the full version explicitly if you need bootstrap CIs:
    from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter
"""
from src.core.correlation.two_point import (
    TwoPointCorrelation,
    chi1_gauss_hermite,
    critical_sigma_w2,
)
from src.core.correlation.estimators import (
    FisherSpectrumMethod,
    ExponentialDecayFitter,       # simple: .fit(xi_values)
    MaximumLikelihoodEstimator,
    TransferMatrixMethod,
    CorrelationLengthResult,
)
from src.core.correlation.exponential_decay_fitter import (
    ExponentialDecayFitter        as ExponentialDecayFitterFull,
    ExponentialDecayFitResult,
)
# TransferMatrixCorrelation not present; TransferMatrixMethod is in estimators.py

__all__ = [
    # Two-point function
    "TwoPointCorrelation", "chi1_gauss_hermite", "critical_sigma_w2",
    # Correlation length estimators (simple interface)
    "FisherSpectrumMethod",
    "ExponentialDecayFitter",     # .fit(xi_values) → CorrelationLengthResult
    "MaximumLikelihoodEstimator",
    "TransferMatrixMethod",
    "CorrelationLengthResult",
    # Full-featured standalone estimator
    "ExponentialDecayFitterFull", # .fit(layers, xi_values) → ExponentialDecayFitResult
    "ExponentialDecayFitResult",

]
