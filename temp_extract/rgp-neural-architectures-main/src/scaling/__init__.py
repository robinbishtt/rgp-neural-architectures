"""
src/scaling — Finite-size scaling (FSS) analysis engine.

Core FSS:
    FSSFitter                 — full FSS ansatz fitting
    CriticalExponentExtractor — extract ν, η, γ, β from FSS data
    DataCollapseVerifier      — quality-of-collapse objective (chi²)
    BootstrapConfidence       — bootstrap CI for any estimator
    CollapseQualityMetrics    — collapse quality metrics (Q-value, residual)

Scaling law analysis:
    ScalingLawFitter          — fit log/power/linear law to any observable
    ScalingFitResult          — result dataclass (params, R², AIC, BIC)

Width / depth analysis:
    WidthScalingAnalyzer      — scaling of correlation length vs width N
    WidthScalingResult        — stores ν, N_c, fitted curve
    DepthWidthAnalyzer        — joint (depth, width) surface mapping
    DepthWidthSurface         — result dataclass for 2D surface

Phase diagram:
    PhaseDiagramMapper        — map (σ_w, σ_b) plane into phase regions
    PhasePoint                — single (σ_w, σ_b, χ₁, phase) record

Comparative / spectral:
    ExponentComparison        — compare fitted exponents across architectures
    ExponentComparisonResult  — result dataclass
    SpectralScalingAnalyzer   — per-layer Jacobian spectrum vs MP law
    SpectralScalingResult     — spectral scaling result dataclass
"""
from src.scaling.fss_analysis        import FSSFitter
from src.scaling.critical_exponents  import CriticalExponentExtractor, CriticalExponentResult
from src.scaling.data_collapse       import DataCollapseVerifier, CollapseQuality
from src.scaling.bootstrap           import BootstrapConfidence, BootstrapResult
from src.scaling.collapse_quality    import CollapseQualityMetrics
from src.scaling.scaling_law_fitter  import ScalingLawFitter, ScalingFitResult
from src.scaling.width_scaling       import WidthScalingAnalyzer, WidthScalingResult
from src.scaling.depth_width_analyzer import DepthWidthAnalyzer, DepthWidthSurface
from src.scaling.phase_diagram       import PhaseDiagramMapper, PhasePoint
from src.scaling.exponent_comparison import ExponentComparison, ExponentComparisonResult
from src.scaling.spectral_scaling    import SpectralScalingAnalyzer, SpectralScalingResult

__all__ = [
    # Core FSS
    "FSSFitter",
    "CriticalExponentExtractor", "CriticalExponentResult",
    "DataCollapseVerifier", "CollapseQuality",
    "BootstrapConfidence", "BootstrapResult",
    "CollapseQualityMetrics",
    # Scaling law
    "ScalingLawFitter", "ScalingFitResult",
    # Width / depth
    "WidthScalingAnalyzer", "WidthScalingResult",
    "DepthWidthAnalyzer", "DepthWidthSurface",
    # Phase diagram
    "PhaseDiagramMapper", "PhasePoint",
    # Comparative / spectral
    "ExponentComparison", "ExponentComparisonResult",
    "SpectralScalingAnalyzer", "SpectralScalingResult",
]
