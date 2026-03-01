"""tests/scaling/test_data_collapse.py — FSS collapse quality chi-squared."""
import numpy as np
from src.scaling.data_collapse import DataCollapseVerifier


def test_perfect_collapse_passes():
    x_scaled = np.linspace(-5, 5, 50)
    y_obs    = 1.0 / (1.0 + np.exp(-x_scaled))
    verifier = DataCollapseVerifier(r2_threshold=0.90)
    result   = verifier.verify(x_scaled, y_obs)
    assert result.passed, f"R²={result.r2:.3f}"
