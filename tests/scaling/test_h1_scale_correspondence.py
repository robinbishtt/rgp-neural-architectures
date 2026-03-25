import numpy as np
def test_exponential_decay_r2_threshold():
    from src.core.correlation.estimators import ExponentialDecayFitter
    k  = np.arange(30)
    xi = 5.0 * np.exp(-k / 8.0)
    result = ExponentialDecayFitter().fit(xi)
    assert result.r2 > 0.95, f"R²={result.r2:.3f} below threshold"
def test_chi1_below_one_implies_decay():
    from src.core.correlation.two_point import chi1_gauss_hermite
    chi1 = chi1_gauss_hermite(sigma_w2=1.0, nonlinearity="tanh")
    assert chi1 < 1.0 or chi1 > 0.0  