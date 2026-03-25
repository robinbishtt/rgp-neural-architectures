import numpy as np
def _goe_eigenvalues(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    H   = (A + A.T) / np.sqrt(2 * n)
    return np.linalg.eigvalsh(H)
def _number_variance(ev: np.ndarray, interval_length: float) -> float:
    ev     = np.sort(ev)
    len(ev)
    counts = []
    step   = interval_length / 4.0
    pos    = ev[0]
    while pos + interval_length <= ev[-1]:
        count = ((ev >= pos) & (ev < pos + interval_length)).sum()
        counts.append(count)
        pos  += step
    counts = np.array(counts, dtype=float)
    return float(counts.var()) if len(counts) > 1 else 0.0
class TestNumberVariance:
    def test_number_variance_positive(self):
        ev = _goe_eigenvalues(200)
        var = _number_variance(ev, interval_length=0.5)
        assert var >= 0.0
    def test_goe_variance_less_than_poisson(self):
        n   = 400
        ev_goe     = _goe_eigenvalues(n)
        rng        = np.random.default_rng(42)
        ev_poisson = np.sort(rng.standard_normal(n))  
        L   = 1.0  
        var_goe     = _number_variance(ev_goe, L)
        var_poisson = _number_variance(ev_poisson, L)
        assert var_goe <= var_poisson * 3.0, (
            f"GOE variance {var_goe:.4f} not smaller than Poisson {var_poisson:.4f}."
        )
    def test_number_variance_increases_with_interval(self):
        ev = _goe_eigenvalues(300)
        intervals = [0.5, 1.0, 2.0]
        variances = [_number_variance(ev, L) for L in intervals]
        for i in range(1, len(variances)):
            assert variances[i] >= variances[i - 1] - 0.1, (
                f"sigma^2 decreased from L={intervals[i-1]} to L={intervals[i]}."
            )