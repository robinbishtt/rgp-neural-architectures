import numpy as np
def _goe_max_eigenvalue(n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    H   = (A + A.T) / np.sqrt(2 * n)
    return float(np.linalg.eigvalsh(H)[-1])
def _centre_scale_goe(lam_max: float, n: int) -> float:
    mu    = 2.0
    sigma = n ** (-2.0 / 3.0)
    return (lam_max - mu) / sigma
class TestTracyWidom:
    def test_max_eigenvalue_near_edge(self):
        n    = 200
        lam  = _goe_max_eigenvalue(n, seed=42)
        assert 1.5 < lam < 2.5, (
            f"GOE max eigenvalue {lam:.4f} not near expected edge of 2.0."
        )
    def test_scaled_statistic_finite(self):
        n   = 200
        lam = _goe_max_eigenvalue(n, seed=7)
        tw  = _centre_scale_goe(lam, n)
        assert np.isfinite(tw), f"TW statistic is non-finite: {tw}."
    def test_tw_statistics_collection_negative_mean(self):
        n       = 150
        n_mats  = 50
        stats   = [
            _centre_scale_goe(_goe_max_eigenvalue(n, seed=i), n)
            for i in range(n_mats)
        ]
        mean_tw = np.mean(stats)
        assert mean_tw < 0.5, (
            f"TW sample mean {mean_tw:.3f} not consistent with TW distribution."
        )
    def test_max_eigenvalue_not_below_bulk(self):
        for seed in range(10):
            lam = _goe_max_eigenvalue(200, seed=seed)
            assert lam > 0.0, (
                f"GOE max eigenvalue {lam:.4f} is below zero (inside Wigner bulk edge)."
            )