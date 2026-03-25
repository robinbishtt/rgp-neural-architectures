import numpy as np
def test_logarithmic_scaling_r2():
    from src.proofs.theorem3_depth_scaling import verify_logarithmic_scaling
    assert verify_logarithmic_scaling(kc=5.0, r2_threshold=0.90)
def test_lmin_increases_with_xi():
    from src.proofs.theorem3_depth_scaling import lmin_theoretical
    xi = np.array([2.0, 5.0, 10.0, 20.0])
    l  = lmin_theoretical(xi, kc=5.0)
    assert (np.diff(l) > 0).all(), "L_min should increase with xi_data"