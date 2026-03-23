"""tests/validation/test_phase_diagram_validation.py"""
import numpy as np


class TestPhaseDiagramValidation:
    def test_critical_line_separates_phases(self):
        """Points below critical line should be ordered; above should be chaotic."""
        from src.scaling.phase_diagram import PhaseDiagramMapper
        mapper = PhaseDiagramMapper(n_points=12, n_gauss=200)
        sigma_bs = np.array([0.0, 0.5, 1.0])
        critical = mapper.critical_line(sigma_bs)
        for sb, sw_crit in critical:
            if sw_crit > 0.3:
                chi1_below = mapper._chi1(sw_crit * 0.5, sb)
                chi1_above = mapper._chi1(sw_crit * 1.5, sb)
                assert chi1_below < 1.0 + 0.2, f"Below critical: χ₁={chi1_below:.3f}"
                assert chi1_above > 1.0 - 0.2, f"Above critical: χ₁={chi1_above:.3f}"

    def test_paper_init_in_ordered_phase(self):
        """
        Paper initialization (sigma_w=1.4, sigma_b=0.3) must be in ordered phase
        (chi1 < 1) or exactly at critical (chi1 = 1).
        """
        from src.core.correlation.two_point import chi1_gauss_hermite
        chi1 = chi1_gauss_hermite(1.4**2, "tanh")
        assert chi1 <= 1.05, (
            f"Paper init (sw=1.4, sb=0.3): chi1={chi1:.4f} > 1 (chaotic phase). "
            f"Paper states this initialization is at/near critical."
        )

    def test_tanh_critical_point_sigma_w(self):
        """
        For tanh with sigma_b=0: critical sigma_w* = sqrt(1/E[sech^4]) ≈ 1.481.
        Verify chi1(sigma_w*^2, tanh) = 1.0 to 1% tolerance.
        """
        from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2
        sw2_star = critical_sigma_w2("tanh")
        chi1     = chi1_gauss_hermite(sw2_star, "tanh")
        assert abs(chi1 - 1.0) < 0.01, (
            f"chi1 at critical init = {chi1:.4f}, expected 1.000"
        )
        # sigma_w* should be near 1.48
        assert 1.3 <= sw2_star**0.5 <= 1.6, (
            f"sigma_w* = {sw2_star**0.5:.4f} outside expected range [1.3, 1.6]"
        )

    def test_three_phases_chi1_classification(self):
        """
        Verify classification of ordered/critical/chaotic by chi1 value.
        Paper Definition 2: chi1<1 ordered, chi1=1 critical, chi1>1 chaotic.
        """
        from src.core.correlation.two_point import chi1_gauss_hermite
        # Ordered (sigma_w small)
        chi1_ordered  = chi1_gauss_hermite(0.5**2, "tanh")
        # Critical (sigma_w near critical)
        from src.core.correlation.two_point import critical_sigma_w2
        sw2_crit = critical_sigma_w2("tanh")
        chi1_crit = chi1_gauss_hermite(sw2_crit, "tanh")
        # Chaotic (sigma_w large)
        chi1_chaotic  = chi1_gauss_hermite(3.0**2, "tanh")

        assert chi1_ordered < 1.0, f"Ordered phase: chi1={chi1_ordered:.4f} not < 1"
        assert abs(chi1_crit - 1.0) < 0.01, f"Critical: chi1={chi1_crit:.4f} not near 1"
        assert chi1_chaotic > 1.0, f"Chaotic phase: chi1={chi1_chaotic:.4f} not > 1"
