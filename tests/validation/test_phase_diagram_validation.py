"""tests/validation/test_phase_diagram_validation.py"""
import pytest
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
 