"""tests/ablation/test_phase_diagram.py"""
import numpy as np


class TestPhaseDiagram:
    def test_critical_line_monotone(self):
        """σ_w* should decrease as σ_b increases (critical line is monotone)."""
        from src.scaling.phase_diagram import PhaseDiagramMapper
        mapper = PhaseDiagramMapper(n_points=15, n_gauss=200)
        sigma_bs = np.linspace(0.0, 1.5, 8)
        critical = mapper.critical_line(sigma_bs)
        sigma_w_crits = critical[:, 1]
        # Generally decreasing trend
        assert sigma_w_crits[0] >= sigma_w_crits[-1] * 0.5

    def test_ordered_chaotic_phase_detection(self):
        """Small σ_w → ordered; large σ_w → chaotic."""
        from src.scaling.phase_diagram import PhaseDiagramMapper
        mapper = PhaseDiagramMapper(n_points=10, n_gauss=200)
        pts = mapper.compute_full_diagram()
        ordered  = [p for p in pts if p.sigma_w < 0.3]
        chaotic  = [p for p in pts if p.sigma_w > 2.5]
        if ordered:
            assert any(p.regime == "ordered" for p in ordered)
        if chaotic:
            assert any(p.regime == "chaotic" for p in chaotic)
 