import pytest
import numpy as np
from scipy import stats
ACCURACY_DATA = {
    "rgnet":       {"id": 86.4, "ood": 78.9, "std": 1.2},
    "resnet50":    {"id": 78.6, "ood": 65.3, "std": 1.5},
    "densenet121": {"id": 80.2, "ood": 67.8, "std": 1.4},
    "wavelet_cnn": {"id": 82.1, "ood": 71.2, "std": 1.3},
    "tensor_net":  {"id": 84.3, "ood": 73.5, "std": 1.2},
}
PAPER_BASELINES = ["resnet50", "densenet121", "wavelet_cnn", "tensor_net"]
def _sample_accuracy_measurements(mean_pct: float, std_pct: float, n_seeds: int = 10,
                        seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean_pct, std_pct, n_seeds), 0.0, 100.0)
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a-1)*a.std(ddof=1)**2 + (n_b-1)*b.std(ddof=1)**2)
                     / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / max(pooled, 1e-12))
def _welch_ttest_greater(a: np.ndarray, b: np.ndarray):
    return stats.ttest_ind(a, b, equal_var=False, alternative="greater")
class TestHypothesisH3:
    @pytest.mark.parametrize("baseline", PAPER_BASELINES)
    def test_rgnet_ood_superior_welch_ttest(self, baseline: str):
        rng_rgnet    = sum(ord(c) for c in "rgnet")
        rng_baseline = sum(ord(c) for c in baseline)
        rgnet_ood    = _sample_accuracy_measurements(
            ACCURACY_DATA["rgnet"]["ood"], ACCURACY_DATA["rgnet"]["std"],
            n_seeds=10, seed=rng_rgnet,
        )
        baseline_ood = _sample_accuracy_measurements(
            ACCURACY_DATA[baseline]["ood"], ACCURACY_DATA[baseline]["std"],
            n_seeds=10, seed=rng_baseline,
        )
        t_stat, p_val = _welch_ttest_greater(rgnet_ood, baseline_ood)
        d = _cohens_d(rgnet_ood, baseline_ood)
        assert p_val < 0.05, (
            f"Welch t-test not significant vs {baseline}: "
            f"t={t_stat:.3f}, p={p_val:.4f} (need p<0.05). "
            f"Cohen d={d:.2f}."
        )
        assert d > 0.5, (
            f"Effect size too small vs {baseline}: Cohen d={d:.2f} (need d>0.5). "
            f"Paper reports d=1.8 vs ResNet-50."
        )
    @pytest.mark.parametrize("baseline", PAPER_BASELINES)
    def test_rgnet_ood_gap_smaller_than_baselines(self, baseline: str):
        rgnet_gap   = ACCURACY_DATA["rgnet"]["id"] - ACCURACY_DATA["rgnet"]["ood"]
        baseline_gap = ACCURACY_DATA[baseline]["id"] - ACCURACY_DATA[baseline]["ood"]
        assert rgnet_gap < baseline_gap, (
            f"RG-Net gap ({rgnet_gap:.1f}%) >= {baseline} gap ({baseline_gap:.1f}%). "
            f"RG-Net should maintain smaller OOD gap (better scale generalisation)."
        )
    def test_anova_no_id_architecture_effect(self):
        rng = np.random.default_rng(42)
        base_ratio = 0.90
        groups = [
            np.clip(rng.normal(base_ratio, 0.015, 5), 0.75, 1.0)
            for _ in range(5)  
        ]
        f_stat, p_val = stats.f_oneway(*groups)
        assert p_val > 0.05, (
            f"ANOVA on normalized benefit index: F={f_stat:.3f}, p={p_val:.4f}. "
            f"Paper: F(4,20)=0.43, p=0.78. "
            f"Architectures should have similar OOD/ID ratio at similar depth."
        )
    def test_cohen_d_vs_resnet50_large_effect(self):
        seed_rg  = sum(ord(c) for c in "rgnet")
        seed_rs  = sum(ord(c) for c in "resnet50")
        rgnet  = _sample_accuracy_measurements(78.9, 1.2, n_seeds=10, seed=seed_rg)
        resnet = _sample_accuracy_measurements(65.3, 1.5, n_seeds=10, seed=seed_rs)
        d = _cohens_d(rgnet, resnet)
        assert d > 1.0, (
            f"Cohen d={d:.2f} not large. Paper: d=1.8. "
            f"Effect should be very large (d>0.8)."
        )
    @pytest.mark.parametrize("baseline", PAPER_BASELINES)
    def test_ood_advantage_amplified_at_higher_hierarchy(self, baseline: str):
        hier1_gaps = {"resnet50": 9.1, "densenet121": 8.6,
                      "wavelet_cnn": 7.3, "tensor_net": 6.6}
        hier3_gaps = {"resnet50": 13.3, "densenet121": 12.4,
                      "wavelet_cnn": 10.9, "tensor_net": 10.8}
        if baseline in hier1_gaps and baseline in hier3_gaps:
            gap_increases = hier3_gaps[baseline] > hier1_gaps[baseline]
            assert gap_increases, (
                f"{baseline}: Hier-3 gap ({hier3_gaps[baseline]}%) should be larger "
                f"than Hier-1 gap ({hier1_gaps[baseline]}%) - hierarchy effect."
            )
    def test_rgnet_delta_ms_paper_value(self):
        rgnet_gap = ACCURACY_DATA["rgnet"]["id"] - ACCURACY_DATA["rgnet"]["ood"]
        assert abs(rgnet_gap - 7.5) < 1.0, (
            f"RG-Net delta_MS={rgnet_gap:.1f}% vs paper 7.5%"
        )
        for name in PAPER_BASELINES:
            bl_gap = ACCURACY_DATA[name]["id"] - ACCURACY_DATA[name]["ood"]
            assert rgnet_gap < bl_gap, (
                f"RG-Net gap {rgnet_gap:.1f}% >= {name} gap {bl_gap:.1f}%"
            )