# Paper–Code Correspondence

> Part of the [RGP documentation set](README.md).

This document maps every major claim in the paper to the code that implements or validates it.

## Main Theorems

| Theorem | Paper Location | Code Implementation | Proof Verification |
|---|---|---|---|
| Thm 1: Metric Contraction | Sec 3.1 | `src/core/fisher/fisher_metric.py:pullback()` | `src/proofs/theorem1_fisher_transform.py` |
| Thm 2: Exp. Correlation Decay | Sec 3.2 | `src/core/correlation/two_point.py:TwoPointCorrelation` | `src/proofs/theorem2_exponential_decay.py` |
| Thm 3: Log Depth Bound | Sec 3.3 | `src/proofs/theorem3_depth_scaling.py:lmin_theoretical()` | `src/proofs/theorem3_depth_scaling.py` |
| Lemma: Critical Init | Sec 3, Def 2 | `src/core/correlation/two_point.py:critical_sigma_w2()` | `src/proofs/lemma_critical_init.py` |

## Hypotheses

| Hypothesis | Paper Claim | Code Experiment | Expected Output |
|---|---|---|---|
| H1 Scale Corr. | R²=0.997±0.001 | `experiments/h1_scale_correspondence/run_h1_validation.py` | `results/h1/h1_results.json` |
| H2 Depth Scaling | α̂=0.98±0.06, p<0.001 | `experiments/h2_depth_scaling/run_h2_validation.py` | `results/h2/h2_results.json` |
| H3 Arch. Adv. | Cohen d=1.8, p=0.006 | `experiments/h3_multiscale_generalization/run_h3_validation.py` | `results/h3/h3_results.json` |

## Figures

| Figure | Caption | Generator Script | Data Source |
|---|---|---|---|
| Fig 1 | RG correspondence overview | `figures/manuscript/generate_figure1.py` | Synthetic |
| Fig 2 | RG operator decomposition | `figures/manuscript/generate_figure2.py` | Synthetic |
| Fig 3 | H1 validation (3 panels) | `figures/manuscript/generate_figure3.py` | `results/h1/` |
| Fig 4 | H2 depth scaling law | `figures/manuscript/generate_figure4.py` | `results/h2/` |
| Fig 5 | H3 generalization | `figures/manuscript/generate_figure5.py` | `results/h3/` |

## Critical Hyperparameters

| Parameter | Paper Value | Code Location |
|---|---|---|
| σ_w (critical, tanh) | 1.4 | `config/architectures/rg_net.yaml: sigma_w` |
| σ_b (critical, tanh) | 0.3 | `config/architectures/rg_net.yaml: sigma_b` |
| χ₁ at criticality | ≈1.000 | `src/core/correlation/two_point.py:chi1_gauss_hermite` |
| ξ_depth (tanh critical) | ≈100 | Computed as -1/log(χ₁) |
| L_min threshold | 95% accuracy | `config/experiments/h2_depth_scaling.yaml: accuracy_threshold` |
| ξ_data values (Hier-1..5) | {5, 15, 50, 100, 200} | `config/experiments/h2_depth_scaling.yaml: correlation_lengths` |
| Batch size | 256 | `config/training/base_training.yaml: batch_size` |
| Optimizer | Adam (lr=1e-3) | `config/training/base_training.yaml: optimizer` |

## Known Mismatches (Fixed in This Version)

1. **Fisher metric formula**: Paper uses pullback g=J^T G J; code previously used pushforward J G J^T. **Fixed in `fisher_metric.py`**.
2. **H3 baselines**: Paper compares Wavelet-CNN and Tensor-Net; code previously used MLP and VGG. **Fixed: new `wavelet_baseline.py` and `tensor_net_baseline.py`**.
3. **H3 statistical test**: Paper uses Welch's t-test; code previously used Wilcoxon as primary. **Fixed in `run_h3_validation.py`**.
4. **H2 threshold**: Paper uses 95% accuracy; code previously used 85%. **Fixed in `run_h2_validation.py`**.
5. **σ_w, σ_b defaults**: Paper uses (1.4, 0.3); code defaults were (1.0, 0.05). **Fixed across all configs**.
