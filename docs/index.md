---
layout: default
title: Master Thesis
nav_order: 1
has_children: true
---

# RGP Neural Architectures

## Core thesis: depth as RG coarse-graining index

The repository operationalizes the claim
\[
\text{network depth } L \equiv \text{number of RG coarse-graining steps required to drive } \xi_{\text{data}} \to \xi_{\text{target}}.
\]
Layerwise dynamics are encoded by:
\[
G^{(\ell)} = \left(J^{(\ell)}\right)^\top G^{(\ell-1)} J^{(\ell)},
\qquad
\eta^{(\ell)} := \|G^{(\ell)}\|_2,
\qquad
\eta^{(\ell)} \le \chi_1\,\eta^{(\ell-1)}.
\]
with \(\chi_1 \le 1\) in the ordered/critical regime (`src/core/fisher/fisher_metric.py`, `src/core/correlation/two_point.py`).

The H1 hypothesis (scale correspondence) empirical result is documented as
\[
\xi(k)=\xi_0\exp\!\left(-\frac{k}{k_c}\right), \qquad R^2 = 0.997\pm 0.001,
\]
with repository-level claim in `README.md` and table mappings in `docs/PAPER_CODE_CORRESPONDENCE.md`.

## Mathematical constants extracted from code

| Constant | Value (code-grounded) | Extraction path |
|---|---:|---|
| \(\chi_1\) | \(0.894\) (default asymptotic contraction) | `src/core/rg_flow_solver.py` (`RGFlowSolver.__init__(chi_infty=0.894)`) |
| \(\xi_{\text{depth}}\) (formula) | \(-1/\log(\chi_1)\) | `src/core/rg_flow_solver.py` (`self.k_c = -1/log(chi_infty)`) |
| \(\xi_{\text{depth}}\) at \(\chi_1=0.894\) | \(8.925\) | computed from code formula |
| \(\sigma_w\) | \(1.4\) (critical-theorem setting) | `src/proofs/theorem1_fisher_transform.py` (`sigma_w = 1.4`) |
| \(\sigma_b\) | \(0.3\) | `config/architectures/rg_net.yaml` |
| \(\alpha\) (H2 depth-scaling log-law slope) | fitted OLS coefficient in \(L_{\min}=\alpha\log(\xi_{\text{data}}/\xi_{\text{target}})+\beta\) | `experiments/h2_depth_scaling/statistical_analysis.py` (`fit_log_scaling`) |
| \(\alpha\) reference value | \(1.0\) (null used in `test_alpha_equals_one`) | `experiments/h2_depth_scaling/statistical_analysis.py` |
| \(\epsilon_0\) | \(1-\chi_1\) | `src/core/rg_flow_solver.py` (`BetaFunctionSolver.eps_0`) |
| \(\epsilon_0\) at \(\chi_1=0.894\) | \(0.106\) | computed from code formula |

## RG operator stack (Mermaid)

```mermaid
flowchart TB
    IN[Input x in R^d_in] --> EMB[Embedding Linear]
    EMB --> SEL{Operator family}

    SEL --> STD[StandardRGOperator\nh^l = phi(W^l h^(l-1)+b^l)]
    SEL --> RES[ResidualRGOperator\nh^l = Standard(h^(l-1))+P h^(l-1)]
    SEL --> ATT[AttentionRGOperator\nMHA + FFN + LayerNorm + residual]
    SEL --> WAV[WaveletRGOperator\nHaar split lo/hi, mix, tanh-combine]

    STD --> JAC[Jacobian assembly J^(l)]
    RES --> JAC
    ATT --> JAC
    WAV --> JAC

    JAC --> FISH[Fisher pullback\nG^(l)=J^(l)^T G^(l-1) J^(l)]
    FISH --> ETA[Metric norm eta^(l)=||G^(l)||_2]
    ETA --> CONTR[Contraction regime\neta^(l) <= chi_1 eta^(l-1)]
    CONTR --> CORR[Correlation flow\nxi(k)=xi_0 exp(-k/k_c)]
    CORR --> DEPTH[Depth law\nL_min=xi_depth log(xi_data/xi_target)]
    DEPTH --> OUT[Classifier head / logits]

    subgraph Variants in src/architectures/rg_net/rg_net.py
      V1[RGNetStandard]
      V2[RGNetDeep\nperiodic skip]
      V3[RGNetUltraDeep\ncheckpoint segments]
      V4[RGNetMultiScale\nquarter-depth state fusion]
    end

    V1 --> SEL
    V2 --> SEL
    V3 --> SEL
    V4 --> SEL
```

## Navigation

- [Theory: formal derivations for Theorems 1-3](Theory)
- [Fisher Geometry: contraction mechanics](Fisher_Geometry)
- [Benchmarks: RG-Net vs ResNet/DenseNet effect sizes](Benchmarks)
