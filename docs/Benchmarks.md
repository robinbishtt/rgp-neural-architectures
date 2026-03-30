---
layout: default
title: Benchmarks
parent: Master Thesis
nav_order: 4
---

# RG-Net benchmark comparison

## H3 comparative matrix (repository validation fixture)

Source: `tests/validation/test_hypothesis_h3.py`.

| Architecture | ID Accuracy (%) | OOD Accuracy (%) | Std (%) | OOD Gap (%) |
|---|---:|---:|---:|---:|
| RG-Net | 86.4 | 78.9 | 1.2 | 7.5 |
| ResNet-50 | 78.6 | 65.3 | 1.5 | 13.3 |
| DenseNet-121 | 80.2 | 67.8 | 1.4 | 12.4 |

RG-Net has the smallest ID→OOD degradation in the provided benchmark fixture.

## Cohen's \(d\) protocol

The repository effect-size function is
\[
d = \frac{\bar{x}_{\text{RG}}-\bar{x}_{\text{BL}}}{s_p},
\qquad
s_p=\sqrt{\frac{(n_{RG}-1)s_{RG}^2+(n_{BL}-1)s_{BL}^2}{n_{RG}+n_{BL}-2}}.
\]
The test suite enforces:
- generic baseline comparisons: \(d>0.5\),
- RG-Net vs ResNet-50 specific check: \(d>1.0\),
- manuscript reference: \(d\approx1.8\).

## Approximate effect sizes from fixture means/stds

Using a simplified pooled-standard-deviation approximation from listed std values (equal-sample-size approximation for quick fixture-scale interpretation):
\[
d_{\text{RG,ResNet}} \approx \frac{78.9-65.3}{\sqrt{(1.2^2+1.5^2)/2}} \approx 10.0,
\]
\[
d_{\text{RG,DenseNet}} \approx \frac{78.9-67.8}{\sqrt{(1.2^2+1.4^2)/2}} \approx 8.5.
\]
These are deterministic fixture-level magnitudes (not final manuscript inferential estimates), but they are directionally consistent with the “very large effect” regime and with hypothesis tests coded in the repository.

## H1 and H2 context for benchmarking claims

- H1 (scale correspondence): `README.md` and `docs/PAPER_CODE_CORRESPONDENCE.md` report \(R^2=0.997\pm0.001\) for exponential \(\xi(k)\) decay.
- H2 (depth scaling): `experiments/h2_depth_scaling/statistical_analysis.py` fits
\[
L_{\min}=\alpha\log\left(\frac{\xi_{\text{data}}}{\xi_{\text{target}}}\right)+\beta,
\]
reports \(R^2\), standard errors, bootstrap CIs, and null test \(H_0:\alpha=1\).

Together with H3, the benchmarks operationalize the thesis that RG-consistent depth and operator design improves multiscale out-of-distribution transfer relative to conventional baselines.
