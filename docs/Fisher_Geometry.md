---
layout: default
title: Fisher Geometry
parent: Master Thesis
nav_order: 3
---

# Fisher metric contraction mechanics

Code anchors:
- `src/core/fisher/fisher_metric.py`
- `src/core/rg_flow_solver.py`
- `src/core/correlation/two_point.py`

## Pullback recurrence

At layer \(\ell\):
\[
G^{(\ell)} = \left(J^{(\ell)}\right)^\top G^{(\ell-1)}J^{(\ell)}.
\]
With \(G^{(0)}=I\), positive semidefinite structure is preserved. The implementation applies eigenvalue clipping to
\[
\lambda_i(G^{(\ell)}) \leftarrow \max\left(\lambda_i(G^{(\ell)}),10^{-10}\right)
\]
for numerical stability.

## Contraction inequality

Define spectral scalar
\[
\eta^{(\ell)} := \|G^{(\ell)}\|_2.
\]
Given
\[
\|J^{(\ell)}\|_2^2 \le \chi_1,
\]
we obtain
\[
\eta^{(\ell)} \le \eta^{(\ell-1)}(1-\epsilon_0),
\quad 1-\epsilon_0 = \chi_1,
\quad \epsilon_0 = 1-\chi_1.
\]
This exact parameterization is mirrored in `BetaFunctionSolver`:
\[
\epsilon_0 = \max(1-\chi_1,0).
\]
For default \(\chi_1=0.894\),
\[
\epsilon_0 = 0.106,
\qquad
\xi_{\text{depth}} = -1/\log(0.894) \approx 8.925.
\]

## Mean-field estimate of \(\chi_1\)

The code computes
\[
\chi_1 = \sigma_w^2\int Dz\,[\phi'(\sqrt{q_*}z)]^2
\]
with Gauss-Hermite quadrature (`chi1_gauss_hermite`). Fixed-point variance \(q_*\) is solved iteratively by
\[
q_{t+1}=\sigma_w^2\,\mathbb{E}[\phi(\sqrt{q_t}z)^2]+\sigma_b^2.
\]

## Lyapunov-spectrum consistency

`src/core/lyapunov/lyapunov.py` estimates exponents from Jacobian products using QR re-orthogonalization. Ordered/critical behavior corresponds to
\[
\lambda_{\max} \le 0,
\]
which is consistent with geometric contraction and non-expansive pullback iteration.

## Practical implication for architecture depth

Contraction that is too weak (\(\chi_1\to1^-\)) implies very large \(\xi_{\text{depth}}\), i.e., slow coarse-graining and larger required depth. Stronger contraction (smaller \(\chi_1\)) accelerates scale collapse but can over-compress representations. Repository defaults sit near the edge where theorem checks pass and depth scaling remains logarithmic.
