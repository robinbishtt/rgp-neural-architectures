---
layout: default
title: Theory
parent: Master Thesis
nav_order: 2
---

# Theorem-level derivations

## Theorem 1 (metric pullback contraction)

Code anchor: `src/core/fisher/fisher_metric.py`, `src/proofs/theorem1_fisher_transform.py`.

For a layer map \(h^{(\ell)} = f^{(\ell)}(h^{(\ell-1)})\) with Jacobian
\[
J^{(\ell)} = \frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} \in \mathbb{R}^{d_\ell\times d_{\ell-1}},
\]
the pullback of a metric tensor \(G^{(\ell-1)}\) is
\[
G^{(\ell)} = \left(J^{(\ell)}\right)^\top G^{(\ell-1)}J^{(\ell)}.
\]
Define \(\eta^{(\ell)}=\|G^{(\ell)}\|_2\). Submultiplicativity yields
\[
\eta^{(\ell)} \le \|J^{(\ell)}\|_2^2\,\eta^{(\ell-1)}.
\]
Mean-field initialization sets
\[
\chi_1 := \sigma_w^2\,\mathbb{E}[\phi'(z)^2],
\]
implemented via Gauss-Hermite quadrature in `src/core/correlation/two_point.py`. Ordered/critical phase imposes \(\chi_1\le 1\), hence
\[
\eta^{(\ell)} \le \chi_1\eta^{(\ell-1)} = (1-\epsilon_0)\eta^{(\ell-1)}, \quad \epsilon_0 := 1-\chi_1.
\]

`verify_pushforward_numerically()` in `src/proofs/theorem1_fisher_transform.py` enforces contraction empirically (e.g., \(\eta_{50}<0.5\), negative top Lyapunov estimate).

## Theorem 2 (exponential two-point decay)

Code anchor: `src/proofs/theorem2_exponential_decay.py`.

Repository recurrence:
\[
c^{(\ell+1)} = \chi_1 c^{(\ell)},
\]
with closed form
\[
c^{(\ell)} = c^{(0)}\chi_1^{\ell} = c^{(0)}\exp\!\left(-\frac{\ell}{k_c}\right),
\qquad k_c = -\frac{1}{\log\chi_1}.
\]
The proof script fits the exponential form and validates
\[
\frac{|k_c^{\text{fit}}-k_c^{\text{analytic}}|}{k_c^{\text{analytic}}}<0.05.
\]

## Theorem 3 (logarithmic depth law)

Code anchor: `src/proofs/theorem3_depth_scaling.py`.

Assume target coarse-graining threshold \(\xi_{\text{target}}\) and initial scale \(\xi_{\text{data}}\). Under exponential shrinkage
\[
\xi(\ell)=\xi_{\text{data}}\exp\!\left(-\frac{\ell}{k_c}\right),
\]
minimum depth \(L_{\min}\) solving \(\xi(L_{\min})=\xi_{\text{target}}\) is
\[
L_{\min} = k_c\log\!\left(\frac{\xi_{\text{data}}}{\xi_{\text{target}}}\right)
=\xi_{\text{depth}}\log\!\left(\frac{\xi_{\text{data}}}{\xi_{\text{target}}}\right),
\quad \xi_{\text{depth}}:=k_c.
\]
The script `verify_logarithmic_scaling` performs nonlinear fit and requires \(R^2\ge 0.95\).

## Coupling of Theorems 1-3


a) Theorem 1 controls local information geometry via \(G^{(\ell)}\) contraction.

b) Theorem 2 upgrades local contraction to global correlation-length decay.

c) Theorem 3 inverts decay to produce a required depth lower bound.

The chain is algorithmically encoded by `src/core/rg_flow_solver.py`:
\[
\beta_\xi(\xi) = -\frac{\xi}{k_c} + \frac{c_1}{N\xi},
\quad c_1=\frac{\sigma_w^2}{2},
\]
with finite-width correction term \(\propto 1/(N\xi)\).
