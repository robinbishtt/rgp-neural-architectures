<div align="center">

# RGP Neural Architectures

## Renormalization-Group Principles for Deep Neural Architectures

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-100%2B%20files-brightgreen.svg)](#testing)
[![Notebooks](https://img.shields.io/badge/notebooks-18%20files-blue.svg)](#notebooks)

*A rigorous, fully reproducible framework connecting renormalization-group (RG) transformations
from statistical physics to the hierarchical depth structure of deep neural networks.*

**Anonymous Submission**

</div>

---

## Core Thesis

> **Depth is not merely expressive power - it is the minimal number of information-geometric
> coarse-graining steps required to collapse multi-scale data correlations into an effective theory.**

Each layer performs one RG step, contracting the Fisher metric by factor $(1-\varepsilon_0)$.
This gives an exact logarithmic depth lower bound:

$$L_{\min} = \xi_{\text{depth}} \cdot \log\!\left(\frac{\xi_{\text{data}}}{\xi_{\text{target}}}\right), \qquad \xi_{\text{depth}} = -\frac{1}{\log \chi_1}$$

---

## Key Results

### H1: Scale Correspondence (R²=0.997±0.001)

The correlation length ξ(k) decays exponentially with depth: ξ(k) = ξ₀·exp(-k/k_c).

![H1 Validation](figures/out/fig4_h1_validation.png)

### H2: Logarithmic Depth Scaling Law (α̂=0.98±0.06, p<0.001)

Minimum depth scales logarithmically with data correlation length: L_min ∝ log(ξ_data).

![H2 Scaling](figures/out/fig5_h2_scaling.png)

### H3: RG-Net Architectural Advantage (Cohen d=1.8, p=0.006)

RG-Net outperforms matched baselines on OOD tasks requiring multi-scale reasoning.

![H3 Results](figures/out/fig6_h3_comparison.png)

---

## Three Theorems

| Theorem | Statement | Code Verification |
|:---:|---|---|
| **Thm 1** | Metric contraction: η^(ℓ) ≤ η^(ℓ-1)(1-ε₀) | `src/proofs/theorem1_fisher_transform.py` |
| **Thm 2** | Exponential decay: c^(ℓ)=c^(0)·χ^ℓ | `src/proofs/theorem2_exponential_decay.py` |
| **Thm 3** | Log depth bound: L_min=ξ_depth·log(ξ_data/ξ_target) | `src/proofs/theorem3_depth_scaling.py` |

**Critical initialization**: tanh at (σ_w=1.4, σ_b=0.3) gives χ₁≈1, ξ_depth≈100.

---

## Quick Start

### Requirements

| Component | Version |
|---|---|
| Python | 3.9.18 |
| PyTorch | 2.0.1 |
| CUDA | 11.8 *(optional)* |

### Installation

```bash
git clone https://anonymous.4open.science/r/rgp-neural-architectures-BB30
cd rgp-neural-architectures

# GPU (CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt && pip install -e "[dev]"

# CPU-only (reviewers / CI)
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e "[dev]"
```

### Reviewer Fast-Track (≤ 5 Minutes, CPU-Only)

```bash
make verify_pipeline    # Smoke test: 7 checks, < 60 seconds
make reproduce_fast     # Full fast-track: H1 + H2 + H3 + figures, 3-5 min
```

All fast-track outputs are tagged `[FAST_TRACK_UNVERIFIED]`.

### Full Reproduction

```bash
make reproduce_all    # Complete pipeline: 24-72 hours on RTX 3090
make reproduce_h1     # H1 only: 4-6 hours
make reproduce_h2     # H2 only: 24-36 hours
make reproduce_h3     # H3 only: 6-8 hours
```

---

## Repository Structure

```
rgp-neural-architectures/
│
├── src/                        Source code - 137 Python modules
│   ├── core/                     Tier 1: Mathematical foundations
│   │   ├── fisher/                 Fisher information geometry (pullback g=J^T G J)
│   │   ├── jacobian/               Jacobian computation (autograd, JVP, VJP, FD)
│   │   ├── spectral/               RMT: Marchenko-Pastur, Wigner, Tracy-Widom
│   │   ├── correlation/            Two-point functions, chi1, exponential decay fit
│   │   └── lyapunov/               Lyapunov spectrum via Benettin QR algorithm
│   ├── rg_flow/operators/          Standard, Residual, Attention, Wavelet, Learned
│   ├── proofs/                     SymPy + numerical theorem verification
│   ├── architectures/            Tier 2: Neural architectures
│   │   ├── rg_net/                 7 RG-Net variants (shallow to ultra-deep)
│   │   └── baselines/              ResNet-50, DenseNet-121, Wavelet-CNN, Tensor-Net
│   ├── training/                   Trainer, optimizers, schedulers, mixed precision
│   ├── scaling/                    FSS analysis, critical exponents, AIC model selection
│   ├── datasets/                   Hierarchical datasets (Hier-1..5), OOD loaders
│   └── utils/                    ICL: SeedRegistry, DeviceManager, checkpointing
│
├── experiments/                Hypothesis validation pipelines
│   ├── h1_scale_correspondence/    H1: xi(k) exponential decay validation
│   ├── h2_depth_scaling/           H2: L_min ~ log(xi_data) validation
│   └── h3_multiscale_generalization/ H3: RG-Net vs baselines comparison
│
├── ablation/                   8 ablation study scripts
│   ├── run_activation_ablation.py       Tanh vs ReLU vs GELU
│   ├── run_initialization_ablation.py   5 init configs vs paper critical
│   ├── run_width_ablation.py            1/N corrections vs mean-field
│   ├── run_depth_ablation.py            L vs L_min threshold test
│   ├── run_operator_ablation.py         Standard vs Residual vs Wavelet vs Attention
│   ├── run_skip_connection_ablation.py  Skip interval effect
│   ├── run_off_critical_ablation.py     H1 falsification at off-critical init
│   └── run_all_ablations.py             Master runner
│
├── notebooks/                  21 Jupyter notebooks
│
├── tests/                      83 test files across 8 categories
│   ├── unit/          19 files - core mathematics (Fisher, Jacobian, Lyapunov, RMT)
│   ├── stability/     11 files - critical initialization, gradient flow, weight norms
│   ├── scaling/        9 files - H1/H2/H3 quantitative, FSS, scaling law fit
│   ├── spectral/       7 files - Marchenko-Pastur, Wigner, Tracy-Widom, level spacing
│   ├── validation/     7 files - determinism, reproducibility, hypothesis checks
│   ├── integration/    7 files - checkpoint, training convergence, full pipeline
│   ├── robustness/     9 files - adversarial, OOD, noise, corruption
│   └── ablation/      14 files - per-component ablation validation
│
├── figures/                    Figure generators + 15 PNG outputs
│   ├── manuscript/               5 manuscript figures (generate_figure1.py – 5.py)
│   ├── extended_data/            11 extended data figures (run_extended_figure1.py – 11.py)
│   ├── supplementary/            Supplementary figures and tables
│   ├── styles/                   Publication stylesheet, color palette
│   └── out/                      Generated PNGs (15 figures)
│
├── config/                     Hydra hierarchical configuration
│   ├── architectures/            RG-Net configs, baseline configs
│   ├── experiments/              Per-hypothesis experiment configs
│   ├── training/                 Optimizer, scheduler, fast-track overrides
│   └── scaling/, telemetry/, datasets/
│
├── containers/                 Docker + Singularity containers, HPC job scripts
├── scripts/                    23 shell + Python automation scripts
├── docs/                       9 documentation files
│   └── API.md, ARCHITECTURE.md, DATASETS.md, HPC_GUIDE.md, INSTALLATION.md,
│       MODULES.md, PAPER_CODE_CORRESPONDENCE.md, QUICKSTART.md, REPRODUCIBILITY.md
├── .github/workflows/          6 CI workflows (Tests, Lint, Experiments,
│                               Model Validator, Notebooks, Reproduce)
├── CITATION.cff
├── LICENSE (MIT)
├── CONTRIBUTING.md
├── requirements.txt (pinned, Python 3.11)
└── pyproject.toml
```

---

## Paper–Code Correspondence

See [`docs/PAPER_CODE_CORRESPONDENCE.md`](docs/PAPER_CODE_CORRESPONDENCE.md) for a complete
mapping from every paper claim to the implementing code.
---

---

## Reproducibility

All results are bit-exact reproducible via the `SeedRegistry` singleton.

| Parameter | Value |
|---|---|
| Master seed | 42 |
| Hardware (paper) | NVIDIA RTX 3090, 24 GB VRAM |
| PyTorch | 2.0.1+cu118 |
| Python | 3.9.18 |
| CUDA | 11.8 |
| H1 runtime | 4–6 hours |
| H2 runtime | 24–36 hours |
| H3 runtime | 6–8 hours |
| Fast-track (CPU) | < 5 minutes |

```bash
# Verify bit-exact determinism across 3 runs
bash scripts/validate_determinism.sh --seed 42 --n-trials 3
```

Fast-track outputs are tagged `[FAST_TRACK_UNVERIFIED]` and run at reduced scale (depth=10, width=64, 2 epochs). For quantitative verification of paper claims, run the full pipeline.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for complete reproducibility documentation.

---

## Testing

```bash
make test              # Full suite (100+ files)
make test_unit         # Core mathematics (no GPU, fast)
make test_stability    # Critical initialization tests
make test_scaling      # H1/H2/H3 quantitative hypothesis checks
make test_spectral     # RMT distribution fitting (MP, Wigner, TW)
make validate          # Determinism + hypothesis validation
```

---

## Notebooks

18 Jupyter notebooks covering every theorem, hypothesis, and experimental component:

```bash
jupyter lab notebooks/
```

Start with:
- `notebooks/quick_start_demo.ipynb` - 5-minute end-to-end demo
- `notebooks/theorem1_metric_contraction.ipynb` - verify the pullback formula
- `notebooks/h1_scale_correspondence.ipynb` - reproduce H1 main result

---

## Containerization

```bash
# Docker GPU
docker build -t rgp:latest -f containers/Dockerfile .
docker run --gpus all -v $(pwd)/results:/workspace/results rgp:latest make reproduce_fast

# Docker CPU (reviewers)
docker build -t rgp:cpu -f containers/Dockerfile.cpu .
docker run -v $(pwd)/results:/workspace/results rgp:cpu make reproduce_fast

# Singularity (HPC)
bash containers/singularity_build.sh --fakeroot
singularity run --nv rgp-neural.sif make reproduce_fast
```

---

## Citation

```bibtex
@software{rgp_neural_architectures_2026,
  title  = {Renormalization-Group Principles for Deep Neural Architectures},
  year   = {2026},
  url    = {https://anonymous.4open.science/r/rgp-neural-architectures-BB30},
  note   = {Anonymous Submission},
}
```

---

## License

[MIT License](LICENSE) - see the [LICENSE](LICENSE) file.

Code and pre-trained configurations:
**[https://anonymous.4open.science/r/rgp-neural-architectures-BB30](https://anonymous.4open.science/r/rgp-neural-architectures-BB30)**
