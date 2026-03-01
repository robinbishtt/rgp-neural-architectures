<div align="center">

# RGP Neural Architectures

**Renormalization-Group Principles for Deep Neural Network Depth**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18822434.svg)](https://doi.org/10.5281/zenodo.18822434)
[![Tests](https://img.shields.io/badge/tests-100%2B%20files-brightgreen.svg)](#testing)
[![Docker](https://img.shields.io/badge/container-Docker%20%7C%20Singularity-2496ED.svg)](#containerization)

*A rigorous, fully reproducible framework connecting renormalization-group transformations from statistical physics to the hierarchical depth structure of deep neural networks.*

</div>

---

## Overview

This repository implements the complete theoretical and empirical framework for studying how **renormalization-group (RG) theory** the physics formalism describing scale transformations maps onto the **depth architecture** of deep neural networks. Three core hypotheses are validated:

| Hypothesis | Statement | Primary Metric |
|:---:|---|---|
| **H1** | Scale correspondence: correlation length decays as ξ(k) ∝ exp(−k / k_c) | R² ≥ 0.95, KS p-value > 0.05 |
| **H2** | Depth scaling law: L_min ~ k_c · log(ξ_data / ξ_target) | AIC selects logarithmic over linear and power-law |
| **H3** | RG-Net achieves statistically significant advantage on hierarchical data | Wilcoxon p < 0.01 vs. ResNet, DenseNet, MLP, VGG |

The codebase is organized as a six-tier architecture (≈ 250 files) covering mathematical foundations, neural architectures, a 100-file test suite, configuration management, figure generation, and a cross-cutting infrastructure layer providing determinism, telemetry, and data provenance.

---

## Repository Structure

```
rgp_neural_architectures/
│
├── src/                            Source code
│   ├── core/                         Tier 1 — Mathematical foundations
│   │   ├── fisher/                     Fisher information geometry
│   │   ├── jacobian/                   Jacobian analysis (autograd, JVP, VJP, finite-diff, symbolic)
│   │   ├── spectral/                   Random matrix theory (Marchenko-Pastur, Wigner, Tracy-Widom)
│   │   ├── correlation/                Two-point correlation functions, χ₁ via Gauss-Hermite quadrature
│   │   └── lyapunov/                   Lyapunov spectrum via Benettin QR algorithm
│   ├── rg_flow/operators/              RG operators (standard, residual, attention, wavelet, learned)
│   ├── proofs/                         Symbolic SymPy verification of Theorems 1–3
│   ├── architectures/                Tier 2 — Neural architectures
│   │   ├── rg_net/                     RG-Net (shallow / standard / deep / ultra-deep / variable-width / multi-scale)
│   │   └── baselines/                  ResNet, DenseNet, MLP, VGG baselines
│   ├── training/                       Training infrastructure
│   │   └── optimizers/                 Adam variants, SGD, second-order, layer-wise LR
│   ├── scaling/                        Finite-size scaling analysis, critical exponent extraction
│   ├── datasets/                       Hierarchical dataset generators
│   │   └── loaders/                    Deterministic, distributed, cached, streaming data loaders
│   ├── utils/                        ICL — Infrastructure Cross-Layer
│   │   ├── seed_registry.py            Global determinism controller (Singleton)
│   │   ├── device_manager.py           Hardware-agnostic CUDA / MPS / CPU detection
│   │   ├── error_handler.py            OOM, NaN, and checkpoint-resume recovery
│   │   └── fast_track_validator.py     Fast-track output validation
│   ├── telemetry/                      TensorBoard / WandB / MLflow / JSONL logging
│   ├── checkpoint/                     Distributed async checkpointing
│   ├── orchestration/                  DAG executor and Hydra configuration management
│   └── provenance/                     SHA-256 data integrity verification
│
├── experiments/                    Tier 5 — Experimental pipelines
│   ├── h1_scale_correspondence/
│   ├── h2_depth_scaling/
│   └── h3_multiscale_generalization/
│
├── tests/                          Tier 3 — 100+ test files across 8 categories
│   ├── unit/                         Core mathematics correctness
│   ├── integration/                  End-to-end pipeline tests
│   ├── stability/                    Gradient stability and critical initialization
│   ├── scaling/                      H1 / H2 / H3 quantitative checks
│   ├── ablation/                     Component removal studies
│   ├── robustness/                   Noise, adversarial, OOD evaluation
│   ├── spectral/                     RMT distribution fitting
│   └── validation/                   Determinism and reproducibility
│
├── figures/                        Figure generators
│   ├── manuscript/                   Figures 1–5
│   ├── extended_data/                Extended Data Figures 1–6
│   └── styles/                       publication.mplstyle, color palette, font config
│
├── config/                         Hydra hierarchical configuration
│   ├── experiments/                  H1 / H2 / H3 / baseline / FSS / ablation configs
│   ├── fast_track/                   Reviewer fast-track overrides (L=10, 2 epochs)
│   ├── architectures/                Model configuration variants
│   ├── training/                     Optimizer, scheduler, mixed-precision configs
│   ├── scaling/                      FSS analysis parameters
│   └── telemetry/                    Logging backend selection
│
├── containers/                     Reproducibility containers
│   ├── Dockerfile                    GPU image (CUDA 11.8 + cuDNN 8)
│   ├── Dockerfile.cpu                CPU-only image
│   ├── docker-compose.yml            Multi-service orchestration
│   └── Singularity.def               HPC Singularity / Apptainer definition
│
├── scripts/                        Automation and validation scripts (14 files)
├── notebooks/                      Exploratory analysis
├── docs/                           Extended documentation
├── Makefile                        Primary orchestration interface
├── requirements.txt                Pinned dependencies (exact versions)
└── pyproject.toml                  Package metadata
```

---

## Quick Start

### Requirements

| Component | Version |
|---|---|
| Python | 3.9.18 |
| PyTorch | 2.0.1 |
| CUDA | 11.8 *(optional)* |
| Minimum GPU VRAM | 8 GB |
| Recommended GPU VRAM | 24 GB |

### Installation

```bash
git clone https://github.com/robinbishtt/rgp-neural-architectures.git
cd rgp-neural-architectures

# GPU install (CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ".[dev]"

# CPU-only install (reviewers, MacBook, CI)
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ".[dev]"
```

For automated setup including directory creation and environment validation:

```bash
bash scripts/setup_environment.sh         # GPU
bash scripts/setup_environment.sh --cpu-only  # CPU
```

---

## Reviewer Fast-Track (≤ 5 Minutes)

Full experiments (L=1000, FSS analysis) require 48–72 hours on 80 GB VRAM. The fast-track system runs the complete pipeline at reduced scale depth=10, width=64, 2 epochs on any hardware including CPU-only machines. All outputs are tagged `[FAST_TRACK_UNVERIFIED]`.

```bash
# Smoke test — validates imports, device detection, seed registry, forward/backward pass (< 1 min)
make verify_pipeline

# Full fast-track: H1 + H2 + H3 + all figures (3–5 min, synthetic data)
make reproduce_fast

# Individual hypothesis fast-track
make reproduce_fast_h1
make reproduce_fast_h2
make reproduce_fast_h3
```

---

## Full Reproduction

```bash
# Complete pipeline — data generation, training, evaluation, figures
make reproduce_all          # 24–72 hours on RTX 3090

# Individual experiments
make reproduce_h1           # H1 Scale Correspondence         (4–6 hours)
make reproduce_h2           # H2 Depth Scaling Law             (24–36 hours)
make reproduce_h3           # H3 Multi-Scale Generalization    (6–8 hours)

# Figures and tables from saved results
make reproduce_figures
make reproduce_extended_data
make reproduce_tables

# Generate figures from pre-trained checkpoints (no training required)
bash scripts/download_pretrained_checkpoints.sh
make reproduce_figures_from_checkpoints
```

### Hardware Reference

| Configuration | VRAM | GPU Example | Suitable For |
|---|---|---|---|
| Minimal | 8 GB | RTX 3070 / GTX 1080 Ti | L = 10–50, prototyping |
| Standard | 24 GB | RTX 3090 / RTX 4090 | L = 100, main experiments |
| Extended | 48 GB | RTX A6000 / A40 | L = 500, scaling studies |
| Ultra-Deep | 80 GB | A100 / H100 | L = 1000+, full validation |

---

## Containerization

Containers freeze the exact environment OS, Python version, PyTorch, and all libraries ensuring the code runs identically in 2026 and 2028.

```bash
# Docker — GPU
docker build -t rgp-neural:latest -f containers/Dockerfile .
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/results:/workspace/results \
    rgp-neural:latest make reproduce_fast

# Docker — CPU only
docker build -t rgp-neural:cpu -f containers/Dockerfile.cpu .
docker run -v $(pwd)/results:/workspace/results \
    rgp-neural:cpu make verify_pipeline

# Singularity / Apptainer — HPC clusters (SLURM / PBS / LSF)
bash containers/singularity_build.sh --fakeroot
singularity run --nv rgp-neural.sif make reproduce_fast

# Multi-service: GPU training + Jupyter Lab + TensorBoard
docker-compose up jupyter           # Jupyter at localhost:8888 (token: rgp2026)
docker-compose up tensorboard       # TensorBoard at localhost:6006
docker-compose run gpu make reproduce_fast
```

---

## Testing

The test suite contains over 100 files organized across eight categories. Every theorem, architectural component, and pipeline stage has corresponding tests.

```bash
make test                 # Full suite
make test_unit            # Core mathematics correctness (no GPU required, fast)
make test_integration     # End-to-end pipeline validation
make test_stability       # Gradient stability and critical initialization
make test_spectral        # RMT distribution fitting (MP law, Wigner, Tracy-Widom)
make test_scaling         # H1 / H2 / H3 quantitative hypothesis checks
make validate             # Determinism verification + hypothesis validation
```

---

## Reproducibility

The following mechanisms guarantee bit-exact reproducibility across different machines, operating systems, and runs.

**Determinism.** All random number generators Python `random`, NumPy, PyTorch CPU, PyTorch CUDA are seeded from a single master seed managed by `SeedRegistry`, a thread-safe Singleton that prevents any module from calling `random.seed()` independently.

**Hardware portability.** `DeviceManager` auto-selects CUDA → MPS → CPU. No `.cuda()` calls are hardcoded anywhere in the codebase.

**Data integrity.** `DataAuditor` computes SHA-256 checksums of all generated datasets and verifies them against `src/provenance/master_hashes.py` before any training begins.

**Environment lock.** `requirements.txt` pins every dependency to an exact version. `Dockerfile` and `Singularity.def` freeze the complete OS and library stack.

```bash
# Explicitly verify determinism across three independent runs
bash scripts/validate_determinism.sh --seed 42 --n-trials 3

# Verify all three hypotheses with statistical significance tests
bash scripts/validate_hypotheses.sh
```

For a detailed explanation of every reproducibility mechanism, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Architecture Overview

```
Tier 1  Nervous System         src/core/, src/rg_flow/, src/proofs/
Tier 2  Engine Room            src/architectures/, src/training/, src/scaling/
Tier 3  Audit Bureau           tests/   (100+ files)
Tier 4  Command Center         config/, src/datasets/, containers/
Tier 5  Publication Machine    experiments/, figures/, scripts/
ICL     Infrastructure         src/utils/, src/telemetry/, src/checkpoint/,
        Cross-Layer            src/orchestration/, src/provenance/
```

For the full module-by-module design rationale, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Six-tier system design and module rationale |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Step-by-step setup and first-run guide |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Determinism, checksums, containers, CI |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Dataset generation, provenance, and checksums |
| [`docs/MODULES.md`](docs/MODULES.md) | API reference for all source modules |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |
| [`SECURITY.md`](SECURITY.md) | Security policy |

---

## License

This project is licensed under the [MIT License](LICENSE).
