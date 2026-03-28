# RGP Neural Architectures

Renormalization-Group (RG) principles for deep neural architectures, with code for theorem verification, hypothesis-driven experiments, ablations, and reproducibility workflows.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

This repository provides:

- **Theory components**: Fisher geometry, Jacobians, spectral analysis, correlation dynamics, and symbolic proof helpers.
- **Modeling components**: RG-Net variants and baseline architectures.
- **Experiment pipelines**: H1/H2/H3 validation scripts, fast-track and full reproduction paths.
- **Quality and reproducibility tooling**: deterministic seeds, validation scripts, test suites, and CI workflows.

Core depth-scaling relation used throughout the project:

\[
L_{\min} = \xi_{\text{depth}} \cdot \log\left(\frac{\xi_{\text{data}}}{\xi_{\text{target}}}\right)
\]

---

## Quick Start

### 1) Clone

```bash
git clone https://github.com/robinbishtt/rgp-neural-architectures.git
cd rgp-neural-architectures
```

### 2) Install dependencies

#### CPU-only

```bash
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ".[dev]"
```

#### CUDA 11.8

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3) Verify pipeline

```bash
make verify_pipeline
make reproduce_fast
```

---

## Common Commands

### Reproduction

```bash
make reproduce_fast      # Fast-track verification
make reproduce_fast_h1   # H1 only
make reproduce_fast_h2   # H2 only
make reproduce_fast_h3   # H3 only
make reproduce_all       # Full pipeline
```

### Testing and validation

```bash
make test
make test_unit
make test_integration
make test_stability
make test_scaling
make test_spectral
make validate
```

### Code quality

```bash
make lint
make format
make typecheck
```

---

## Repository Structure

```text
rgp-neural-architectures/
├── src/            # Core math, architectures, training, scaling, datasets, utilities
├── experiments/    # H1/H2/H3 experiment runners and analysis
├── ablation/       # Ablation study scripts
├── tests/          # Unit, integration, stability, scaling, spectral, robustness, validation
├── figures/        # Figure generation pipelines and outputs
├── config/         # Hydra configuration tree
├── scripts/        # Setup, validation, and automation scripts
├── docs/           # Extended project documentation
├── containers/     # Docker/Singularity assets for reproducible runs
└── notebooks/      # Jupyter notebooks for interactive exploration
```

---

## Documentation

Detailed documentation lives under [`docs/`](docs/README.md):

- [Documentation Index](docs/README.md)
- [Quickstart](docs/QUICKSTART.md)
- [Installation](docs/INSTALLATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Module Reference](docs/MODULES.md)
- [Datasets](docs/DATASETS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [HPC Guide](docs/HPC_GUIDE.md)
- [Paper-Code Correspondence](docs/PAPER_CODE_CORRESPONDENCE.md)

---

## Reproducibility Notes

- Fast-track outputs are generated at reduced scale for quick verification.
- Full experimental claims should be validated with full pipelines.
- Determinism and hypothesis checks are available via:

```bash
bash scripts/validate_determinism.sh
bash scripts/validate_hypotheses.sh
```

---

## Contributing

Please see:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

---

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

---

## License

This project is licensed under the [MIT License](LICENSE).
