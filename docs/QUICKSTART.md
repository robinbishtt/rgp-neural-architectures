# Quickstart Guide

> Part of the [RGP documentation set](README.md).

This guide takes you from a fresh clone to a running pipeline verification in under ten minutes. It covers three paths: a CPU-only fast-track for reviewers, a GPU install for researchers, and a container-based setup for HPC environments.

---

## Prerequisites

Confirm you have the following before starting.

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.9 | `python3 --version` |
| pip | 22.0 | `pip --version` |
| git | 2.30 | `git --version` |
| CUDA (optional) | 11.8 | `nvcc --version` |
| Docker (optional) | 20.10 | `docker --version` |

---

## Step 1  Clone the Repository

```bash
git clone https://anonymous.4open.science/r/rgp-neural-architectures-BB30
cd rgp-neural-architectures
```

---

## Step 2  Install Dependencies

Choose the installation path that matches your hardware.

### CPU-Only (Reviewers and MacBook Users)

This path works on any machine without GPU hardware, including Apple Silicon. The fast-track pipeline will complete in three to five minutes.

```bash
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ".[dev]"
```

### GPU (CUDA 11.8)

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Automated Setup (Creates Directory Structure + Validates)

If you prefer a single command that handles PyTorch installation, dependency installation, directory creation (`results/`, `checkpoints/`, `logs/`, `data/`, `figures/out/`), and environment validation:

```bash
bash scripts/setup_environment.sh           # GPU
bash scripts/setup_environment.sh --cpu-only  # CPU
```

---

## Step 3  Verify the Pipeline (< 60 Seconds)

Run the smoke test before anything else. This validates seven properties: all imports resolve, device detection works, the seed registry initializes correctly, a model forward and backward pass completes without NaN, a checkpoint round-trip is bit-exact, the Marchenko-Pastur distribution fits synthetic eigenvalue data, and the exponential decay fitter converges on synthetic correlation lengths.

```bash
make verify_pipeline
# or equivalently:
python3 scripts/verify_pipeline.py
```

Expected output:

```
=== Pipeline Verification ===

  [PASS] Core imports              torch=2.0.1, numpy=1.24.3  (0.3s)
  [PASS] Device manager            device=cpu, name=cpu  (0.1s)
  [PASS] Seed registry             master_seed=42, worker_seed_0=...  (0.1s)
  [PASS] Forward/backward pass     loss=1.3842, grad_norm=0.2177  (0.4s)
  [PASS] Checkpoint round-trip     checkpoint save/load round-trip passed  (0.2s)
  [PASS] Spectral (MP law)         MP KS p-value=0.412 (should be >0.05)  (0.8s)
  [PASS] Correlation length        ξ₀=5.01, k_c=4.03, R²=0.997  (0.3s)

  7/7 checks passed in 2.2s

All checks passed. Pipeline is functional.
```

If any check fails, see the [Troubleshooting](#troubleshooting) section below.

---

## Step 4  Run the Fast-Track Pipeline (3–5 Minutes)

The fast-track pipeline runs all three hypotheses at reduced scale  depth=10, width=64, 2 epochs  using synthetic data. All outputs are tagged `[FAST_TRACK_UNVERIFIED]` and are intended for pipeline verification only, not for scientific conclusions.

```bash
# All three hypotheses
make reproduce_fast

# Individual hypotheses
make reproduce_fast_h1    # H1 Scale Correspondence
make reproduce_fast_h2    # H2 Depth Scaling Law
make reproduce_fast_h3    # H3 Multi-Scale Generalization
```

Outputs are written to `results/h1/`, `results/h2/`, `results/h3/`, and figures to `figures/out/`.

---

## Step 5  Run Full Experiments (GPU Required)

Full experiments require a GPU with at least 24 GB VRAM for the standard configuration (L=100). See the [hardware requirements table](../README.md#hardware-requirements) in the main README for the full breakdown.

```bash
# Complete pipeline - trains all experiments and generates all figures
make reproduce_all

# Individual experiments
make reproduce_h1    # 4–6 hours on RTX 3090
make reproduce_h2    # 24–36 hours on RTX 3090
make reproduce_h3    # 6–8 hours on RTX 3090
```

To generate figures from pre-trained checkpoints without re-training, download the official checkpoints first:

```bash
bash scripts/download_pretrained_checkpoints.sh
make reproduce_figures_from_checkpoints
```

---

## Container-Based Setup

If you prefer an isolated environment or are running on an HPC cluster, use the provided container definitions.

### Docker (Local Development)

```bash
# Build the GPU image
docker build -t rgp-neural:latest -f containers/Dockerfile .

# Run the fast-track pipeline
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/results:/workspace/results \
    rgp-neural:latest make reproduce_fast

# Launch Jupyter Lab at localhost:8888 (token: rgp2026)
docker-compose up jupyter
```

### Singularity / Apptainer (HPC Clusters)

```bash
# Build the Singularity image (requires apptainer or singularity)
bash containers/singularity_build.sh --fakeroot

# Run the fast-track pipeline with GPU
singularity run --nv rgp-neural.sif make reproduce_fast

# Submit to SLURM (example)
sbatch --gpus=1 --wrap="singularity run --nv rgp-neural.sif make reproduce_h1"
```

---

## Makefile Reference

The `Makefile` is the primary interface for all common operations.

| Target | Description | Estimated Time |
|---|---|---|
| `make verify_pipeline` | Smoke test  7 checks | < 1 min |
| `make reproduce_fast` | Fast-track all hypotheses + figures | 3–5 min |
| `make reproduce_fast_h1` | Fast-track H1 only | 1–2 min |
| `make reproduce_fast_h2` | Fast-track H2 only | 1–2 min |
| `make reproduce_fast_h3` | Fast-track H3 only | 1–2 min |
| `make reproduce_h1` | Full H1 experiments | 4–6 hours |
| `make reproduce_h2` | Full H2 experiments | 24–36 hours |
| `make reproduce_h3` | Full H3 experiments | 6–8 hours |
| `make reproduce_all` | Complete pipeline | 24–72 hours |
| `make reproduce_figures` | Figures from saved results | 10–15 min |
| `make test` | Full test suite | 30–60 min |
| `make test_unit` | Unit tests only (no GPU) | 2–5 min |
| `make validate` | Determinism + hypothesis validation | 10–20 min |
| `make lint` | flake8 + isort check | < 1 min |
| `make format` | black + isort | < 1 min |
| `make clean` | Remove results, checkpoints, logs | instant |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**

The package was not installed in editable mode. Run `pip install -e ".[dev]"` from the project root directory.

**`CUDA out of memory`**

The configured depth exceeds your GPU VRAM. Either switch to the fast-track configuration (`make reproduce_fast`) or adjust `config/experiments/` to reduce depth or enable gradient checkpointing.

**`DataIntegrityError: checksum mismatch`**

The dataset was generated with different parameters or a different random seed than the official version. Delete the existing dataset directory and re-run the experiment from scratch, or download the official datasets via `bash scripts/download_pretrained_checkpoints.sh`.

**`CUBLAS_WORKSPACE_CONFIG not set`**

Add `export CUBLAS_WORKSPACE_CONFIG=:4096:8` to your shell profile. This environment variable is required for deterministic CUDA operations and is set automatically inside Docker and Singularity containers.

**Figures are blank or contain no data**

Fast-track figures are placeholders generated from synthetic data. They confirm the figure generation pipeline works but do not reflect experimental results. Run `make reproduce_all` and then `make reproduce_figures` to generate figures from real data.

---

## Next Steps

Once the pipeline is verified and experiments are running, the following documents provide deeper reference material.

- [ARCHITECTURE.md](ARCHITECTURE.md)  full six-tier design rationale and module descriptions.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)  detailed explanation of every determinism and integrity mechanism.
- [DATASETS.md](DATASETS.md)  dataset generation parameters, correlation structure, and provenance checksums.
- [MODULES.md](MODULES.md)  API reference for all public classes and functions.
 
