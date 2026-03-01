# Installation Guide

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | 11.8 (optional) | 12.1 |
| RAM | 16 GB | 32 GB+ |
| GPU VRAM | 8 GB (L≤50 only) | 24 GB (L≤100) |

## Standard Installation

```bash
# 1. Clone the repository
git clone https://github.com/robinbisht/rgp-neural-architectures
cd rgp-neural-architectures

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .

# 5. Verify the installation
make verify_pipeline
```

## Docker Installation (Recommended for Reviewers)

```bash
# GPU-enabled container
docker build -t rgp-neural:latest .
docker run --gpus all -v $(pwd)/data:/data rgp-neural:latest make reproduce_fast

# CPU-only container
docker build -f containers/Dockerfile.cpu -t rgp-neural:cpu .
docker run -v $(pwd)/data:/data rgp-neural:cpu make reproduce_fast
```

## Singularity Installation (HPC Clusters)

```bash
# Build the Singularity image
singularity build rgp-neural.sif containers/Singularity.def

# Run fast-track verification on HPC with GPU
singularity run --nv rgp-neural.sif make reproduce_fast
```

## Environment Setup Script

```bash
# Automated setup including directory creation and validation
bash scripts/setup_environment.sh
```

## Verifying the Installation

After installation, run the following in order of increasing compute time:

```bash
make verify_pipeline       # <1 minute: smoke test
make reproduce_fast        # 3–5 minutes: full fast-track
make validate              # ~30 minutes: full test suite
```

## Common Installation Issues

**ImportError: torch not found**
Ensure PyTorch is installed for your CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA out of memory at L=1000**
Reduce depth or enable gradient checkpointing:
```bash
python experiments/h2_depth_scaling/run_h2_validation.py +training=gradient_checkpoint
```

**Determinism test fails on different machines**
This is expected for GPU execution. Use CPU mode for bit-exact verification:
```bash
CUDA_VISIBLE_DEVICES="" make validate
```

## Pre-trained Checkpoints

Download pre-trained model checkpoints to reproduce figures without retraining:

```bash
bash scripts/download_pretrained_checkpoints.sh
make reproduce_figures_from_checkpoints   # 10–15 minutes
```
