# Installation Guide

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.9.18 | Tested; 3.10 also works |
| PyTorch | 2.0.1 | See below for GPU/CPU variants |
| CUDA | 11.8 | Optional (GPU only) |
| RAM | 8 GB | Minimum |
| Disk | 20 GB | For datasets + checkpoints |

## Step 1: Clone

```bash
git clone https://github.com/robinbishtt/rgp-neural-architectures
cd rgp-neural-architectures
```

## Step 2: Install PyTorch

**GPU (CUDA 11.8):**
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

**CPU-only (reviewers, MacBook):**
```bash
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon (MPS):**
```bash
pip install torch==2.0.1 torchvision==0.15.2
# MPS auto-detected by DeviceManager
```

## Step 3: Install Package

```bash
pip install -r requirements.txt
pip install -e ".[dev]"    # editable install with dev dependencies
```

## Step 4: Verify Installation

```bash
make verify_pipeline
# Expected output: 7/7 checks passed
```

## Step 5: Run Fast-Track

```bash
make reproduce_fast
# Expected: ~5 minutes, all outputs tagged [FAST_TRACK_UNVERIFIED]
```

## Docker (Recommended for Full Reproduction)

```bash
# GPU
docker build -t rgp:latest -f containers/Dockerfile .
docker run --gpus all -v $(pwd)/results:/workspace/results rgp:latest make reproduce_all

# CPU
docker build -t rgp:cpu -f containers/Dockerfile.cpu .
docker run -v $(pwd)/results:/workspace/results rgp:cpu make reproduce_fast
```

## Common Issues

**`ModuleNotFoundError: No module named src`**
Solution: `pip install -e .` from the repo root.

**`CUDA out of memory`**
Solution: Use `--fast-track` mode, or reduce batch size in `config/training/base_training.yaml`.

**`torch.use_deterministic_algorithms error`**
Solution: Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable.

## Conda Installation

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate rgp-neural-architectures

# Or manually:
conda create -n rgp python=3.10
conda activate rgp
conda install pytorch numpy scipy matplotlib -c pytorch -c conda-forge
pip install -e .
```
