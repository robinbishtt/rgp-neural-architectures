#!/usr/bin/env bash
# =============================================================================
# scripts/setup_environment.sh
#
# Initial environment setup: installs dependencies, validates Python version,
# creates required directory structure, and verifies GPU availability.
#
# Usage: bash scripts/setup_environment.sh [--cpu-only]
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CPU_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only) CPU_ONLY=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== RGP Neural Architectures - Environment Setup ==="
cd "$PROJECT_ROOT"

# Python version check
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if (sys.version_info.major==3 and sys.version_info.minor>=9) else 1)"; then
    echo "Python $PYTHON_VERSION - OK"
else
    echo "ERROR: Python 3.9+ required. Found: $PYTHON_VERSION" >&2
    exit 1
fi

# Install PyTorch
if [[ "$CPU_ONLY" == true ]]; then
    echo "Installing PyTorch (CPU only)..."
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch (CUDA 11.8)..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 || \
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu
fi

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install project in editable mode
echo "Installing rgp-neural-architectures in editable mode..."
pip install -e ".[dev]"

# Create required directories
echo "Creating directory structure..."
mkdir -p results/h1 results/h2 results/h3 results/fss results/jacobian
mkdir -p figures/out
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

# Environment validation
echo ""
echo "=== Validating environment ==="
python3 -c "
import torch, numpy, scipy, sympy, matplotlib, hydra
print(f'torch      : {torch.__version__}')
print(f'numpy      : {numpy.__version__}')
print(f'scipy      : {scipy.__version__}')
print(f'sympy      : {sympy.__version__}')
print(f'matplotlib : {matplotlib.__version__}')
print(f'CUDA avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU        : {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Setup complete. Run 'make verify_pipeline' to validate."
