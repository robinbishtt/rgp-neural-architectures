#!/bin/bash
# =============================================================================
# RGP Neural Architectures — Singularity Build Script
# Renormalization-Group Principles for Deep Neural Networks
#
# USAGE:
#   ./singularity_build.sh [options]
#
# OPTIONS:
#   --cpu-only    Build CPU-only variant
#   --fakeroot    Use fakeroot (no sudo required)
#   --remote      Build on Singularity remote builder
#   --clean       Remove existing .sif files before build
#   --test        Run tests after build
#   --help        Show this help message
#
# EXAMPLES:
#   # Build with sudo (local)
#   ./singularity_build.sh
#
#   # Build without sudo (HPC)
#   ./singularity_build.sh --fakeroot
#
#   # Build CPU-only variant
#   ./singularity_build.sh --cpu-only --fakeroot
#
#   # Clean build with tests
#   ./singularity_build.sh --clean --test
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
CONTAINER_NAME="rgp-neural.sif"
CPU_CONTAINER_NAME="rgp-neural-cpu.sif"
DEF_FILE="${SCRIPT_DIR}/Singularity.def"
CPU_DEF_FILE="${SCRIPT_DIR}/Singularity.cpu.def"

# Build options
BUILD_CPU=false
USE_FAKEROOT=false
USE_REMOTE=false
CLEAN=false
RUN_TESTS=false

# ─────────────────────────────────────────────────────────────────────────────
# Parse Arguments
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            BUILD_CPU=true
            shift
            ;;
        --fakeroot)
            USE_FAKEROOT=true
            shift
            ;;
        --remote)
            USE_REMOTE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --help)
            echo "RGP Neural Architectures — Singularity Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --cpu-only    Build CPU-only variant"
            echo "  --fakeroot    Use fakeroot (no sudo required)"
            echo "  --remote      Build on Singularity remote builder"
            echo "  --clean       Remove existing .sif files before build"
            echo "  --test        Run tests after build"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Build with sudo"
            echo "  $0 --fakeroot               # Build without sudo"
            echo "  $0 --cpu-only --fakeroot    # Build CPU-only variant"
            echo "  $0 --clean --test           # Clean build with tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Determine Build Parameters
# ─────────────────────────────────────────────────────────────────────────────
if [ "$BUILD_CPU" = true ]; then
    CONTAINER_NAME="$CPU_CONTAINER_NAME"
    DEF_FILE="$CPU_DEF_FILE"
    echo "Building CPU-only container..."
else
    echo "Building GPU-enabled container..."
fi

CONTAINER_PATH="${SCRIPT_DIR}/${CONTAINER_NAME}"

# ─────────────────────────────────────────────────────────────────────────────
# Clean Existing Container
# ─────────────────────────────────────────────────────────────────────────────
if [ "$CLEAN" = true ] && [ -f "$CONTAINER_PATH" ]; then
    echo "Removing existing container: $CONTAINER_PATH"
    rm -f "$CONTAINER_PATH"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Check Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Checking Prerequisites ==="

# Check Singularity/Apptainer
if command -v apptainer &> /dev/null; then
    SINGULARITY_CMD="apptainer"
    echo "Using Apptainer"
elif command -v singularity &> /dev/null; then
    SINGULARITY_CMD="singularity"
    echo "Using Singularity"
else
    echo "ERROR: Neither Singularity nor Apptainer found!"
    echo "Please install Singularity >= 3.8 or Apptainer >= 1.0"
    exit 1
fi

# Check version
VERSION=$($SINGULARITY_CMD --version | grep -oP '\d+\.\d+' | head -1)
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "WARNING: Singularity version $VERSION < $REQUIRED_VERSION"
    echo "Some features may not work correctly"
fi

# Check definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

echo "Definition file: $DEF_FILE"
echo "Output: $CONTAINER_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# Build Container
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Building Container ==="
echo "Start time: $(date)"
echo ""

# Build command construction
BUILD_ARGS=()

if [ "$USE_FAKEROOT" = true ]; then
    BUILD_ARGS+=("--fakeroot")
    echo "Using fakeroot (no sudo required)"
elif [ "$USE_REMOTE" = true ]; then
    BUILD_ARGS+=("--remote")
    echo "Using remote builder"
else
    echo "Using sudo (local build)"
fi

# Execute build
echo "Running: $SINGULARITY_CMD build ${BUILD_ARGS[*]} $CONTAINER_PATH $DEF_FILE"
echo ""

if [ "$USE_FAKEROOT" = false ] && [ "$USE_REMOTE" = false ]; then
    # Local build requires sudo
    sudo $SINGULARITY_CMD build "${BUILD_ARGS[@]}" "$CONTAINER_PATH" "$DEF_FILE"
else
    $SINGULARITY_CMD build "${BUILD_ARGS[@]}" "$CONTAINER_PATH" "$DEF_FILE"
fi

BUILD_EXIT_CODE=$?

# ─────────────────────────────────────────────────────────────────────────────
# Build Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Build Summary ==="
echo "End time: $(date)"
echo "Exit code: $BUILD_EXIT_CODE"

if [ $BUILD_EXIT_CODE -ne 0 ]; then
    echo "✗ Build FAILED"
    exit $BUILD_EXIT_CODE
fi

if [ -f "$CONTAINER_PATH" ]; then
    CONTAINER_SIZE=$(du -h "$CONTAINER_PATH" | cut -f1)
    echo "✓ Build SUCCESSFUL"
    echo "Container: $CONTAINER_PATH"
    echo "Size: $CONTAINER_SIZE"
else
    echo "✗ Build failed - container not found"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────────────────────────
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "=== Running Container Tests ==="

    # Basic test
    echo "[1/3] Container test..."
    $SINGULARITY_CMD test "$CONTAINER_PATH"

    # Python version check
    echo ""
    echo "[2/3] Python environment..."
    $SINGULARITY_CMD exec "$CONTAINER_PATH" python --version

    # PyTorch check
    echo ""
    echo "[3/3] PyTorch installation..."
    if [ "$BUILD_CPU" = true ]; then
        $SINGULARITY_CMD exec "$CONTAINER_PATH" python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    else
        $SINGULARITY_CMD exec --nv "$CONTAINER_PATH" python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
    fi

    echo ""
    echo "✓ All tests passed!"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Next Steps
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Build Complete!                                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Container location: $CONTAINER_PATH"
echo ""
echo "Quick test:"
if [ "$BUILD_CPU" = true ]; then
    echo "  singularity run $CONTAINER_PATH make verify_pipeline"
else
    echo "  singularity run --nv $CONTAINER_PATH make verify_pipeline"
fi
echo ""
echo "For HPC deployment, see: docs/HPC_GUIDE.md"
echo ""
