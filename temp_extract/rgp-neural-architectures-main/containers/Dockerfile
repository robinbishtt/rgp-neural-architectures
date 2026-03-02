# =============================================================================
# RGP Neural Architectures — GPU Docker Image
# Base: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#
# Build:
#   docker build -t rgp-neural:latest -f containers/Dockerfile .
#
# Run (fast-track, GPU):
#   docker run --gpus all -v $(pwd)/data:/workspace/data \
#              -v $(pwd)/results:/workspace/results \
#              rgp-neural:latest make reproduce_fast
#
# Run (full experiments, GPU):
#   docker run --gpus all \
#              -v $(pwd)/data:/workspace/data \
#              -v $(pwd)/results:/workspace/results \
#              -v $(pwd)/figures/out:/workspace/figures/out \
#              rgp-neural:latest make reproduce_all
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="RGP Neural Architectures"
LABEL description="Reproducibility container for RGP Neural Architectures"
LABEL cuda="11.8.0"
LABEL cudnn="8"
LABEL python="3.9.18"

# ---------------------------------------------------------------------------
# System environment
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        libhdf5-dev \
        libhdf5-serial-dev \
        hdf5-tools \
        libffi-dev \
        libssl-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        pkg-config \
        cmake \
        ninja-build \
        rsync \
        unzip \
        vim \
        htop \
        tree \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python 3.9.18 (deadsnakes PPA for exact version pinning)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        python3.9-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.9

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------------------------
# PyTorch (CUDA 11.8) — installed before requirements to leverage layer cache
# ---------------------------------------------------------------------------
RUN pip install \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------------------------
# Python dependencies (pinned via requirements.txt)
# ---------------------------------------------------------------------------
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# ---------------------------------------------------------------------------
# Copy source code
# ---------------------------------------------------------------------------
COPY . /workspace/

# ---------------------------------------------------------------------------
# Install package in editable mode
# ---------------------------------------------------------------------------
RUN pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# CUDA / cuDNN determinism settings
# ---------------------------------------------------------------------------
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDNN_V8_API_ENABLED=1

# ---------------------------------------------------------------------------
# Create result directories
# ---------------------------------------------------------------------------
RUN mkdir -p /workspace/results/h1 \
             /workspace/results/h2 \
             /workspace/results/h3 \
             /workspace/results/fss \
             /workspace/results/jacobian \
             /workspace/figures/out \
             /workspace/checkpoints \
             /workspace/logs \
             /workspace/data

# ---------------------------------------------------------------------------
# Validate installation
# ---------------------------------------------------------------------------
RUN python -c "import torch; print('PyTorch:', torch.__version__); \
               print('CUDA available:', torch.cuda.is_available())" || true

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["make verify_pipeline"]
