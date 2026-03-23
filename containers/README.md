# RGP Neural Architectures  Containerization

This directory contains container definitions for reproducible execution of RGP experiments across different computing environments.

## Overview

| File | Purpose | Use Case |
|------|---------|----------|
| `Singularity.def` | GPU-enabled HPC container | Production experiments on clusters |
| `Singularity.cpu.def` | CPU-only container | Testing, CI/CD, CPU partitions |
| `Dockerfile` | GPU Docker image | Local development, cloud |
| `Dockerfile.cpu` | CPU Docker image | Lightweight testing |
| `docker-compose.yml` | Multi-service orchestration | Jupyter + TensorBoard |
| `singularity_build.sh` | Build automation script | HPC deployment |
| `hpc_jobs/` | SLURM/PBS job scripts | Batch submission |

## Quick Reference

### Singularity (HPC)
```bash
# Build (HPC - no sudo needed)
singularity build --fakeroot rgp-neural.sif Singularity.def

# Run fast-track (5 min)
singularity run --nv rgp-neural.sif make reproduce_fast

# Run full H1 experiment (4-6 hours)
singularity exec --nv rgp-neural.sif make reproduce_h1
```

### Docker (Local/Cloud)
```bash
# Build
docker build -t rgp-neural:latest -f Dockerfile .

# Run
docker run --gpus all -v $(pwd)/results:/workspace/results rgp-neural:latest make reproduce_fast
```

## Singularity for HPC

### Why Singularity?

Most university research labs and supercomputers (HPC clusters) **do not allow Docker** due to security concerns. Singularity/Apptainer is the standard for containerized scientific computing on HPC.

**Key advantages:**
- ✅ No root privileges required (with `--fakeroot`)
- ✅ Native integration with SLURM, PBS, LSF schedulers
- ✅ Direct GPU access via `--nv` flag
- ✅ Same user permissions inside and outside container
- ✅ Single-file container (.sif) easy to share

### Build Options

#### Option 1: HPC Build (Recommended - No sudo)
```bash
# On your HPC cluster
singularity build --fakeroot rgp-neural.sif Singularity.def
```

#### Option 2: Local Build (Requires sudo)
```bash
# On your local machine with Singularity installed
sudo singularity build rgp-neural.sif Singularity.def

# Copy to cluster
scp rgp-neural.sif user@cluster:/path/to/project/
```

#### Option 3: Automated Build
```bash
# Using the build script
./singularity_build.sh --fakeroot --test

# CPU-only variant
./singularity_build.sh --cpu-only --fakeroot
```

### Running on HPC

#### Interactive (Testing)
```bash
# Verify pipeline (< 1 minute)
singularity run --nv rgp-neural.sif make verify_pipeline

# Interactive shell
singularity shell --nv rgp-neural.sif
Singularity> python --version
Singularity> python -c "import torch; print(torch.__version__)"
```

#### Batch Jobs (SLURM)
```bash
# Submit pre-configured job
sbatch hpc_jobs/slurm_rgp_fast.sh    # 5 minutes
sbatch hpc_jobs/slurm_rgp_h1.sh      # 4-6 hours
sbatch hpc_jobs/slurm_rgp_h2.sh      # 24-36 hours
sbatch hpc_jobs/slurm_rgp_h3.sh      # 6-8 hours
sbatch hpc_jobs/slurm_rgp_full.sh    # 48-72 hours
```

#### Batch Jobs (PBS/Torque)
```bash
qsub hpc_jobs/pbs_rgp_fast.pbs
qsub hpc_jobs/pbs_rgp_h1.pbs
```

### Data Binding

Singularity containers are read-only by default. Use `--bind` to mount host directories:

```bash
# Basic binding
singularity run --nv \
    --bind /cluster/data:/workspace/data \
    --bind /cluster/results:/workspace/results \
    rgp-neural.sif make reproduce_h1

# Using scratch space (recommended for temp files)
singularity run --nv \
    --bind $SCRATCH:/workspace/data \
    rgp-neural.sif make reproduce_h1
```

### GPU Support

```bash
# Check GPU availability on host
nvidia-smi

# Run with GPU support (--nv flag)
singularity run --nv rgp-neural.sif python -c "import torch; print(torch.cuda.is_available())"

# CPU-only (omit --nv)
singularity run rgp-neural-cpu.sif make verify_pipeline
```

### Environment Variables

```bash
# Set custom seed
singularity exec --nv --env RGP_SEED=123 rgp-neural.sif make reproduce_h1

# Enable fast-track mode
singularity exec --nv --env RGP_FAST_TRACK=1 rgp-neural.sif make reproduce_fast

# Multiple variables
singularity exec --nv \
    --env RGP_SEED=42 \
    --env RGP_DEVICE=cuda \
    rgp-neural.sif make reproduce_h1
```

## Docker for Local Development

### Build
```bash
# GPU version
docker build -t rgp-neural:latest -f Dockerfile .

# CPU version
docker build -t rgp-neural:cpu -f Dockerfile.cpu .
```

### Run
```bash
# Fast-track verification
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/results:/workspace/results \
    rgp-neural:latest make reproduce_fast

# Interactive shell
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    rgp-neural:latest bash
```

### Docker Compose (Multi-Service)
```bash
# Start Jupyter + TensorBoard
docker-compose up jupyter      # http://localhost:8888
docker-compose up tensorboard  # http://localhost:6006

# Run experiment
docker-compose run gpu make reproduce_fast
```

## Container Comparison

| Aspect | Docker | Singularity |
|--------|--------|-------------|
| HPC Support | ❌ Limited | ✅ Native |
| Root Required | ✅ Yes | ❌ No (with --fakeroot) |
| Scheduler Integration | ❌ Manual | ✅ SLURM/PBS/LSF |
| GPU Support | `--gpus all` | `--nv` |
| Image Format | Layered | Single .sif file |
| User Permissions | Root inside | Same as host |
| Best For | Local dev, Cloud | HPC, Research labs |

## Resource Requirements

| Experiment | Container | GPU | CPU | Memory | Time |
|------------|-----------|-----|-----|--------|------|
| verify_pipeline | Either | Optional | 2 | 8 GB | < 1 min |
| reproduce_fast | GPU | 1 | 4 | 16 GB | 3-5 min |
| reproduce_h1 | GPU | 1 | 8 | 32 GB | 4-6 hours |
| reproduce_h2 | GPU | 1 | 16 | 64 GB | 24-36 hours |
| reproduce_h3 | GPU | 1 | 8 | 32 GB | 6-8 hours |
| reproduce_all | GPU | 1 | 16 | 128 GB | 48-72 hours |
| test_unit | CPU | - | 4 | 8 GB | 2-5 min |
| test_integration | CPU | - | 8 | 16 GB | 5-10 min |

## Troubleshooting

### Build Issues

**"FATAL: container creation failed"**
```bash
# Check Singularity version
singularity --version  # Should be >= 3.8

# Verify definition file
singularity verify Singularity.def

# Try fakeroot
singularity build --fakeroot rgp-neural.sif Singularity.def
```

### Runtime Issues

**"CUDA not available"**
```bash
# Check host GPU
nvidia-smi

# Verify driver version (>= 520.61.05 for CUDA 11.8)
nvidia-smi | grep "Driver Version"

# Use --nv flag
singularity run --nv rgp-neural.sif python -c "import torch; print(torch.cuda.is_available())"
```

**"Out of memory"**
```bash
# Request more memory in job script
#SBATCH --mem=64G

# Or use CPU container
singularity run rgp-neural-cpu.sif make verify_pipeline
```

**"Permission denied"**
```bash
# Make container executable
chmod +x rgp-neural.sif

# Check file ownership
ls -la rgp-neural.sif
```

## Advanced Topics

### Custom Build Arguments

Edit `Singularity.def` to customize:

```singularity
# Change Python version
conda install -y python=3.10

# Add custom packages
pip install your-package==1.0.0

# Add system packages
apt-get install -y your-package
```

### Multi-Stage Builds

For smaller containers, use multi-stage builds (see `Singularity.def` comments).

### Signing Containers

```bash
# Sign your container
singularity sign rgp-neural.sif

# Verify signature
singularity verify rgp-neural.sif
```

### Pushing to Library

```bash
# Login to Singularity Library
singularity remote login

# Push container
singularity push rgp-neural.sif library://yourusername/rgp/rgp-neural:latest
```

## References

- [Singularity Documentation](https://sylabs.io/docs/)
- [Apptainer Documentation](https://apptainer.org/documentation/)
- [HPC Guide](../docs/HPC_GUIDE.md) - Detailed HPC deployment instructions
- [Main README](../README.md) - Project overview

## License

MIT License - See [LICENSE](../LICENSE) file.
