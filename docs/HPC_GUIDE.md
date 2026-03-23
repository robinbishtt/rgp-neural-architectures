# RGP Neural Architectures  HPC Deployment Guide

**Renormalization-Group Principles for Deep Neural Network Depth**

This guide provides comprehensive instructions for deploying and running RGP experiments on High-Performance Computing (HPC) clusters using Singularity/Apptainer containers.

---

## Table of Contents

1. [Why Singularity for HPC?](#why-singularity-for-hpc)
2. [Prerequisites](#prerequisites)
3. [Container Build Options](#container-build-options)
4. [Quick Start](#quick-start)
5. [SLURM Job Submission](#slurm-job-submission)
6. [PBS/Torque Job Submission](#pbsjob-submission)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)

---

## Why Singularity for HPC?

| Feature | Docker | Singularity/Apptainer |
|---------|--------|----------------------|
| Root privileges required | Yes | No (with `--fakeroot`) |
| HPC scheduler integration | Poor | Native support |
| GPU support | `--gpus all` | `--nv` flag |
| User permissions | Root inside | Same as host |
| Security model | Daemon-based | Single executable |

**Key advantages for research labs:**
- No root access required (critical for shared HPC)
- Direct integration with SLURM, PBS, LSF
- Reproducible environments across different clusters
- Easy sharing via Singularity Library or local files

---

## Prerequisites

### On Your Local Machine
```bash
# Install Singularity/Apptainer
# Ubuntu/Debian
sudo apt-get install singularity-container

# Or install Apptainer (recommended, newer)
sudo apt-get install apptainer

# macOS (via Homebrew)
brew install singularity

# Verify installation
singularity --version
```

### On HPC Cluster
```bash
# Most HPC clusters have Singularity pre-installed
module avail singularity
module load singularity

# Check version
singularity --version  # Should be >= 3.8
```

---

## Container Build Options

### Option 1: Local Build (Requires sudo)
```bash
# Clone repository
git clone https://anonymous.4open.science/r/rgp-neural-architectures-BB30
cd rgp-neural-architectures

# Build container
sudo singularity build rgp-neural.sif containers/Singularity.def
```

### Option 2: Remote Build (No sudo)
```bash
# Create Singularity Library account at https://cloud.sylabs.io/
singularity remote login

# Build remotely
singularity build --remote rgp-neural.sif containers/Singularity.def
```

### Option 3: HPC Build (Fakeroot - Recommended)
```bash
# On HPC cluster (no sudo needed)
singularity build --fakeroot rgp-neural.sif containers/Singularity.def
```

### Option 4: CPU-Only Build
```bash
# For CPU partitions or testing
singularity build --fakeroot rgp-neural-cpu.sif containers/Singularity.cpu.def
```

---

## Quick Start

### 1. Verify Container
```bash
# Test container integrity
singularity test rgp-neural.sif

# Check installed software
singularity exec rgp-neural.sif python --version
singularity exec --nv rgp-neural.sif python -c "import torch; print(torch.__version__)"
```

### 2. Fast-Track Verification (5 minutes)
```bash
# GPU mode
singularity run --nv rgp-neural.sif make reproduce_fast

# CPU mode
singularity run rgp-neural-cpu.sif make verify_pipeline
```

### 3. Interactive Shell
```bash
# Interactive development
singularity shell --nv rgp-neural.sif
Singularity> cd /workspace
Singularity> python experiments/h1_scale_correspondence/run_h1_validation.py
```

---

## SLURM Job Submission

### Directory Structure
```
rgp-neural-architectures/
├── containers/
│   ├── rgp-neural.sif          # Your built container
│   └── hpc_jobs/
│       ├── slurm_rgp_fast.sh   # Quick test
│       ├── slurm_rgp_h1.sh     # H1 experiment
│       ├── slurm_rgp_h2.sh     # H2 experiment
│       ├── slurm_rgp_h3.sh     # H3 experiment
│       └── slurm_rgp_full.sh   # Complete pipeline
├── logs/                        # Job logs (auto-created)
└── results/                     # Experiment results
```

### Submit Jobs
```bash
# Create logs directory
mkdir -p logs

# Quick verification (5 minutes)
sbatch containers/hpc_jobs/slurm_rgp_fast.sh

# Individual hypotheses
sbatch containers/hpc_jobs/slurm_rgp_h1.sh  # 4-6 hours
sbatch containers/hpc_jobs/slurm_rgp_h2.sh  # 24-36 hours
sbatch containers/hpc_jobs/slurm_rgp_h3.sh  # 6-8 hours

# Full pipeline (48-72 hours)
sbatch containers/hpc_jobs/slurm_rgp_full.sh
```

### Monitor Jobs
```bash
# Check queue
squeue -u $USER

# View job details
scontrol show job <job_id>

# Monitor output
tail -f logs/rgp-h1-<job_id>.out

# Cancel job
scancel <job_id>
```

### Array Jobs (Multiple Seeds)
```bash
# Submit array job for 5 different seeds
#SBATCH --array=42,123,456,789,101112
sbatch containers/hpc_jobs/slurm_rgp_h1.sh
```

---

## PBS/Torque Job Submission

### Submit Jobs
```bash
# Quick verification
qsub containers/hpc_jobs/pbs_rgp_fast.pbs

# Full experiments
qsub containers/hpc_jobs/pbs_rgp_h1.pbs
```

### Monitor Jobs
```bash
# Check queue
qstat -u $USER

# View details
qstat -f <job_id>

# Cancel job
qdel <job_id>
```

---

## Advanced Configuration

### Custom Data Binding
```bash
# Bind multiple directories
singularity run --nv \
    --bind /cluster/data:/workspace/data \
    --bind /cluster/results:/workspace/results \
    --bind /scratch/$USER:/workspace/tmp \
    rgp-neural.sif make reproduce_h1
```

### Environment Variables
```bash
# Set custom seed
singularity exec --nv \
    --env RGP_SEED=123 \
    rgp-neural.sif make reproduce_h1

# Enable fast-track mode
singularity exec --nv \
    --env RGP_FAST_TRACK=1 \
    rgp-neural.sif make reproduce_fast
```

### GPU Selection
```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
singularity run --nv rgp-neural.sif make reproduce_h1

# Multi-GPU (if supported by experiment)
export CUDA_VISIBLE_DEVICES=0,1
singularity run --nv rgp-neural.sif make reproduce_h1
```

---

## Troubleshooting

### Issue: "FATAL: container creation failed"
**Solution:**
```bash
# Check container integrity
singularity verify rgp-neural.sif

# Rebuild if corrupted
singularity build --fakeroot rgp-neural.sif containers/Singularity.def
```

### Issue: "CUDA not available"
**Solution:**
```bash
# Verify GPU on host
nvidia-smi

# Use --nv flag
singularity run --nv rgp-neural.sif python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA driver compatibility
# Container CUDA 11.8 requires host driver >= 520.61.05
```

### Issue: "Out of memory"
**Solution:**
```bash
# Request more memory in job script
#SBATCH --mem=64G

# Or use CPU-only container
singularity run rgp-neural-cpu.sif make verify_pipeline
```

### Issue: "Permission denied"
**Solution:**
```bash
# Ensure container is executable
chmod +x rgp-neural.sif

# Check Singularity version
singularity --version  # Should be >= 3.8
```

### Issue: Slow I/O
**Solution:**
```bash
# Use scratch space for temporary files
#SBATCH --tmp=100G

# Or bind scratch directory
singularity run --nv \
    --bind $SCRATCH:/workspace/tmp \
    rgp-neural.sif make reproduce_h1
```

---

## Performance Tuning

### GPU Optimization
```bash
# Set environment for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY

singularity exec --nv \
    --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    rgp-neural.sif make reproduce_h1
```

### CPU Optimization
```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

singularity exec \
    --env OMP_NUM_THREADS=8 \
    rgp-neural-cpu.sif make test_unit
```

### I/O Optimization
```bash
# Use node-local scratch for better performance
#SBATCH --gres=ssd:1  # If available

singularity run --nv \
    --bind $SLURM_TMPDIR:/workspace/data \
    rgp-neural.sif make reproduce_h1
```

---

## Resource Requirements

| Experiment | GPU | CPU | Memory | Time |
|------------|-----|-----|--------|------|
| verify_pipeline | Optional | 2 | 8 GB | < 1 min |
| reproduce_fast | 1x GPU | 4 | 16 GB | 3-5 min |
| reproduce_h1 | 1x GPU | 8 | 32 GB | 4-6 hours |
| reproduce_h2 | 1x GPU | 16 | 64 GB | 24-36 hours |
| reproduce_h3 | 1x GPU | 8 | 32 GB | 6-8 hours |
| reproduce_all | 1x GPU | 16 | 128 GB | 48-72 hours |

---

## Citation

If you use this container for your research, please cite:

```bibtex
@software{rgp_neural_architectures,
  author = {Bisht, Robin},
  title = {RGP Neural Architectures: Renormalization-Group Principles for Deep Neural Networks},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://anonymous.4open.science/r/rgp-neural-architectures-BB30
}
```

---

## Support

For issues related to:
- **Container build**: Open an issue on GitHub
- **HPC cluster**: Contact your cluster administrator
- **RGP experiments**: See main documentation in `docs/`

---

## License

MIT License - See LICENSE file for details.
