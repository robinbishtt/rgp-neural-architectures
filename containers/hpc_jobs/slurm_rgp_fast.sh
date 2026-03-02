#!/bin/bash
# =============================================================================
# SLURM Job Script: RGP Fast-Track Verification
# Renormalization-Group Principles for Deep Neural Networks
#
# DESCRIPTION:
#   Quick verification job (3-5 minutes) for testing container and
#   environment setup before submitting long-running experiments.
#
# SUBMIT:
#   sbatch slurm_rgp_fast.sh
#
# MONITOR:
#   squeue -u $USER
#   tail -f logs/rgp-fast-%j.out
# =============================================================================

#SBATCH --job-name=rgp-fast
#SBATCH --partition=gpu          # Change to your GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # CPU cores
#SBATCH --mem=16G                # Memory
#SBATCH --time=00:15:00          # Max runtime (15 minutes)
#SBATCH --output=logs/%x-%j.out  # STDOUT
#SBATCH --error=logs/%x-%j.err   # STDERR
#SBATCH --mail-type=END,FAIL     # Email notifications
#SBATCH --mail-user=your.email@university.edu

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-${SLURM_JOB_ID}"

# Create scratch directory
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/results"

# ─────────────────────────────────────────────────────────────────────────────
# Load Singularity/Apptainer Module
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Loading Modules ==="
module purge
module load singularity          # Or: module load apptainer
module load cuda/11.8            # Match container CUDA version

# Verify GPU availability
echo ""
echo "=== GPU Information ==="
nvidia-smi

# ─────────────────────────────────────────────────────────────────────────────
# Run Fast-Track Verification
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Starting RGP Fast-Track Verification ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Container: ${CONTAINER_PATH}"
echo "Scratch: ${SCRATCH_DIR}"
echo "Start Time: $(date)"
echo ""

singularity run --nv \
    --bind "${SCRATCH_DIR}:/workspace/data" \
    --bind "${WORK_DIR}/results:/workspace/results" \
    --bind "${WORK_DIR}/logs:/workspace/logs" \
    "${CONTAINER_PATH}" \
    make reproduce_fast

EXIT_CODE=$?

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup and Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo ""

# Copy results from scratch
if [ -d "${SCRATCH_DIR}" ]; then
    echo "Copying results from scratch..."
    cp -r "${SCRATCH_DIR}"/* "${WORK_DIR}/results/" 2>/dev/null || true
    rm -rf "${SCRATCH_DIR}"
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Fast-track verification completed successfully!"
else
    echo "✗ Fast-track verification failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
