#!/bin/bash
# =============================================================================
# SLURM Job Script: RGP H1 Scale Correspondence Experiment
# Renormalization-Group Principles for Deep Neural Networks
#
# DESCRIPTION:
#   Full H1 hypothesis validation: Scale correspondence between neural
#   network depth and RG correlation length.
#
# EXPECTED RUNTIME: 4-6 hours on V100/A100 GPU
# OUTPUT: results/h1/ directory with correlation measurements
#
# SUBMIT:
#   sbatch slurm_rgp_h1.sh
# =============================================================================

#SBATCH --job-name=rgp-h1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu

# Optional: Array job for multiple seeds
#SBATCH --array=42,123,456,789,101112

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-h1-${SLURM_JOB_ID}"

# Use array task ID as seed if running array job
SEED=${SLURM_ARRAY_TASK_ID:-42}

# Create directories
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/results/h1"

# ─────────────────────────────────────────────────────────────────────────────
# Load Modules
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Loading Modules ==="
module purge
module load singularity
module load cuda/11.8

echo ""
echo "=== GPU Information ==="
nvidia-smi

# ─────────────────────────────────────────────────────────────────────────────
# Run H1 Experiment
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  RGP H1: Scale Correspondence Experiment                       ║"
echo "║  Testing correlation length ξ(k) ~ exp(-k/k_c)                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Seed: ${SEED}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo ""

# Set environment variables
export RGP_SEED=${SEED}
export RGP_FAST_TRACK=0

singularity exec --nv \
    --env RGP_SEED=${SEED} \
    --env RGP_FAST_TRACK=0 \
    --bind "${SCRATCH_DIR}:/workspace/data" \
    --bind "${WORK_DIR}/results:/workspace/results" \
    --bind "${WORK_DIR}/logs:/workspace/logs" \
    "${CONTAINER_PATH}" \
    make reproduce_h1

EXIT_CODE=$?

# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo ""

# Archive results
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ H1 experiment completed successfully!"
    echo "Results saved to: ${WORK_DIR}/results/h1/"

    # Create completion marker
    touch "${WORK_DIR}/results/h1/.completed_seed_${SEED}"
else
    echo "✗ H1 experiment failed with exit code ${EXIT_CODE}"
    touch "${WORK_DIR}/results/h1/.failed_seed_${SEED}"
fi

# Cleanup scratch
if [ -d "${SCRATCH_DIR}" ]; then
    rm -rf "${SCRATCH_DIR}"
fi

exit ${EXIT_CODE}
