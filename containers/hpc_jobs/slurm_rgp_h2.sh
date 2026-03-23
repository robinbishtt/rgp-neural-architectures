#!/bin/bash
# =============================================================================
# SLURM Job Script: RGP H2 Depth Scaling Law Experiment
# Renormalization-Group Principles for Deep Neural Networks
#
# DESCRIPTION:
#   Full H2 hypothesis validation: Depth scaling law
#   L_min ~ k_c * log(ξ_data / ξ_target)
#
# EXPECTED RUNTIME: 24-36 hours on V100/A100 GPU
# OUTPUT: results/h2/ directory with scaling measurements
#
# SUBMIT:
#   sbatch slurm_rgp_h2.sh
# =============================================================================

#SBATCH --job-name=rgp-h2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@institution.edu

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-h2-${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}" "${WORK_DIR}/logs" "${WORK_DIR}/results/h2"

# ─────────────────────────────────────────────────────────────────────────────
# Load Modules
# ─────────────────────────────────────────────────────────────────────────────
module purge
module load singularity
module load cuda/11.8

echo "=== GPU Information ==="
nvidia-smi

# ─────────────────────────────────────────────────────────────────────────────
# Run H2 Experiment
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  RGP H2: Depth Scaling Law Experiment                          ║"
echo "║  Testing L_min ~ k_c * log(ξ_data / ξ_target)                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo ""

singularity exec --nv \
    --bind "${SCRATCH_DIR}:/workspace/data" \
    --bind "${WORK_DIR}/results:/workspace/results" \
    --bind "${WORK_DIR}/logs:/workspace/logs" \
    "${CONTAINER_PATH}" \
    make reproduce_h2

EXIT_CODE=$?

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ H2 experiment completed successfully!"
else
    echo "✗ H2 experiment failed with exit code ${EXIT_CODE}"
fi

rm -rf "${SCRATCH_DIR}"
exit ${EXIT_CODE}
