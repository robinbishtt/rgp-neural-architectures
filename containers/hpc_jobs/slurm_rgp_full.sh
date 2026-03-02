#!/bin/bash
# =============================================================================
# SLURM Job Script: RGP Full Pipeline (H1 + H2 + H3)
# Renormalization-Group Principles for Deep Neural Networks
#
# DESCRIPTION:
#   Complete reproduction of all three hypotheses.
#   WARNING: This takes 48-72 hours on high-end GPU.
#
# EXPECTED RUNTIME: 48-72 hours on A100/H100 GPU
# OUTPUT: results/{h1,h2,h3}/ directories
#
# SUBMIT:
#   sbatch slurm_rgp_full.sh
# =============================================================================

#SBATCH --job-name=rgp-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-full-${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}" "${WORK_DIR}/logs" "${WORK_DIR}/results"

# ─────────────────────────────────────────────────────────────────────────────
# Load Modules
# ─────────────────────────────────────────────────────────────────────────────
module purge
module load singularity
module load cuda/11.8

echo "=== GPU Information ==="
nvidia-smi

# ─────────────────────────────────────────────────────────────────────────────
# Run Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  RGP FULL PIPELINE                                             ║"
echo "║  H1 + H2 + H3 Complete Reproduction                            ║"
echo "║  Estimated Runtime: 48-72 hours                               ║"
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
    make reproduce_all

EXIT_CODE=$?

echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Full pipeline completed successfully!"
    echo "All results saved to: ${WORK_DIR}/results/"
else
    echo "✗ Pipeline failed with exit code ${EXIT_CODE}"
fi

rm -rf "${SCRATCH_DIR}"
exit ${EXIT_CODE}
