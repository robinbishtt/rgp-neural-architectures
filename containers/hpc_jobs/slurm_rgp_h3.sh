#!/bin/bash
# =============================================================================
# SLURM Job Script: RGP H3 Multi-Scale Generalization Experiment
# Renormalization-Group Principles for Deep Neural Networks
#
# DESCRIPTION:
#   Full H3 hypothesis validation: RG-Net achieves statistically
#   significant advantage on hierarchical data (Wilcoxon p < 0.01)
#
# EXPECTED RUNTIME: 6-8 hours on V100/A100 GPU
# OUTPUT: results/h3/ directory with generalization metrics
#
# SUBMIT:
#   sbatch slurm_rgp_h3.sh
# =============================================================================

#SBATCH --job-name=rgp-h3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
SCRATCH_DIR="${SCRATCH}/rgp-h3-${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}" "${WORK_DIR}/logs" "${WORK_DIR}/results/h3"

# ─────────────────────────────────────────────────────────────────────────────
# Load Modules
# ─────────────────────────────────────────────────────────────────────────────
module purge
module load singularity
module load cuda/11.8

echo "=== GPU Information ==="
nvidia-smi

# ─────────────────────────────────────────────────────────────────────────────
# Run H3 Experiment
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  RGP H3: Multi-Scale Generalization Experiment                 ║"
echo "║  Testing RG-Net vs Baselines (Wilcoxon p < 0.01)              ║"
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
    make reproduce_h3

EXIT_CODE=$?

echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ H3 experiment completed successfully!"
else
    echo "✗ H3 experiment failed with exit code ${EXIT_CODE}"
fi

rm -rf "${SCRATCH_DIR}"
exit ${EXIT_CODE}
