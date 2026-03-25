


CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-h1-${SLURM_JOB_ID}"

SEED=${SLURM_ARRAY_TASK_ID:-42}

mkdir -p "${SCRATCH_DIR}"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/results/h1"

echo "=== Loading Modules ==="
module purge
module load singularity
module load cuda/11.8

echo ""
echo "=== GPU Information ==="
nvidia-smi

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

echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ H1 experiment completed successfully!"
    echo "Results saved to: ${WORK_DIR}/results/h1/"

    touch "${WORK_DIR}/results/h1/.completed_seed_${SEED}"
else
    echo "✗ H1 experiment failed with exit code ${EXIT_CODE}"
    touch "${WORK_DIR}/results/h1/.failed_seed_${SEED}"
fi

if [ -d "${SCRATCH_DIR}" ]; then
    rm -rf "${SCRATCH_DIR}"
fi

exit ${EXIT_CODE}
