

CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-h3-${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}" "${WORK_DIR}/logs" "${WORK_DIR}/results/h3"

module purge
module load singularity
module load cuda/11.8

echo "=== GPU Information ==="
nvidia-smi

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
