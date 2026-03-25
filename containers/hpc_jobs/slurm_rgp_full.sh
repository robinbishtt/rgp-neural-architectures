

CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-full-${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}" "${WORK_DIR}/logs" "${WORK_DIR}/results"

module purge
module load singularity
module load cuda/11.8

echo "=== GPU Information ==="
nvidia-smi

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
