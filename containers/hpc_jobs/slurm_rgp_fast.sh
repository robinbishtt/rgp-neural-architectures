

CONTAINER_NAME="rgp-neural.sif"
CONTAINER_PATH="${HOME}/rgp-neural-architectures/containers/${CONTAINER_NAME}"
WORK_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/rgp-${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/results"

echo "=== Loading Modules ==="
module purge
module load singularity
module load cuda/11.8

echo ""
echo "=== GPU Information ==="
nvidia-smi

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

echo ""
echo "=== Job Summary ==="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo ""

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
