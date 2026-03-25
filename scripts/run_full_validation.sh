
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_SLOW=false
while [[ $
    case $1 in
        --skip-slow) SKIP_SLOW=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== Full Validation Suite ==="
START=$(date +%s)

echo ""
echo "--- Unit Tests ---"
pytest tests/unit/ -v --timeout=120

echo ""
echo "--- Integration Tests ---"
pytest tests/integration/ -v --timeout=600

echo ""
echo "--- Stability Tests ---"
pytest tests/stability/ -v --timeout=300

echo ""
echo "--- Spectral Tests ---"
pytest tests/spectral/ -v --timeout=300

echo ""
echo "--- Scaling Tests ---"
pytest tests/scaling/ -v --timeout=600

if [[ "$SKIP_SLOW" == false ]]; then
    echo ""
    echo "--- Ablation Studies ---"
    pytest tests/ablation/ -v --timeout=1200

    echo ""
    echo "--- Robustness Tests ---"
    pytest tests/robustness/ -v --timeout=600

    echo ""
    echo "--- Validation Tests ---"
    pytest tests/validation/ -v --timeout=1800
fi

echo ""
echo "--- Hypothesis Validation ---"
bash scripts/validate_hypotheses.sh

echo ""
echo "=== Full validation complete in $(( $(date +%s) - START ))s ==="
