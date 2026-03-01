#!/usr/bin/env bash
# =============================================================================
# scripts/download_pretrained_checkpoints.sh
#
# Downloads pre-trained checkpoints from Zenodo/OSF for reviewer verification.
# Allows generating all figures without re-training (10-15 minutes).
#
# Usage: bash scripts/download_pretrained_checkpoints.sh [--output-dir DIR]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="checkpoints/pretrained"
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "=== Downloading pre-trained checkpoints ==="
echo "Target: $OUTPUT_DIR"
echo ""
echo "NOTE: Zenodo DOI will be populated upon paper acceptance."
echo "      Checkpoints include: L=100, L=500, L=1000 networks"
echo "      with full training history for all three hypotheses."
echo ""
echo "Once downloaded, run:"
echo "  python3 scripts/extract_from_checkpoints.py"
echo "  python3 figures/generate_all.py --results-root results/ --output figures/out/"
