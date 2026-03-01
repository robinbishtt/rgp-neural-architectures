#!/usr/bin/env bash
# =============================================================================
# containers/singularity_build.sh
#
# Builds the Singularity .sif image from Singularity.def.
# Handles both local sudo builds and HPC fakeroot builds.
#
# Usage:
#   bash containers/singularity_build.sh [OPTIONS]
#
# Options:
#   --fakeroot          Use --fakeroot instead of sudo (HPC clusters)
#   --remote            Build remotely via Sylabs Cloud (no root needed)
#   --output FILE       Output .sif path (default: rgp-neural.sif)
#   --force             Overwrite existing .sif without prompting
#   --test              Run %test section after build
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
OUTPUT_SIF="${PROJECT_ROOT}/rgp-neural.sif"
DEF_FILE="${SCRIPT_DIR}/Singularity.def"
USE_FAKEROOT=false
USE_REMOTE=false
FORCE=false
RUN_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fakeroot)  USE_FAKEROOT=true  ; shift ;;
        --remote)    USE_REMOTE=true    ; shift ;;
        --output)    OUTPUT_SIF="$2"    ; shift 2 ;;
        --force)     FORCE=true         ; shift ;;
        --test)      RUN_TEST=true      ; shift ;;
        -h|--help)
            sed -n '/^#/p' "$0" | head -30 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

# Check Singularity / Apptainer availability
if command -v apptainer &>/dev/null; then
    SIF_CMD=apptainer
elif command -v singularity &>/dev/null; then
    SIF_CMD=singularity
else
    echo "ERROR: Neither 'apptainer' nor 'singularity' found in PATH." >&2
    echo "Install from: https://apptainer.org/docs/admin/main/installation.html" >&2
    exit 1
fi

echo "=== RGP Neural Architectures — Singularity Build ==="
echo "  Command  : $SIF_CMD"
echo "  Def file : $DEF_FILE"
echo "  Output   : $OUTPUT_SIF"
echo ""

# Check for existing .sif
if [[ -f "$OUTPUT_SIF" ]]; then
    if [[ "$FORCE" == false ]]; then
        read -rp "  $OUTPUT_SIF exists. Overwrite? [y/N]: " answer
        [[ "$answer" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    fi
    rm -f "$OUTPUT_SIF"
fi

# Build command
BUILD_ARGS=("build")

if [[ "$USE_REMOTE" == true ]]; then
    BUILD_ARGS+=("--remote")
elif [[ "$USE_FAKEROOT" == true ]]; then
    BUILD_ARGS+=("--fakeroot")
else
    # Try sudo if available, otherwise fall back to fakeroot
    if sudo -n true 2>/dev/null; then
        BUILD_ARGS=("build")   # sudo handled by prepending sudo below
    else
        echo "  sudo not available — trying --fakeroot"
        BUILD_ARGS+=("--fakeroot")
    fi
fi

BUILD_ARGS+=("$OUTPUT_SIF" "$DEF_FILE")

echo "  Running: $SIF_CMD ${BUILD_ARGS[*]}"
echo ""

START_TIME=$(date +%s)

if [[ "$USE_FAKEROOT" == true || "$USE_REMOTE" == true ]]; then
    "$SIF_CMD" "${BUILD_ARGS[@]}"
else
    sudo "$SIF_CMD" "${BUILD_ARGS[@]}"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "=== Build complete in ${ELAPSED}s ==="
echo "  Image: $OUTPUT_SIF ($(du -sh "$OUTPUT_SIF" | cut -f1))"

# Optional test
if [[ "$RUN_TEST" == true ]]; then
    echo ""
    echo "=== Running container test ==="
    "$SIF_CMD" test "$OUTPUT_SIF"
fi

echo ""
echo "Usage:"
echo "  $SIF_CMD run --nv $OUTPUT_SIF make verify_pipeline"
echo "  $SIF_CMD run --nv $OUTPUT_SIF make reproduce_fast"
echo "  $SIF_CMD exec --nv $OUTPUT_SIF bash"
