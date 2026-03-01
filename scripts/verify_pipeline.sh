#!/usr/bin/env bash
# scripts/verify_pipeline.sh
# Smoke test — runs in under 60 seconds on any hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/verify_pipeline.py
