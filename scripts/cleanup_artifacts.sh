#!/usr/bin/env bash
# scripts/cleanup_artifacts.sh — Remove generated files for clean state.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Removing generated artifacts..."
rm -rf results/h1/* results/h2/* results/h3/* results/fss/*
rm -rf figures/out/*
rm -rf checkpoints/
rm -rf logs/
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "Clean state restored."
