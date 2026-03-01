#!/usr/bin/env bash
# scripts/reproduce_fast_h1.sh — Fast-track H1 (Scale Correspondence) only.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/reproduce_fast.sh --hypothesis h1
