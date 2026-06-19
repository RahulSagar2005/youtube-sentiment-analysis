#!/usr/bin/env bash
# run_pipeline.sh — Reproduce the full DVC training pipeline
# Usage: bash scripts/run_pipeline.sh [stage]
#   - no args  : run all stages
#   - stage    : run only that stage (e.g. data_ingestion, model_building)

set -euo pipefail

STAGE="${1:-}"

# Activate venv if it exists
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v dvc >/dev/null 2>&1; then
  echo "[pipeline] ERROR: dvc is not installed. Run 'pip install dvc' first." >&2
  exit 1
fi

if [ ! -f "dvc.yaml" ]; then
  echo "[pipeline] ERROR: dvc.yaml not found in $(pwd)" >&2
  exit 1
fi

echo "[pipeline] Starting DVC pipeline at $(date)"

if [ -z "$STAGE" ]; then
  echo "[pipeline] Running all stages..."
  dvc repro
else
  echo "[pipeline] Running single stage: $STAGE"
  dvc repro "$STAGE"
fi

echo "[pipeline] Done at $(date)"
