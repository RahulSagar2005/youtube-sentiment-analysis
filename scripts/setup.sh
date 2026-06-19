#!/usr/bin/env bash
# setup.sh — Bootstrap the local dev environment for youtube-sentiment-analysis
# Usage: bash scripts/setup.sh

set -euo pipefail

echo "[setup] Creating virtual environment at .venv..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo "[setup] Installing project requirements..."
pip install -r requirements.txt

if [ -f "requirements-dev.txt" ]; then
  echo "[setup] Installing dev requirements (pytest, coverage)..."
  pip install -r requirements-dev.txt
fi

echo "[setup] Pulling DVC-tracked data/artifacts (if configured)..."
if command -v dvc >/dev/null 2>&1 && [ -f "dvc.yaml" ]; then
  dvc pull || echo "[setup] dvc pull skipped (no remote configured)"
else
  echo "[setup] DVC not installed or dvc.yaml missing — skipping data pull"
fi

echo "[setup] Running test suite to verify install..."
pytest -q || echo "[setup] Tests failed — see output above"

echo "[setup] Done. Activate with: source .venv/bin/activate"
