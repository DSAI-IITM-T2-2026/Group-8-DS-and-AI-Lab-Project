#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
echo "=== $(date) resume tune + maps (full-year dataset ready) ==="
python -u scripts/train_models.py --tune --epochs 15 --trials 8
python -u scripts/map_predictions.py --n 6
python -u scripts/map_state_risk.py --split test
echo "=== $(date) ALL DONE ==="
