#!/usr/bin/env bash
# Full-year 2025 candidate: Jan–Nov (Dec S2 often incomplete) → train → maps.
# Streams GCS; reuses data/cache/ when present (DEM already local).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export GS_NO_SIGN_REQUEST=YES
export PYTHONUNBUFFERED=1

START="${START:-2025-01-01}"
END="${END:-2025-11-30}"

echo "=== $(date) FULL YEAR build ${START} → ${END} ==="
python -u scripts/build_dataset.py --year 2025 --start "$START" --end "$END"

echo "=== $(date) train + tune + maps ==="
python -u scripts/train_models.py
python -u scripts/train_models.py --tune --epochs 15 --trials 8
python -u scripts/map_predictions.py --n 6
python -u scripts/map_state_risk.py --split test

echo "=== $(date) ALL DONE ==="
