#!/usr/bin/env bash
# Full fire-season build (2025 Jun–Nov) then train + maps.
# DEM/S2 partial caches are reused. Needs GCS + disk (~120 GiB free OK).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export GS_NO_SIGN_REQUEST=YES
export PYTHONUNBUFFERED=1

echo "=== $(date) estimate disk ==="
python scripts/estimate_disk.py --year 2025 --start 2025-06-01 --end 2025-11-30 --require-free || {
  echo "Disk check failed or estimate errored — continuing only if you re-run without --require-free"
  python scripts/estimate_disk.py --year 2025 --start 2025-06-01 --end 2025-11-30 || true
}

echo "=== $(date) build_dataset fire season ==="
python -u scripts/build_dataset.py --year 2025 --start 2025-06-01 --end 2025-11-30

echo "=== $(date) train ==="
python -u scripts/train_models.py
python -u scripts/train_models.py --tune --epochs 15 --trials 8
python -u scripts/map_predictions.py --n 6
python -u scripts/map_state_risk.py --split test

echo "=== $(date) ALL DONE ==="
