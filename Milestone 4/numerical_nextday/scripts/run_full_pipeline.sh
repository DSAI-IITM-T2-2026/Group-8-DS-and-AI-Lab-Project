#!/usr/bin/env bash
# Solo full Milestone 4 pipeline: GCS data build (lag-5) → train → eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export GS_NO_SIGN_REQUEST=YES
WORKER="${WORKER:-local}"
YEARS="${YEARS:-2019-2025}"
MONTHS="${MONTHS:-1-12}"

echo "== M4 full pipeline =="
echo "ROOT=$ROOT WORKER=$WORKER YEARS=$YEARS MONTHS=$MONTHS"

python scripts/run_pipeline.py run --stage build_data \
  --years "$YEARS" --months "$MONTHS" --worker "$WORKER" --era5-lag-days 5

python scripts/run_pipeline.py run --stage train_all

echo "== done: see artifacts/ and outputs/m4_shared_cache/ =="
