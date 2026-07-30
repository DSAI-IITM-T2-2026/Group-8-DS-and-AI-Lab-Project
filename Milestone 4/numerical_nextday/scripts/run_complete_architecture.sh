#!/usr/bin/env zsh
# Completes architecture after lag-5 train_all: MLP + lag-0 ablation + eval.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export GS_NO_SIGN_REQUEST=YES
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset M4_SKIP_MLP || true
export MPLCONFIGDIR="$PWD/outputs/.mplconfig"
mkdir -p "$MPLCONFIGDIR" outputs
LOG="outputs/complete_architecture.log"
exec > >(tee -a "$LOG") 2>&1

echo "===== START $(date) ====="

echo "=== 1/3 MLP schedule (Stage C fire_season) ==="
python -u - <<'PY'
from numerical_nextday.config import load_config
from numerical_nextday.train.mlp import run_mlp_schedule
raise SystemExit(run_mlp_schedule(load_config()))
PY

echo "=== 2/3 build_lag0_data ==="
python -u scripts/run_pipeline.py run --stage build_lag0_data \
  --years 2019-2025 --months 1-12 --worker local

echo "=== 3/3 train_lag0_ablation + eval_figures ==="
python -u scripts/run_pipeline.py run --stage train_lag0_ablation
python -u scripts/run_pipeline.py run --stage eval_figures

echo "===== END $(date) ====="
