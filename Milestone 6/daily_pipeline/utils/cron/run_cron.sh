#!/usr/bin/env bash
# Cron / Cloud Scheduler entry: produce final_processed/YYYY-MM-DD_test.parquet
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if date -u -d "tomorrow" +%F >/dev/null 2>&1; then
  LABEL_DATE="$(date -u -d "tomorrow" +%F)"
else
  LABEL_DATE="$(date -u -v+1d +%F)"
fi

cd "${PIPELINE_DIR}"
python3 run_daily.py all --label-date "${LABEL_DATE}"
