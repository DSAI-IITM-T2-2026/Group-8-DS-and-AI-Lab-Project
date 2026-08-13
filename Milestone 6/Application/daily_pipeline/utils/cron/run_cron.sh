#!/usr/bin/env bash
# Cron / Cloud Scheduler: write final_processed/YYYY-MM-DD_test.parquet
#
# Predict California "today" (America/Los_Angeles) unless LABEL_DATE is set.
# Do NOT use UTC-tomorrow — Python caps labels at pipeline today, so that
# combination exits with "All requested labels are after as_of".
#
# Needs: ADC or GOOGLE_APPLICATION_CREDENTIALS, CDS_API_KEY in .env,
# persistent daily_pipeline/.cache (S2 CSVs are cached there).
#
# crontab example (06:30 Pacific on a Linux VM):
#   30 6 * * * /path/to/daily_pipeline/utils/cron/run_cron.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PIPELINE_DIR}/logs"
CACHE_DIR="${PIPELINE_DIR}/.cache"
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

PIDFILE="${CACHE_DIR}/daily_pipeline.pid"
if [[ -f "${PIDFILE}" ]]; then
  old_pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "daily_pipeline already running (pid ${old_pid}); skip overlapping cron" >&2
    exit 0
  fi
fi
echo "$$" >"${PIDFILE}"
trap 'rm -f "${PIDFILE}"' EXIT

if [[ -x "${PIPELINE_DIR}/.venv/bin/python" ]]; then
  PYTHON="${PIPELINE_DIR}/.venv/bin/python"
elif [[ -x "${PIPELINE_DIR}/../../.venv/bin/python" ]]; then
  PYTHON="${PIPELINE_DIR}/../../.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

export TZ="${PIPELINE_TZ:-America/Los_Angeles}"
LABEL_DATE="${LABEL_DATE:-$(date +%F)}"
LOG="${LOG_DIR}/${LABEL_DATE}.log"

cd "${PIPELINE_DIR}"
{
  echo "=== $(date -Is) label=${LABEL_DATE} python=${PYTHON} tz=${TZ} ==="
  "${PYTHON}" run_daily.py all --label-date "${LABEL_DATE}"
} >>"${LOG}" 2>&1
