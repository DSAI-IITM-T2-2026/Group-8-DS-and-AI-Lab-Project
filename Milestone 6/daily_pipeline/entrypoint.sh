#!/usr/bin/env bash
set -euo pipefail

LABEL_DATE="${LABEL_DATE:-$(TZ=America/Los_Angeles date +%F)}"

echo "Running daily pipeline for ${LABEL_DATE}"

exec python run_daily.py all --label-date "${LABEL_DATE}"
