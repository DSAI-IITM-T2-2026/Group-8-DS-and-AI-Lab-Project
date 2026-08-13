# Build context MUST be "Milestone 6/Application" (the parent of this file),
# not "backend/" alone. backend/app/main.py imports the sibling "api" package
# and the pipeline worker subprocess-execs the sibling "daily_pipeline"
# directory, so both must be copied into the image alongside "backend".
#
#   docker build -f backend/Dockerfile -t wildfire-backend .
#
FROM python:3.12-slim

# libgomp1: required at runtime by lightgbm/xgboost on slim Debian images.
# curl: used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
COPY backend/requirements.txt backend/requirements.txt
COPY daily_pipeline/requirements.txt daily_pipeline/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application source: api/ and daily_pipeline/ are siblings backend/ imports
# and subprocess-execs at runtime (see backend/app/main.py, backend/app/worker.py).
COPY api/ api/
COPY backend/ backend/
COPY daily_pipeline/ daily_pipeline/

ENV PYTHONUNBUFFERED=1 \
    PIPELINE_PYTHON=/usr/local/bin/python \
    PIPELINE_ROOT=/app/daily_pipeline \
    PIPELINE_STATE_DIR=/app/backend/.state

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
