# Wildfire IQ Developer Guide

This guide covers the essentials for running, understanding, and changing the
application. See the [application README](../README.md) for full deployment and
cloud configuration details.

## How the application works

```text
React frontend
    -> FastAPI backend and job queue
        -> Daily data pipeline
            -> ERA5, FIRMS, Sentinel-2, Sentinel-5P, and GCS
        -> Inference API
            -> Champion model and validated 86-feature parquet
    <- Risk map, ranked cells, and cell details
```

| Folder | Purpose |
| --- | --- |
| `frontend/` | React, TypeScript, Vite, and the browser UI |
| `backend/` | Public FastAPI app, SQLite job state, and one pipeline worker |
| `daily_pipeline/` | Downloads/reuses source data and builds the daily parquet |
| `api/` | Loads the model and serves risk, region, and prediction endpoints |
| `documentation/` | Product and technical documentation |

A forecast request is stored in SQLite and processed by the background worker.
The pipeline reuses an existing valid parquet where possible. Otherwise, it
prepares missing causal inputs, validates the 86-feature contract, and writes
`final_processed/YYYY-MM-DD_test.parquet`. The inference service then scores
the California grid and returns the risk map and ranked cells.

## Requirements

- Python 3.12
- Node.js 20 or newer and npm
- Champion model access through `WILDFIRE_MODEL_URI`
- Google Cloud Storage and Earth Engine access for live data preparation
- Copernicus CDS credentials for new ERA5 downloads

## Run locally

From `Milestone 6/Application/`, create the environment files:

```bash
cp backend/.env.example backend/.env
cp daily_pipeline/.env.example daily_pipeline/.env
```

Set the model URI and required cloud/CDS credentials in those files. Never
commit `.env` files or service-account keys.

Install and start the backend:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --app-dir backend --reload --port 8000
```

In another terminal, start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Check backend health at
`http://127.0.0.1:8000/api/v1/health` and API documentation at
`http://127.0.0.1:8000/docs`.

## Run with Docker

After creating and editing both `.env` files, provide the service-account file
expected by `docker-compose.yml` at `secrets/service-account.json`. If you use a
different credential method, adjust that volume mount first. Then run:

```bash
docker compose up --build
```

Open the frontend at `http://localhost:8080`. The backend is available on port
`8000`. Keep only one backend worker running because jobs share quotas, cache,
and persistent state.

## Test before committing

Run Python tests from `Milestone 6/Application/`:

```bash
PYTHONPATH="$PWD:$PWD/backend" .venv/bin/python -m pytest \
  daily_pipeline/tests api/tests backend/tests -q
```

Run frontend checks from `frontend/`:

```bash
npm run typecheck
npm test
npm run build
npm run test:sites
```

## Logs and state

```text
backend/.state/pipeline-runs.sqlite3  Run history
backend/.state/logs/                  One log file per run
daily_pipeline/.cache/                Pipeline cache and Earth Engine tasks
```

These paths are ignored by Git. Preserve them in deployed environments so jobs
can recover after a restart.

## Common changes

- **UI or user flow:** update `frontend/src/` and its tests.
- **Queue or run lifecycle:** update `backend/app/` and backend tests.
- **Model scoring or API response:** update `api/app/` and API tests.
- **Data preparation or feature building:** update `daily_pipeline/`, its frozen
  feature contract, and pipeline tests together.

Keep frontend and backend response types aligned. The detailed contract is in
[`frontend-backend-api-contract.md`](frontend-backend-api-contract.md).

## Troubleshooting

- **Frontend cannot reach the API:** confirm port `8000`, Vite's `/api` proxy,
  and `PIPELINE_CORS_ORIGINS`.
- **Model is unavailable:** check `WILDFIRE_MODEL_URI`, credentials, and the
  pinned Python 3.12 dependencies.
- **Cloud authentication fails:** configure Application Default Credentials or
  a non-interactive service account with the required GCS/Earth Engine access.
- **A run appears stuck:** inspect its file in `backend/.state/logs/`; external
  Earth Engine exports may legitimately take several minutes.
