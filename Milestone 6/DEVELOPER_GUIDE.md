# Wildfire IQ Developer Guide

This guide covers the essentials for running, understanding, and changing the
application. See the [application README](Application/README.md) for full deployment and
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

`POST /api/v1/pipeline-runs/{runId}/cancel` atomically changes an active run to
`interrupted` with `cancelled_by_user`, then sends `SIGTERM` to the pipeline's
process group and escalates to `SIGKILL` after a short grace period. Conditional
store updates keep buffered pipeline events from overwriting the terminal
state. Queued cancellation prevents the worker from claiming the run. This
stops local orchestration only; already-submitted Earth Engine exports are not
deleted.

The supported maximum label date is tomorrow in `America/Los_Angeles`. The
cutoff is 06:30 California time on D−1 and is offset-aware across daylight
saving changes. Before the cutoff, the request is `unavailable`. After it, a
tomorrow artifact is reused only when both its parquet and adjacent provenance
JSON validate for the requested date and cutoff.

Tomorrow inventory is read-only. ERA5 is exact through D−6 with complete
rolling history; FIRMS is exact through D−2; DEM is static; Sentinel-2 selects
the latest completed window ending no later than D−1; Sentinel-5P selects the
latest observation through D−1 with a maximum age of seven days. Objects first
created after the cutoff are excluded. Missing inputs return source-specific
messages and do not schedule downloads or Earth Engine exports. Today and
historical dates retain their normal preparation behaviour.

The frontend also reads `GET /api/v1/model/evaluation`. This versioned response
contains the champion model's held-out 2025 scorecard and must remain clearly
labeled as historical evaluation rather than selected-forecast performance.

After a risk map succeeds, the frontend requests
`GET /api/v1/validation/day?date=YYYY-MM-DD`. Dates before California today use
the exact `load_scored_day` roster already cached for the map. Truth comes from
a predicate-pushed historical archive slice for 2019–2025 or a completed daily
FIRMS GeoTIFF for 2026 onward. FIRMS confidence must be at least 30. Today and
tomorrow return `not_mature`; absent completed prior-day objects return
`pending`, without scheduling any cloud work. Authentication and decode errors
remain explicit 503 responses. Keep this label path read-only and separate from
the 86-feature parquet so observed outcomes can never leak into inference.

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
[`frontend-backend-api-contract.md`](Application/documentation/frontend-backend-api-contract.md).

## Troubleshooting

- **Frontend cannot reach the API:** confirm port `8000`, Vite's `/api` proxy,
  and `PIPELINE_CORS_ORIGINS`.
- **Model is unavailable:** check `WILDFIRE_MODEL_URI`, credentials, and the
  pinned Python 3.12 dependencies.
- **Cloud authentication fails:** configure Application Default Credentials or
  a non-interactive service account with the required GCS/Earth Engine access.
- **A run appears stuck:** inspect its file in `backend/.state/logs/`; external
  Earth Engine exports may legitimately take several minutes.
- **Tomorrow is unavailable:** inspect `sourceInventory` for the exact source,
  required-through date, selected-through date, age, selection mode, and
  message. This terminal non-error state starts no cloud preparation.
