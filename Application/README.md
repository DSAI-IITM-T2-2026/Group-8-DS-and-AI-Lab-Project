# Wildfire IQ Application

`Application/` is the complete, deployable Wildfire IQ application. It is
self-contained and does not import runtime code from the repository's milestone
folders. The original `Milestone 6/api` and `Milestone 6/daily_pipeline`
directories remain unchanged for milestone history and comparison.

The application accepts a California prediction date, checks or prepares all
causal source data, creates and validates
`final_processed/YYYY-MM-DD_test.parquet`, scores every supported grid cell with
the champion model, and displays the daily risk map and ranked results.

## Application structure

```text
Application/
├── frontend/        React, TypeScript, and Vite user interface
├── backend/         Public FastAPI server, SQLite queue, and pipeline worker
├── api/             Inference routers and champion-model scoring services
├── daily_pipeline/  GCS, Earth Engine, CDS, and parquet preparation pipeline
└── documentation/   API contract and implementation notes
```

## Prerequisites

- Python 3.12
- Node.js 20 or newer with npm
- Access to the configured Google Cloud Storage bucket
- An Earth Engine-enabled Google Cloud identity for dates requiring new exports
- Copernicus CDS credentials for dates requiring new ERA5 downloads
- The champion model object configured by `WILDFIRE_MODEL_URI`

Use Python 3.12 because the champion model was trained with Python 3.12 and the
application pins compatible numerical packages.

## 1. Clone and enter the application

Run all commands from the repository checkout unless a command changes the
working directory explicitly.

```bash
git clone https://github.com/DSAI-IITM-T2-2026/Group-8-DS-and-AI-Lab-Project.git
cd Group-8-DS-and-AI-Lab-Project/Application
```

## 2. Configure the backend and pipeline

Create local environment files from the committed examples:

```bash
cp backend/.env.example backend/.env
cp daily_pipeline/.env.example daily_pipeline/.env
```

At minimum, review these values:

```env
# backend/.env
PIPELINE_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
WILDFIRE_MODEL_URI=gs://wildfire-detection-first/champion_training_outputs/champion_training_outputs_stage_c_knn_high_medium_fire/models/champion_model.joblib
WILDFIRE_ALLOW_GCS=true

# daily_pipeline/.env
GOOGLE_CLOUD_PROJECT=plated-mechanic-418917
CDS_API_KEY=your-cds-uid:your-cds-api-key
# GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Do not commit either `.env` file or a service-account key. On a developer
machine, Application Default Credentials may be used instead of a JSON key. A
deployed worker must use a non-interactive identity; it must never depend on a
browser-based Earth Engine login.

The runtime identity needs:

- read/write access to the configured GCS input and output objects;
- Earth Engine access in the configured project when raw exports are missing;
- permission to use the configured Earth Engine grid asset.

## 3. Install and run the backend

From `Application/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Keep this terminal running. The backend provides both the preparation endpoints
and inference endpoints at `http://127.0.0.1:8000/api/v1`.

Check readiness in another terminal:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

The service intentionally runs one pipeline worker. Do not start multiple
worker replicas against the same cache because preparation jobs share Earth
Engine quotas and persistent state.

## 4. Install and run the frontend

Open a second terminal:

```bash
cd Group-8-DS-and-AI-Lab-Project/Application/frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. In development, Vite proxies `/api` to the
backend at `http://127.0.0.1:8000`, so no frontend URL change is needed.

## 5. Run a prediction from the UI

1. Confirm the health endpoint reports that the API, worker, pipeline, cloud
   credentials, and model are ready.
2. Open the frontend and choose the date to predict.
3. Select **Generate wildfire forecast**.
4. The worker first checks for a valid final test parquet.
5. If it is absent, the worker checks GCS for all required ERA5, FIRMS,
   Sentinel-2, and Sentinel-5P inputs. Existing inputs are reused; only missing
   inputs are prepared or scheduled.
6. The pipeline builds and validates the 86-feature parquet.
7. The inference service scores the full supported California grid and the UI
   displays the map, top-25 cells, ranked list, probabilities, and cell details.

Earth Engine exports can remain in a waiting state for several minutes. A
completed cloud export and local pipeline cache are retained across retries.

## Logs and local state

Every preparation run has a separate log:

```text
Application/backend/.state/logs/<run-id>.log
```

Follow the newest local run from the repository root:

```bash
tail -f "$(ls -t Application/backend/.state/logs/*.log | head -1)"
```

Other persistent runtime paths:

```text
Application/backend/.state/pipeline-runs.sqlite3  job history
Application/daily_pipeline/.cache/                processed cache and EE task registry
```

These paths are ignored by Git. Mount both on persistent storage in a VM or
container deployment so restarts can reconcile previous work.

## Test and build commands

Backend, inference, and pipeline tests, from `Application/`:

```bash
PYTHONPATH="$PWD:$PWD/backend" .venv/bin/python -m pytest \
  daily_pipeline/tests api/tests backend/tests -q
```

Frontend checks, from `Application/frontend/`:

```bash
npm run typecheck
npm test
npm run build
npm run test:sites
```

A successful frontend deployment build contains:

```text
frontend/dist/client/index.html
frontend/dist/server/index.js
frontend/dist/.openai/hosting.json
```

## Deployment configuration

Deploy the backend and frontend separately.

Backend environment values:

```env
WILDFIRE_MODEL_URI=gs://wildfire-detection-first/champion_training_outputs/champion_training_outputs_stage_c_knn_high_medium_fire/models/champion_model.joblib
WILDFIRE_ALLOW_GCS=true
PIPELINE_CORS_ORIGINS=https://your-frontend.example
PIPELINE_PYTHON=/absolute/path/to/Application/.venv/bin/python
PIPELINE_STATE_DIR=/persistent/wildfire-iq/backend-state
GOOGLE_CLOUD_PROJECT=plated-mechanic-418917
GOOGLE_APPLICATION_CREDENTIALS=/secure/mount/service-account.json
CDS_API_KEY=your-cds-uid:your-cds-api-key
```

For a frontend build, provide the public backend URL including `/api/v1`:

```bash
cd Application/frontend
npm ci
VITE_API_BASE_URL=https://your-backend.example/api/v1 npm run build
```

Deployment requirements:

- one backend worker replica;
- persistent storage for `backend/.state` and `daily_pipeline/.cache`;
- sufficient memory and temporary disk for large satellite CSV/parquet files;
- a non-interactive cloud identity with GCS and Earth Engine access;
- restricted CORS origins matching the deployed frontend;
- backend health checks against `/api/v1/health`.

## Troubleshooting

- **Frontend cannot reach backend:** confirm the backend is on port 8000 and
  production `VITE_API_BASE_URL` ends with `/api/v1`.
- **Cloud authentication failed:** configure service-account credentials or
  valid ADC. Worker runs intentionally do not open interactive OAuth pages.
- **Model unavailable:** verify `WILDFIRE_MODEL_URI`, bucket read access, and
  the model's Python dependency versions.
- **Pipeline appears idle:** inspect the per-run log. Large parquet conversion,
  feature construction, and Earth Engine exports can be quiet for several
  minutes.
- **Pipeline busy:** wait for the active job to finish. The single-job lock
  prevents API, cron, and manual pipeline runs from overlapping.

Never commit `.env`, credentials, model artifacts, SQLite state, pipeline
caches, Python environments, `node_modules`, frontend `dist`, coverage output,
or logs.
