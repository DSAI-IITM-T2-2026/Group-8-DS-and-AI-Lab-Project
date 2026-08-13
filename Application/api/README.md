# Wildfire IQ Inference API

Inference implementation of [`frontend-backend-api-contract.md`](../documentation/frontend-backend-api-contract.md),
built directly on top of [`daily_pipeline`](../daily_pipeline/) — no endpoint
returns fabricated data. Where a real dependency is missing (a trained
model artifact, GCS credentials, a historical archive), the endpoint
answers a documented `503`/`409` instead of inventing numbers.

## What's implemented, and with what data

| Endpoint | Status | Real data source |
| --- | --- | --- |
| `GET /health` | ✅ Live | Checks `daily_pipeline` config load, grid file presence, model-load state, GCS config — all real, dynamic checks. |
| `GET /model/metadata` | ✅ Live | `modelVersion`/`explanationCapability` reflect whether a real model artifact is loaded (see below); `dataFreshness` is derived from the newest `final_processed/*.parquet` actually found locally or in GCS. `threshold` is `null` — see "Design decisions" below. |
| `GET /model/features` | ✅ Live | The 86 keys come straight from `daily_pipeline/utils/contracts/champion_86_features.json` (the frozen SHAP keep-list). `min`/`max`/`defaultValue` are populated only where there's a real basis (DEM statistics, or values mathematically bounded like sin/cos terms); everything else is `null` rather than guessed. |
| `GET /regions` / `/regions/{id}` | ✅ Live | One region, `california`, backed by the pipeline's own 0.25° grid (`era5_grid_dem_features.parquet`). Bounds/center are computed from the real grid extent, not hardcoded. |
| `GET /regions/{id}/geometry` | ✅ Live, semantic caveat | Each polygon is a real 0.25°×0.25° grid cell the model actually scores (id = `cell_id`, e.g. `"37.75_-119.25"`), not a county. The pipeline has no county boundaries — see "Design decisions". |
| `GET /risk-map` | ⚠️ Wired, needs a model | Streams the real `final_processed/<date>_test.parquet` from GCS (with local fallback), validates its causal 86-feature contract, and scores the complete day with the classifier, calibrator, ranker, and daily blend. Without `WILDFIRE_MODEL_URI` configured, returns `503 model_unavailable`. `mode=forecast_7d` returns `422` (the pipeline only supports a 1-day lead). |
| `POST /predictions` | ⚠️ Wired, needs a model | Same as above, for one grid cell, with real scenario `featureOverrides` applied on top of the real feature row. |
| `GET /validation/events` | ⚠️ Wired, needs a model + archive | Needs both a loaded model and a local copy of the multi-year historical archive (not shipped — see below). Returns `503` otherwise. |

Nothing here returns mock/sample rows dressed up as real ones: the
"⚠️ Wired" endpoints are fully implemented against real pipeline output and
will produce real predictions the moment their one missing dependency is
supplied; until then they fail loudly and explain why.

## Why a trained model isn't included

The champion model (LightGBM classifier + ranker, calibrated, scored on
the 86 features above — see `Milestone 5/metrics_summary.json` and
`Milestone 5/Wildfire_Inference.ipynb`) is produced by an external Kaggle
training notebook as `champion_model.joblib`. It is not checked into this
repository. To stream it from the model-output prefix at runtime:

```bash
cp .env.example .env
# WILDFIRE_MODEL_URI=gs://wildfire-detection-first/.../models/champion_model.joblib
```

The deployment must use Python 3.12 and the pinned pandas, NumPy,
scikit-learn, and LightGBM versions in `Application/backend/requirements.txt`;
they match the artifact's run manifest. GCS Application Default Credentials
must grant object read access. The model is deserialized directly from bytes
and is not persisted to the application filesystem.

## Why `/validation/events` needs an extra download

Per-cell actual-vs-predicted history isn't precomputed anywhere in the
repo. Rather than fabricate it, this endpoint re-scores the real historical
archive on demand. Fetch it once (see `daily_pipeline/README.md`
"Zero-download 2025 replay"):

```bash
gsutil cp gs://wildfire-detection-first/final_processed/2019_2025/2019-2025.parquet /tmp/
```

then set `WILDFIRE_HISTORICAL_ARCHIVE=/tmp/2019-2025.parquet` in `.env`.

## Design decisions (documented, not fabricated)

- **Areas are grid cells, not counties.** `frontend-backend-api-contract.md`
  section 7/8 explicitly calls out "replacing county predictions with
  grid-cell predictions" as a product-semantics change the frontend team
  should be looped in on — this deployment makes exactly that choice
  because the pipeline models a 0.25° California grid, not counties.
  `areaId` / `regionId` (for `/predictions`) is the cell's `cell_id`.
- **No single probability `threshold`.** The champion architecture blends
  within-day classifier and ranker percentiles and alerts on the daily top 25,
  not a fixed probability cutoff. `threshold` is therefore `null`.
  `probability` is the calibrated classifier output, while `alertScore`,
  `priorityRank`, and `alertTop25` reproduce the Milestone 5 inference
  notebook. `riskClass` is a display-only quintile of `alertScore`.
- **`actualAcres` is always `null`.** FIRMS gives thermal-anomaly pixel
  counts, not burned acreage; this pipeline never computes acres anywhere,
  so the field is honestly omitted rather than estimated.

## Running

```bash
cd "Application/backend"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then `GET http://localhost:8000/api/v1/health`. Swagger UI is at
`/docs`. The offline endpoints (`/health`, `/regions*`, `/model/features`)
work with zero configuration — they only touch files already in this repo.

## Layout

```text
Application/api/
  requirements.txt
  .env.example
  README.md
  app/
    main.py              FastAPI app, routing, CORS, error handlers
    config.py             Settings + reuses daily_pipeline's own config loader
    errors.py              Shared error envelope (contract section 5)
    schemas.py              Pydantic request/response models (contract section 3-4)
    grid_catalog.py       /regions* — real DEM grid + fire-region categories
    feature_catalog.py    /model/features — real 86-feature contract + DEM stats
    model_registry.py     Loads a real model artifact if configured; never fakes one
    data_access.py        Finds final_processed/<date>_test.parquet (local/GCS)
    risk_service.py       /risk-map + /predictions orchestration
    validation_service.py /validation/events orchestration
    status_service.py     /health + /model/metadata dynamic checks
    routers/               One module per endpoint group
```
