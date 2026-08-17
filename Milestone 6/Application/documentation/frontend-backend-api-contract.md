# Wildfire IQ prediction-data API contract

**Base path:** `/api/v1`
**Workflow:** asynchronous Milestone 6 feature preparation followed by model inference

The browser submits a prediction date and polls a persistent backend job. After the validated daily parquet is ready, the inference service scores that exact artifact; it never downloads raw inputs or rebuilds features.

## Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API, worker, pipeline, and credential readiness |
| `GET` | `/pipeline/config` | Supported dates, timezone, lookback, and feature count |
| `POST` | `/pipeline-runs` | Queue or reuse an active run for a prediction date |
| `GET` | `/pipeline-runs/{runId}` | Read progress and the final artifact |
| `POST` | `/pipeline-runs/{runId}/cancel` | Stop queued or locally running preparation |
| `GET` | `/pipeline-runs` | Recover recent runs, optionally filtered by date |
| `GET` | `/model/metadata` | Loaded model version, explanation capability, and data freshness |
| `GET` | `/model/evaluation` | Versioned held-out champion evaluation scorecard |
| `GET` | `/model/features` | Frozen 86-feature model catalog |
| `GET` | `/regions/california/geometry` | Supported 0.25° California model grid |
| `GET` | `/risk-map` | Complete-day calibrated probability and priority results |
| `POST` | `/predictions` | Detailed result for one already-scored grid cell |
| `GET` | `/validation/day?date=YYYY-MM-DD` | Selected-day FIRMS truth and Top-25 capture summary |
| `GET` | `/validation/events` | Optional historical actual-versus-predicted records |

### Configuration

```json
{
  "minPredictionDate": "2019-01-01",
  "maxPredictionDate": "2026-08-14",
  "timezone": "America/Los_Angeles",
  "cutoffLocalTime": "06:30",
  "lookbackDays": 30,
  "expectedFeatureCount": 86
}
```

### Create a run

```http
POST /api/v1/pipeline-runs
Content-Type: application/json

{ "predictionDate": "2025-08-01" }
```

The API returns `202 Accepted`. If the same date already has a `queued`, `running`, or `waiting_external` run, that run is returned. A new request after a terminal run creates a fresh run and rebuilds the final parquet.

### Pipeline run

```json
{
  "runId": "uuid",
  "predictionDate": "2025-08-01",
  "status": "waiting_external",
  "stage": "sentinel5p",
  "message": "Waiting for Sentinel-5P Earth Engine exports.",
  "progressCompleted": 12,
  "progressTotal": 31,
  "sourceInventory": {
    "era5": {
      "required": 44, "available": 44, "missing": 0, "scheduled": 0, "pending": 0,
      "requiredThroughDate": "2026-08-08", "selectedThroughDate": "2026-08-08",
      "ageDays": 0, "mode": "exact", "ready": true,
      "message": "ERA5 history is complete through 2026-08-08."
    }
  },
  "artifact": null,
  "errorCode": null,
  "createdAt": "2026-08-13T10:00:00Z",
  "startedAt": "2026-08-13T10:00:01Z",
  "finishedAt": null
}
```

Statuses:

```text
queued | running | waiting_external | succeeded | unavailable | failed | interrupted
```

Cancellation is idempotent. Cancelling an active run persists terminal
`interrupted` with `errorCode: "cancelled_by_user"` before terminating the
local pipeline process group. This prevents queued work from being claimed and
prevents late pipeline events from replacing the stopped state. Earth Engine
exports submitted before cancellation may continue remotely and can be reused
by a later run.

Stages:

```text
validating | inventory | era5 | firms | sentinel2 | sentinel5p |
preprocessing | exporting | completed
```

Successful runs contain:

```json
{
  "artifact": {
    "objectUri": "gs://wildfire-detection-first/final_processed/2026-08-14_test.parquet",
    "rowCount": 437,
    "featureCount": 86,
    "cellCount": 437,
    "labelDate": "2026-08-14",
    "eoAsOfDate": "2026-08-13",
    "featureEndDate": "2026-08-08",
    "createdAt": "2026-08-13T11:00:00Z",
    "cutoffAt": "2026-08-13T06:30:00-07:00",
    "timezone": "America/Los_Angeles",
    "forecastMode": "provisional_tomorrow",
    "sourceSnapshots": {},
    "provenanceUri": "gs://wildfire-detection-first/final_processed/2026-08-14_test.provenance.json"
  }
}
```

## Date and data policy

- Accept `2019-01-01` through California tomorrow; reject later dates.
- Treat the selected date as label/prediction day `D`.
- Use ERA5 through `D−6` and FIRMS neighbour history through `D−2`.
- For tomorrow, use the latest completed Sentinel-2 window ending no later
  than `D−1`, the latest Sentinel-5P observation through `D−1` with maximum
  age seven days, and the verified static DEM.
- Build rolling features from the preceding 30 label days.
- A tomorrow run is eligible at 06:30 California time on `D−1`. Reuse requires
  a matching parquet and provenance sidecar. Otherwise perform a cutoff-aware,
  read-only inventory and return source-specific `unavailable` without
  scheduling downloads or Earth Engine work when a causal input is missing.
- Reject unsupported dates with `422` and a `fieldErrors.predictionDate` entry.
- Do not expose credentials, local paths, tracebacks, or private configuration.

## Model evaluation contract

`GET /model/evaluation` returns a versioned held-out 2025 summary with model
identity, split label, row and positive counts, the naive PR-AUC baseline, and
PR-AUC, ROC-AUC, Recall@25, Precision@25, Brier score, and PR-AUC lift metric
entries. Each metric includes a numeric value, display value, and description.
The frontend must label this as historical model-level evaluation, never as
measured quality for the selected forecast.

## Selected-day validation contract

`GET /validation/day?date=YYYY-MM-DD` compares the exact complete-day scored
roster used by `/risk-map` with post-event FIRMS labels. It returns
`available`, `not_mature`, or `pending`. California today and tomorrow are
always `not_mature`; a completed prior day whose truth object has not arrived
is `pending`. Neither state schedules downloads, exports, or inference.

For 2019–2025, labels are sliced from the configured historical archive with
Parquet predicate pushdown. From 2026 onward, labels come from the completed
`firms_daily_geotiff/YYYY-MM-DD.tif`, using confidence `>= 30`, mapped onto the
supported 0.25-degree model grid. Storage authentication and corrupt-raster
errors are explicit service errors and must not be converted to an all-negative
day.

An available response provides each cell's `actualEvent`, `alertTop25`, FIRMS
pixel evidence when present, and TP/FP/FN/TN outcome. The summary includes
observed cells, captured cells, Recall@25, Precision@25, false alerts, and the
actual returned Top-25 count. Recall is `null` when there are no positive
cells. Observations remain separate from feature and prediction artifacts to
prevent target leakage.

## Inference contract

- The only inference input is `final_processed/YYYY-MM-DD_test.parquet`.
- The parquet must contain exactly one requested label date, `eo_asof_date=D−1`, `feature_end_date=D−6`, unique supported cells, and all 86 finite model features.
- The API applies the artifact classifier, probability calibrator, ranker, within-day percentiles, and artifact blend weights in the same order as `Milestone 5/Wildfire_Inference.ipynb`.
- `probability` is calibrated wildfire probability. `alertScore` is the blended daily priority signal; `priorityRank` and `alertTop25` are derived within the complete prediction day.
- Single-cell requests reuse the complete-day scored result. They do not rank a cell in isolation.

Risk-map items expose:

```json
{
  "areaId": "37.75_-119.25",
  "probability": 0.084,
  "rawProbability": 0.112,
  "alertScore": 0.941,
  "priorityRank": 4,
  "alertTop25": true,
  "riskClass": "very_high"
}
```

## Polling and recovery

The frontend polls every five seconds only while the run is active. It retains the most recent run ID so a reload can recover either active preparation or completed inference results. Earth Engine work may continue after the browser closes.
