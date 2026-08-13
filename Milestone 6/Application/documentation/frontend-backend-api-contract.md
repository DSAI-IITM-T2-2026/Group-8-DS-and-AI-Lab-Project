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
| `GET` | `/pipeline-runs` | Recover recent runs, optionally filtered by date |
| `GET` | `/model/metadata` | Loaded model version, explanation capability, and data freshness |
| `GET` | `/model/features` | Frozen 86-feature model catalog |
| `GET` | `/regions/california/geometry` | Supported 0.25° California model grid |
| `GET` | `/risk-map` | Complete-day calibrated probability and priority results |
| `POST` | `/predictions` | Detailed result for one already-scored grid cell |
| `GET` | `/validation/events` | Optional historical actual-versus-predicted records |

### Configuration

```json
{
  "minPredictionDate": "2019-01-01",
  "maxPredictionDate": "2026-08-13",
  "timezone": "America/Los_Angeles",
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
    "era5": { "required": 44, "available": 44, "missing": 0, "scheduled": 0, "pending": 0 }
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
queued | running | waiting_external | succeeded | failed | interrupted
```

Stages:

```text
validating | inventory | era5 | firms | sentinel2 | sentinel5p |
preprocessing | exporting | completed
```

Successful runs contain:

```json
{
  "artifact": {
    "objectUri": "gs://wildfire-detection-first/final_processed/2025-08-01_test.parquet",
    "rowCount": 437,
    "featureCount": 86,
    "cellCount": 437,
    "labelDate": "2025-08-01",
    "eoAsOfDate": "2025-07-31",
    "featureEndDate": "2025-07-26",
    "createdAt": "2026-08-13T11:00:00Z"
  }
}
```

## Date and data policy

- Accept `2019-01-01` through California's current date.
- Treat the selected date as label/prediction day `D`.
- Use EO and prior-fire information through `D−1` and ERA5 through `D−6`.
- Build rolling features from the preceding 30 label days.
- Reject unsupported dates with `422` and a `fieldErrors.predictionDate` entry.
- Do not expose credentials, local paths, tracebacks, or private configuration.

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
