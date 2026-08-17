from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def settings(tmp_path: Path) -> Settings:
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir(exist_ok=True)
    (pipeline / "run_daily.py").write_text("print('ok')")
    return Settings(
        application_root=tmp_path,
        pipeline_root=pipeline,
        state_dir=tmp_path / "state",
        python_executable="python3",
        cors_origins=("http://localhost:5173",),
        timezone="America/Los_Angeles",
        lookback_days=30,
    )


def test_config_and_date_validation(tmp_path: Path):
    configured = settings(tmp_path)
    with TestClient(create_app(configured, start_worker=False)) as client:
        config = client.get("/api/v1/pipeline/config")
        assert config.status_code == 200
        assert config.json()["minPredictionDate"] == "2019-01-01"
        assert config.json()["cutoffLocalTime"] == "06:30"
        assert config.json()["timezone"] == "America/Los_Angeles"
        california_today = datetime.now(ZoneInfo(configured.timezone)).date()
        assert config.json()["maxPredictionDate"] == (california_today + timedelta(days=1)).isoformat()
        accepted = client.post(
            "/api/v1/pipeline-runs",
            json={"predictionDate": (california_today + timedelta(days=1)).isoformat()},
        )
        assert accepted.status_code == 202
        rejected_future = client.post(
            "/api/v1/pipeline-runs",
            json={"predictionDate": (california_today + timedelta(days=2)).isoformat()},
        )
        assert rejected_future.status_code == 422
        rejected = client.post("/api/v1/pipeline-runs", json={"predictionDate": "2016-11-21"})
        assert rejected.status_code == 422


def test_active_run_is_reused_and_terminal_run_can_rebuild(tmp_path: Path):
    app = create_app(settings(tmp_path), start_worker=False)
    with TestClient(app) as client:
        payload = {"predictionDate": "2025-08-01"}
        first = client.post("/api/v1/pipeline-runs", json=payload).json()
        second = client.post("/api/v1/pipeline-runs", json=payload).json()
        assert second["runId"] == first["runId"]
        app.state.store.update(first["runId"], status="succeeded", stage="completed", message="done")
        third = client.post("/api/v1/pipeline-runs", json=payload).json()
        assert third["runId"] != first["runId"]


def test_run_listing_and_not_found(tmp_path: Path):
    with TestClient(create_app(settings(tmp_path), start_worker=False)) as client:
        created = client.post("/api/v1/pipeline-runs", json={"predictionDate": "2025-09-15"}).json()
        assert client.get(f"/api/v1/pipeline-runs/{created['runId']}").status_code == 200
        assert len(client.get("/api/v1/pipeline-runs?predictionDate=2025-09-15").json()) == 1
        assert client.get("/api/v1/pipeline-runs/missing").status_code == 404


def test_active_run_can_be_cancelled_and_will_not_be_claimed(tmp_path: Path):
    app = create_app(settings(tmp_path), start_worker=False)
    with TestClient(app) as client:
        created = client.post("/api/v1/pipeline-runs", json={"predictionDate": "2025-09-15"}).json()
        cancelled = client.post(f"/api/v1/pipeline-runs/{created['runId']}/cancel")

        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "interrupted"
        assert cancelled.json()["errorCode"] == "cancelled_by_user"
        assert cancelled.json()["finishedAt"] is not None
        assert app.state.store.claim_next() is None
        assert client.post(f"/api/v1/pipeline-runs/{created['runId']}/cancel").json() == cancelled.json()
        assert client.post("/api/v1/pipeline-runs/missing/cancel").status_code == 404


def test_malformed_date_uses_safe_error_contract(tmp_path: Path):
    with TestClient(create_app(settings(tmp_path), start_worker=False)) as client:
        response = client.post("/api/v1/pipeline-runs", json={"predictionDate": "not-a-date"})
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"


def test_active_run_is_marked_interrupted_after_restart(tmp_path: Path):
    app = create_app(settings(tmp_path), start_worker=False)
    with TestClient(app) as client:
        run = client.post("/api/v1/pipeline-runs", json={"predictionDate": "2025-08-01"}).json()
        app.state.store.update(run["runId"], status="running", stage="era5", message="running")
    restarted = create_app(settings(tmp_path), start_worker=False)
    with TestClient(restarted) as client:
        restored = client.get(f"/api/v1/pipeline-runs/{run['runId']}").json()
        assert restored["status"] == "interrupted"
        assert restored["errorCode"] == "worker_restarted"


def test_inference_routes_are_exposed_by_the_unified_backend(tmp_path: Path):
    with TestClient(create_app(settings(tmp_path), start_worker=False)) as client:
        features = client.get("/api/v1/model/features")
        regions = client.get("/api/v1/regions")
        health = client.get("/api/v1/health")

        assert features.status_code == 200
        assert len(features.json()) == 86
        assert regions.status_code == 200
        assert regions.json()[0]["id"] == "california"
        assert "modelLoaded" in health.json()


def test_model_evaluation_exposes_versioned_held_out_metrics(tmp_path: Path):
    with TestClient(create_app(settings(tmp_path), start_worker=False)) as client:
        response = client.get("/api/v1/model/evaluation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluationVersion"] == "milestone-5-champion-2025-v1"
    assert payload["split"] == "Held-out 2025 test set"
    assert payload["rows"] == 93_518
    assert payload["positives"] == 1_325
    assert payload["baseline"]["prAuc"] == 0.0093
    assert {item["key"]: item["displayValue"] for item in payload["metrics"]} == {
        "pr_auc": "0.1451",
        "roc_auc": "0.7718",
        "recall_at_25": "36.38%",
        "precision_at_25": "9.01%",
        "brier": "0.0131",
        "pr_auc_lift": "15.6×",
    }


def test_today_validation_labels_are_not_mature(tmp_path: Path):
    configured = settings(tmp_path)
    california_today = datetime.now(ZoneInfo(configured.timezone)).date()
    with TestClient(create_app(configured, start_worker=False)) as client:
        response = client.get("/api/v1/validation/day", params={"date": california_today.isoformat()})

    assert response.status_code == 200
    assert response.json()["status"] == "not_mature"
    assert response.json()["items"] == []
    assert response.json()["summary"] is None


def test_unavailable_run_is_terminal(tmp_path: Path):
    app = create_app(settings(tmp_path), start_worker=False)
    with TestClient(app) as client:
        created = client.post("/api/v1/pipeline-runs", json={"predictionDate": "2025-08-01"}).json()
        app.state.store.update(
            created["runId"],
            status="unavailable",
            stage="inventory",
            message="Tomorrow's data is not available yet.",
        )
        result = client.get(f"/api/v1/pipeline-runs/{created['runId']}").json()

    assert result["status"] == "unavailable"
    assert result["finishedAt"] is not None
    assert result["errorCode"] is None


def test_fallback_artifact_fields_survive_api_schema(tmp_path: Path):
    app = create_app(settings(tmp_path), start_worker=False)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/pipeline-runs", json={"predictionDate": "2025-08-01"}
        ).json()
        app.state.store.update(
            created["runId"],
            status="succeeded",
            stage="completed",
            message="Prediction data is ready.",
            artifact={
                "objectUri": "gs://test/2025-08-01_test.parquet",
                "rowCount": 437,
                "featureCount": 86,
                "cellCount": 437,
                "labelDate": "2025-08-01",
                "eoAsOfDate": "2025-07-31",
                "featureEndDate": "2025-07-25",
                "requiredFeatureEndDate": "2025-07-26",
                "createdAt": "2025-07-31T14:00:00Z",
                "forecastMode": "provisional_tomorrow",
                "artifactQuality": "era5_fallback",
                "needsRefresh": True,
                "availabilityPolicy": "cutoff_snapshot",
                "sourceSnapshots": {
                    "era5": {
                        "required": 38,
                        "available": 37,
                        "missing": 1,
                        "ready": True,
                        "mode": "latest_causal",
                        "ageDays": 1,
                        "exactAvailable": False,
                    }
                },
            },
        )
        result = client.get(
            f"/api/v1/pipeline-runs/{created['runId']}"
        ).json()

    assert result["artifact"]["artifactQuality"] == "era5_fallback"
    assert result["artifact"]["needsRefresh"] is True
    assert result["artifact"]["sourceSnapshots"]["era5"]["exactAvailable"] is False


def test_risk_map_fails_safely_without_a_model(tmp_path: Path):
    with TestClient(create_app(settings(tmp_path), start_worker=False)) as client:
        response = client.get(
            "/api/v1/risk-map",
            params={
                "region_id": "california",
                "timestamp": "2025-08-01T12:00:00Z",
                "mode": "forecast_24h",
            },
        )
        assert response.status_code in {409, 503}
        assert response.json()["code"] in {"feature_data_unavailable", "model_unavailable"}
        assert "/Users/" not in response.text
