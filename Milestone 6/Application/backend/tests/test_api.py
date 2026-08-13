from pathlib import Path

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
    with TestClient(create_app(settings(tmp_path), start_worker=False)) as client:
        config = client.get("/api/v1/pipeline/config")
        assert config.status_code == 200
        assert config.json()["minPredictionDate"] == "2019-01-01"
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
