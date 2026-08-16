from __future__ import annotations

import os
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import HealthResponse, PipelineConfig, PipelineRun, PipelineRunCreate
from .settings import Settings, load_settings
from .store import RunStore
from .worker import PipelineWorker

logger = logging.getLogger("wildfire_iq_api")

APPLICATION_ROOT = Path(__file__).resolve().parents[2]
if str(APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(APPLICATION_ROOT))

from api.app import status_service as inference_status  # noqa: E402
from api.app.errors import ApiError as InferenceApiError, error_body  # noqa: E402
from api.app.model_registry import get_model_registry  # noqa: E402
from api.app.routers import model, predictions, regions, risk_map, validation  # noqa: E402


def create_app(settings: Settings | None = None, *, start_worker: bool = True) -> FastAPI:
    settings = settings or load_settings()
    store = RunStore(settings.database_path)
    store.interrupt_orphans()
    worker = PipelineWorker(settings, store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="Wildfire IQ Pipeline API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.worker = worker
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins), allow_credentials=False,
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        gcs = bool(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or adc.is_file()
        )
        inference = inference_status.health_checks()
        registry = get_model_registry()
        core_ready = (
            settings.pipeline_root.exists()
            and (worker.alive or not start_worker)
            and inference["status"] == "ready"
            and registry.is_loaded
        )
        return HealthResponse(
            status="ready" if core_ready else "degraded",
            api=True, worker=worker.alive or not start_worker,
            pipelineConfigured=(settings.pipeline_root / "run_daily.py").is_file(),
            gcsCredentialsConfigured=gcs,
            earthEngineCredentialsConfigured=gcs,
            modelLoaded=registry.is_loaded,
            modelVersion=registry.version,
            inferenceChecks=inference["checks"],
        )

    @app.get("/api/v1/pipeline/config", response_model=PipelineConfig)
    def pipeline_config() -> PipelineConfig:
        return PipelineConfig(
            minPredictionDate=settings.min_prediction_date,
            maxPredictionDate=settings.max_prediction_date,
            timezone=settings.timezone,
            lookbackDays=settings.lookback_days,
            expectedFeatureCount=settings.expected_feature_count,
        )

    def validate_date(value: date) -> None:
        minimum = date.fromisoformat(settings.min_prediction_date)
        maximum = date.fromisoformat(settings.max_prediction_date)
        if value < minimum or value > maximum:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unsupported_prediction_date",
                    "message": f"Choose a date from {minimum.isoformat()} through {maximum.isoformat()}.",
                    "fieldErrors": {"predictionDate": "Date is outside the supported range."},
                },
            )

    @app.post("/api/v1/pipeline-runs", response_model=PipelineRun, status_code=status.HTTP_202_ACCEPTED)
    def create_run(payload: PipelineRunCreate) -> PipelineRun:
        validate_date(payload.predictionDate)
        run, _ = store.create_or_reuse(payload.predictionDate.isoformat())
        return PipelineRun.model_validate(run)

    @app.get("/api/v1/pipeline-runs/{run_id}", response_model=PipelineRun)
    def get_run(run_id: str) -> PipelineRun:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": "Pipeline run not found."})
        return PipelineRun.model_validate(run)

    @app.post("/api/v1/pipeline-runs/{run_id}/cancel", response_model=PipelineRun)
    def cancel_run(run_id: str) -> PipelineRun:
        run = worker.cancel(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": "Pipeline run not found."})
        return PipelineRun.model_validate(run)

    @app.get("/api/v1/pipeline-runs", response_model=list[PipelineRun])
    def list_runs(predictionDate: date | None = None, limit: int = Query(20, ge=1, le=100)) -> list[PipelineRun]:
        return [PipelineRun.model_validate(item) for item in store.list(predictionDate.isoformat() if predictionDate else None, limit)]

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"code": "validation_error", "message": "Request validation failed.", "fieldErrors": {str(error["loc"][-1]): error["msg"] for error in exc.errors()}})

    @app.exception_handler(InferenceApiError)
    async def inference_error(request: Request, exc: InferenceApiError):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(status_code=exc.status_code, content=error_body(exc, request_id))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception("Unhandled API error for request %s", request_id)
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "An unexpected error occurred.", "requestId": request_id},
        )

    for router in (model.router, regions.router, risk_map.router, predictions.router, validation.router):
        app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
