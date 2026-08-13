"""FastAPI app implementing frontend-backend-api-contract.md.

See Milestone 6/Application/api/README.md for exactly which endpoints are backed by
real pipeline data today and which return a documented 503 because a
dependency (trained model artifact, GCS credentials, historical archive)
is not part of this repository.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .errors import ApiError, error_body
from .routers import health, model, predictions, regions, risk_map, validation

logger = logging.getLogger("wildfire_api")

settings = get_settings()

app = FastAPI(title=settings.api_title, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(status_code=exc.status_code, content=error_body(exc, request_id))


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    field_errors = {}
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []) if part not in {"body", "query"})
        field_errors[loc or "request"] = err.get("msg", "Invalid value.")
    return JSONResponse(
        status_code=400,
        content={
            "code": "bad_request",
            "message": "The request could not be parsed.",
            "requestId": request_id,
            "fieldErrors": field_errors,
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception("Unhandled error for request %s", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "An unexpected error occurred.",
            "requestId": request_id,
        },
    )


for router in (health.router, model.router, regions.router, risk_map.router, predictions.router, validation.router):
    app.include_router(router, prefix=settings.api_base_path)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.api_title, "apiBasePath": settings.api_base_path}
