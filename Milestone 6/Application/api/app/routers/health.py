from __future__ import annotations

from fastapi import APIRouter

from .. import status_service
from ..model_registry import get_model_registry
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, response_model_by_alias=True)
def get_health() -> HealthResponse:
    info = status_service.health_checks()
    registry = get_model_registry()
    return HealthResponse(
        status=info["status"],
        model_loaded=registry.is_loaded,
        model_version=registry.version,
        checks=info["checks"],
    )
