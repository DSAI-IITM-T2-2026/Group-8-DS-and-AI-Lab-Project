from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from .. import risk_service
from ..schemas import RiskMapResponse

router = APIRouter(tags=["risk-map"])


@router.get("/risk-map", response_model=RiskMapResponse, response_model_by_alias=True)
def get_risk_map(
    region_id: str = Query(...),
    timestamp: datetime = Query(...),
    mode: str = Query("live"),
) -> RiskMapResponse:
    return RiskMapResponse(**risk_service.build_risk_map(region_id, mode, timestamp))
