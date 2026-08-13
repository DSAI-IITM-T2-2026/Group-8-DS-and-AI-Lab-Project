from __future__ import annotations

from fastapi import APIRouter

from .. import risk_service
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["predictions"])


@router.post("/predictions", response_model=PredictionResponse, response_model_by_alias=True)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    result = risk_service.run_prediction(
        request.region_id,
        request.mode,
        request.timestamp,
        feature_overrides=request.feature_overrides,
    )
    return PredictionResponse(**result)
