from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query

from .. import validation_service
from ..schemas import ValidationEventsResponse

router = APIRouter(prefix="/validation", tags=["validation"])


@router.get("/events", response_model=ValidationEventsResponse, response_model_by_alias=True)
def get_validation_events(
    region_id: Optional[str] = Query(default=None),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    model_version: Optional[str] = Query(default=None),
    actual_outcome: Optional[bool] = Query(default=None),
    predicted_class: Optional[str] = Query(default=None),
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> ValidationEventsResponse:
    result = validation_service.list_validation_events(
        region_id=region_id,
        start_date=start_date,
        end_date=end_date,
        model_version=model_version,
        actual_outcome=actual_outcome,
        predicted_class=predicted_class,
        cursor=cursor,
        limit=limit,
    )
    return ValidationEventsResponse(**result)
