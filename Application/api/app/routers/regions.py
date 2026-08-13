from __future__ import annotations

from fastapi import APIRouter

from .. import grid_catalog
from ..schemas import RegionDetail, RegionGeometryResponse, RegionSummary

router = APIRouter(prefix="/regions", tags=["regions"])


@router.get("", response_model=list[RegionSummary], response_model_by_alias=True)
def list_regions() -> list[RegionSummary]:
    return [RegionSummary(**grid_catalog.region_summary())]


@router.get("/{region_id}", response_model=RegionDetail, response_model_by_alias=True)
def get_region(region_id: str) -> RegionDetail:
    return RegionDetail(**grid_catalog.get_region_or_404(region_id))


@router.get("/{region_id}/geometry", response_model=RegionGeometryResponse, response_model_by_alias=True)
def get_region_geometry(region_id: str) -> RegionGeometryResponse:
    return RegionGeometryResponse(**grid_catalog.region_geometry(region_id))
