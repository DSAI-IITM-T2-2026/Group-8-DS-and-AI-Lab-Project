from __future__ import annotations

from fastapi import APIRouter

from .. import status_service
from ..feature_catalog import build_feature_catalog
from ..schemas import FeatureDefinition, ModelMetadataResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metadata", response_model=ModelMetadataResponse, response_model_by_alias=True)
def get_model_metadata() -> ModelMetadataResponse:
    return ModelMetadataResponse(**status_service.model_metadata())


@router.get("/features", response_model=list[FeatureDefinition], response_model_by_alias=True)
def get_model_features() -> list[FeatureDefinition]:
    return [FeatureDefinition(**item) for item in build_feature_catalog()]
