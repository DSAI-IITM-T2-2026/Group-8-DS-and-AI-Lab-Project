from __future__ import annotations

from fastapi import APIRouter

from .. import status_service
from ..feature_catalog import build_feature_catalog
from ..model_registry import get_model_registry
from ..schemas import FeatureDefinition, ModelEvaluationResponse, ModelMetadataResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metadata", response_model=ModelMetadataResponse, response_model_by_alias=True)
def get_model_metadata() -> ModelMetadataResponse:
    return ModelMetadataResponse(**status_service.model_metadata())


@router.get("/features", response_model=list[FeatureDefinition], response_model_by_alias=True)
def get_model_features() -> list[FeatureDefinition]:
    return [FeatureDefinition(**item) for item in build_feature_catalog()]


@router.get("/evaluation", response_model=ModelEvaluationResponse, response_model_by_alias=True)
def get_model_evaluation() -> ModelEvaluationResponse:
    return ModelEvaluationResponse(
        evaluation_version="milestone-5-champion-2025-v1",
        model_version=get_model_registry().version,
        split="Held-out 2025 test set",
        label_year=2025,
        rows=93_518,
        positives=1_325,
        baseline={"label": "Naive constant-rate baseline", "pr_auc": 0.0093},
        metrics=[
            {
                "key": "pr_auc", "label": "PR-AUC", "value": 0.1451,
                "display_value": "0.1451",
                "description": "Precision-recall performance under severe class imbalance.",
            },
            {
                "key": "roc_auc", "label": "ROC-AUC", "value": 0.7718,
                "display_value": "0.7718",
                "description": "Ability to rank fire and non-fire cell-days across thresholds.",
            },
            {
                "key": "recall_at_25", "label": "Recall@25", "value": 0.3638,
                "display_value": "36.38%",
                "description": "Share of positive cell-days captured by the daily Top-25 roster.",
            },
            {
                "key": "precision_at_25", "label": "Precision@25", "value": 0.0901,
                "display_value": "9.01%",
                "description": "Share of Top-25 alerts that corresponded to positive cell-days.",
            },
            {
                "key": "brier", "label": "Brier score", "value": 0.0131,
                "display_value": "0.0131",
                "description": "Mean squared error of calibrated probabilities; lower is better.",
            },
            {
                "key": "pr_auc_lift", "label": "PR-AUC lift", "value": 15.6,
                "display_value": "15.6×",
                "description": "Lift over the Milestone 5 naive PR-AUC baseline of 0.0093.",
            },
        ],
    )
