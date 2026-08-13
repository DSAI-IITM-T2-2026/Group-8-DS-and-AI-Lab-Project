"""Pydantic response/request models mirroring frontend-backend-api-contract.md section 3-4.

Deliberately uses ``typing.Optional``/``Union`` instead of the ``X | None``
PEP 604 syntax: pydantic v2 evaluates field annotations eagerly, and ``X |
None`` needs Python 3.10+ (or the ``eval_type_backport`` package) to resolve
at runtime. Keeping this file 3.9-compatible avoids that extra dependency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from typing_extensions import Literal

RiskClass = Literal["very_low", "low", "moderate", "high", "very_high"]
PredictionMode = Literal["live", "forecast_24h", "forecast_7d", "historical", "scenario"]
FreshnessStatus = Literal["fresh", "stale", "partial", "unavailable"]
ExplanationCapability = Literal["available", "unavailable", "unknown"]
RegionAvailability = Literal["supported", "experimental", "validation_only", "unavailable"]
ValidationOutcome = Literal["true_positive", "true_negative", "false_positive", "false_negative"]


class ApiModel(BaseModel):
    """Base model: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, protected_namespaces=())


# --- 3.1 Health -------------------------------------------------------------


class HealthResponse(ApiModel):
    status: Literal["ready", "degraded", "unavailable"]
    model_loaded: bool
    model_version: str
    checks: Dict[str, str] = Field(default_factory=dict)


# --- 3.2 Model metadata -------------------------------------------------------


class DataFreshness(ApiModel):
    status: FreshnessStatus
    observed_at: Optional[datetime] = None


class ModelMetadataResponse(ApiModel):
    model_version: str
    threshold: Optional[float] = None
    updated_at: datetime
    explanation_capability: ExplanationCapability
    data_freshness: DataFreshness
    provenance: Literal["model"] = "model"


# --- 3.3 Feature metadata -----------------------------------------------------


class FeatureDefinition(ApiModel):
    key: str
    display_name: str
    type: Literal["number", "boolean"] = "number"
    unit: Optional[str] = None
    source: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    default_value: Optional[float] = None
    editable_in_scenario: bool


# --- 3.4 / 3.5 Regions ---------------------------------------------------------


class RegionSummary(ApiModel):
    id: str
    name: str
    country: str
    region_type: str
    geometry_id: str
    availability: RegionAvailability


class RegionDetail(RegionSummary):
    center: Tuple[float, float]
    bounds: Tuple[float, float, float, float]


# --- 3.6 Region geometry -------------------------------------------------------


class RegionGeometryResponse(ApiModel):
    region_id: str
    geometry_version: str
    geojson: dict


# --- 3.7 Risk map ---------------------------------------------------------------


class RiskMapItem(ApiModel):
    area_id: str
    area_name: str
    probability: float
    risk_class: RiskClass
    updated_at: datetime


class RiskMapResponse(ApiModel):
    region_id: str
    timestamp: datetime
    geometry_version: str
    items: List[RiskMapItem]
    provenance: Literal["model"] = "model"


# --- 3.8 Predictions --------------------------------------------------------------


class PredictionRequest(ApiModel):
    region_id: str
    timestamp: datetime
    mode: PredictionMode
    feature_overrides: Optional[Dict[str, float]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class FeatureSnapshotValue(ApiModel):
    key: str
    display_name: str
    value: float
    unit: Optional[str] = None
    source: str
    observed_at: Optional[datetime] = None


class FeatureSnapshot(ApiModel):
    values: List[FeatureSnapshotValue]


class FeatureImportanceEntry(ApiModel):
    feature: str
    display_name: str
    importance: float


class ContributionEntry(ApiModel):
    feature: str
    display_name: str
    contribution: float


class Explanation(ApiModel):
    confidence: float
    feature_importance: List[FeatureImportanceEntry]
    contributions: List[ContributionEntry]
    provenance: Literal["model"] = "model"


class PredictionResponse(ApiModel):
    prediction_id: str
    region_id: str
    timestamp: datetime
    inference_mode: PredictionMode
    probability: float
    risk_class: RiskClass
    threshold: Optional[float] = None
    model_version: str
    data_timestamp: datetime
    feature_snapshot: FeatureSnapshot
    explanation: Optional[Explanation] = None
    provenance: Literal["model"] = "model"


# --- 3.9 Validation events -----------------------------------------------------------


class ValidationEvent(ApiModel):
    id: str
    date: str
    region_id: str
    region_name: str
    actual_event: bool
    actual_acres: Optional[float] = None
    predicted_probability: float
    predicted_risk_class: RiskClass
    outcome: ValidationOutcome
    prediction_error: float
    model_version: str


class ValidationEventsResponse(ApiModel):
    items: List[ValidationEvent]
    next_cursor: Optional[str] = None
    total: int
    provenance: Literal["model"] = "model"
