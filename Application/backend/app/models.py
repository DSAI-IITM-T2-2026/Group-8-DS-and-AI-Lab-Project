from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal["queued", "running", "waiting_external", "succeeded", "failed", "interrupted"]
PipelineStage = Literal[
    "validating", "inventory", "era5", "firms", "sentinel2", "sentinel5p",
    "preprocessing", "exporting", "completed",
]


class PipelineRunCreate(BaseModel):
    predictionDate: date


class SourceInventoryItem(BaseModel):
    required: int = 0
    available: int = 0
    missing: int = 0
    scheduled: int = 0
    pending: int = 0


class ArtifactSummary(BaseModel):
    objectUri: str
    rowCount: int
    featureCount: int
    cellCount: int
    labelDate: date
    eoAsOfDate: date
    featureEndDate: date
    createdAt: datetime


class PipelineRun(BaseModel):
    runId: str
    predictionDate: date
    status: RunStatus
    stage: PipelineStage
    message: str
    progressCompleted: int = 0
    progressTotal: int = 0
    sourceInventory: dict[str, SourceInventoryItem] = Field(default_factory=dict)
    artifact: ArtifactSummary | None = None
    errorCode: str | None = None
    createdAt: datetime
    startedAt: datetime | None = None
    finishedAt: datetime | None = None


class PipelineConfig(BaseModel):
    minPredictionDate: date
    maxPredictionDate: date
    timezone: str
    lookbackDays: int
    expectedFeatureCount: int


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    api: bool
    worker: bool
    pipelineConfigured: bool
    gcsCredentialsConfigured: bool
    earthEngineCredentialsConfigured: bool
    modelLoaded: bool
    modelVersion: str
    inferenceChecks: dict[str, str] = Field(default_factory=dict)
