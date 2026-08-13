"""Orchestrates /risk-map and /predictions on top of real pipeline output.

Both endpoints need the same three real ingredients for a given
(region/cell, label date):

1. The 86-feature row(s) from ``final_processed/<date>_test.parquet``
   (``data_access.load_final_processed``).
2. A loaded trained model to turn features into a probability
   (``model_registry``) -- if none is configured, both endpoints answer
   503 rather than inventing a number.
3. A riskClass bucketing rule. The champion contract does not define a
   fixed probability threshold (see ``daily_blend`` percentiles in
   ``champion_86_features.json`` and ``model.py``'s docstring), so this
   module applies one explicit, documented heuristic band scheme to keep
   ``/risk-map`` and ``/predictions`` consistent with each other. It is a
   deployment-tunable design choice, not a value pulled from training.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from . import grid_catalog
from .data_access import load_final_processed
from .errors import DataUnresolvableError, NotFoundError, ServiceUnavailableError, UnsupportedFeatureOverrideError
from .feature_catalog import build_feature_catalog
from .model_registry import ModelUnavailable, get_model_registry

RISK_MAP_MODES = {"live", "forecast_24h", "forecast_7d", "historical"}

# Fixed probability bands -- see module docstring. Tune per ops guidance;
# these are not derived from the training run.
_RISK_BANDS = (
    (0.02, "very_low"),
    (0.05, "low"),
    (0.15, "moderate"),
    (0.35, "high"),
)


def bucket_risk_class(probability: float) -> str:
    for upper, label in _RISK_BANDS:
        if probability < upper:
            return label
    return "very_high"


def resolve_label_date(mode: str, timestamp: datetime) -> date:
    if mode == "forecast_7d":
        raise UnsupportedFeatureOverrideError(
            "mode=forecast_7d is not supported by this deployment: the champion pipeline "
            "is trained/configured for a 1-day-ahead lead time only (task.lead_days=1 in "
            "daily_pipeline/utils/config.yaml). Use forecast_24h.",
            code="unsupported_mode",
        )
    return timestamp.date()


def _feature_catalog_by_key() -> dict[str, dict[str, Any]]:
    return {f["key"]: f for f in build_feature_catalog()}


def _observed_at(key: str, source: str, row: pd.Series) -> datetime | None:
    """Real as-of date for a feature value, from the row's own causal-lag columns."""
    if source in ("sentinel2-surface-reflectance", "sentinel5p-atmosphere"):
        col = "eo_asof_date"
    elif source == "era5-reanalysis":
        col = "feature_end_date"
    elif source == "firms-neighbor-context":
        col = "label_date"  # y_fire lag2 is relative to label_date; exact date not stored per-row
    else:
        return None
    value = row.get(col)
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def build_feature_snapshot(row: pd.Series) -> list[dict[str, Any]]:
    catalog = _feature_catalog_by_key()
    values = []
    for key, meta in catalog.items():
        if key not in row.index or pd.isna(row[key]):
            continue
        values.append(
            {
                "key": key,
                "display_name": meta["display_name"],
                "value": float(row[key]),
                "unit": meta["unit"],
                "source": meta["source"],
                "observed_at": _observed_at(key, meta["source"], row),
            }
        )
    return values


def _require_feature_row(frame: pd.DataFrame, cell_id: str, label_date: date) -> pd.Series:
    subset = frame.loc[frame["cell_id"].astype(str) == cell_id]
    if subset.empty:
        raise NotFoundError(
            f"No scored feature row for cell {cell_id!r} on {label_date.isoformat()}. "
            "This cell may fall outside the configured cell_subset for that day.",
            code="area_not_scored",
        )
    return subset.iloc[0]


def load_scored_day(label_date: date) -> pd.DataFrame:
    frame = load_final_processed(label_date)
    if frame is None:
        raise DataUnresolvableError(
            f"No final_processed feature table for {label_date.isoformat()}. Generate it with "
            f"`python run_daily.py all --label-date {label_date.isoformat()}` in daily_pipeline/, "
            "or configure GCS access so the API can read it from the bucket.",
            code="feature_data_unavailable",
            field_errors={"timestamp": "Timestamp is outside the available range."},
        )
    return frame


def build_risk_map(region_id: str, mode: str, timestamp: datetime) -> dict[str, Any]:
    if mode not in RISK_MAP_MODES:
        raise UnsupportedFeatureOverrideError(f"Unsupported mode: {mode!r}", code="unsupported_mode")
    region = grid_catalog.get_region_or_404(region_id)
    label_date = resolve_label_date(mode, timestamp)
    frame = load_scored_day(label_date)

    registry = get_model_registry()
    try:
        scored = registry.score(frame)
    except ModelUnavailable as exc:
        raise ServiceUnavailableError(str(exc), code="model_unavailable") from exc

    names = dict(zip(frame["cell_id"].astype(str), frame["cell_id"].astype(str)))
    items = []
    for cell_id, probability in zip(frame["cell_id"].astype(str), scored.probability):
        items.append(
            {
                "area_id": cell_id,
                "area_name": f"Grid cell {names[cell_id]}",
                "probability": float(probability),
                "risk_class": bucket_risk_class(float(probability)),
                "updated_at": timestamp,
            }
        )
    return {
        "region_id": region["id"],
        "timestamp": timestamp,
        "geometry_version": region["geometry_id"],
        "items": items,
    }


def run_prediction(
    region_id: str,
    mode: str,
    timestamp: datetime,
    *,
    feature_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    cells = grid_catalog.cell_lookup(mode="all")
    if region_id not in cells:
        raise NotFoundError(f"Unknown regionId (grid cell): {region_id!r}", code="unknown_area")

    label_date = resolve_label_date("live" if mode == "scenario" else mode, timestamp)
    frame = load_scored_day(label_date)
    row = _require_feature_row(frame, region_id, label_date).copy()

    if feature_overrides:
        catalog = _feature_catalog_by_key()
        field_errors: dict[str, str] = {}
        for key, value in feature_overrides.items():
            meta = catalog.get(key)
            if meta is None:
                field_errors[key] = "Unknown feature key."
                continue
            if not meta["editable_in_scenario"]:
                field_errors[key] = "This feature cannot be overridden in a scenario."
                continue
            if meta["min"] is not None and value < meta["min"]:
                field_errors[key] = f"Below minimum ({meta['min']})."
                continue
            if meta["max"] is not None and value > meta["max"]:
                field_errors[key] = f"Above maximum ({meta['max']})."
                continue
            row[key] = float(value)
        if field_errors:
            raise UnsupportedFeatureOverrideError(
                "One or more feature overrides are invalid.",
                code="invalid_feature_override",
                field_errors=field_errors,
            )

    registry = get_model_registry()
    try:
        scored = registry.score(pd.DataFrame([row]))
    except ModelUnavailable as exc:
        raise ServiceUnavailableError(str(exc), code="model_unavailable") from exc
    probability = float(scored.probability[0])

    explanation = None
    try:
        explained = registry.explain(pd.DataFrame([row]))
    except ModelUnavailable:
        explained = None
    if explained is not None:
        contrib = explained["contributions"][0]
        pairs = sorted(
            zip(explained["feature_columns"], contrib),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:10]
        catalog = _feature_catalog_by_key()
        explanation = {
            "confidence": float(max(probability, 1 - probability)),
            "feature_importance": [
                {
                    "feature": feat,
                    "display_name": catalog.get(feat, {}).get("display_name", feat),
                    "importance": float(abs(value)),
                }
                for feat, value in pairs
            ],
            "contributions": [
                {
                    "feature": feat,
                    "display_name": catalog.get(feat, {}).get("display_name", feat),
                    "contribution": float(value),
                }
                for feat, value in pairs
            ],
        }

    data_timestamp = row.get("eo_asof_date")
    data_timestamp = (
        pd.Timestamp(data_timestamp).tz_localize("UTC").to_pydatetime()
        if data_timestamp is not None and not pd.isna(data_timestamp)
        else datetime.now(timezone.utc)
    )

    return {
        "prediction_id": f"pred-{uuid.uuid4().hex[:16]}",
        "region_id": region_id,
        "timestamp": timestamp,
        "inference_mode": mode,
        "probability": probability,
        "risk_class": bucket_risk_class(probability),
        "model_version": registry.version,
        "data_timestamp": data_timestamp,
        "feature_snapshot": {"values": build_feature_snapshot(row)},
        "explanation": explanation,
    }
