"""Serve real daily inference from the pipeline's validated test parquet."""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from . import grid_catalog
from .config import get_feature_contract
from .data_access import PreparedDataAccessError, load_prepared_day
from .errors import (
    DataUnresolvableError,
    NotFoundError,
    ServiceUnavailableError,
    UnsupportedFeatureOverrideError,
)
from .feature_catalog import build_feature_catalog
from .model_registry import ModelUnavailable, get_model_registry

RISK_MAP_MODES = {"live", "forecast_24h", "forecast_7d", "historical"}
_CACHE_LIMIT = 8


@dataclass
class ScoredDay:
    frame: pd.DataFrame
    identity: str
    model_version: str


_score_cache: OrderedDict[tuple[str, str, str], ScoredDay] = OrderedDict()
_score_cache_lock = Lock()


def clear_score_cache() -> None:
    with _score_cache_lock:
        _score_cache.clear()


def bucket_risk_class(alert_score: float) -> str:
    """Display-only quintiles for the model's within-day priority score."""
    if alert_score < 0.2:
        return "very_low"
    if alert_score < 0.4:
        return "low"
    if alert_score < 0.6:
        return "moderate"
    if alert_score < 0.8:
        return "high"
    return "very_high"


def resolve_label_date(mode: str, timestamp: datetime) -> date:
    if mode == "forecast_7d":
        raise UnsupportedFeatureOverrideError(
            "mode=forecast_7d is not supported: the champion pipeline has a one-day lead.",
            code="unsupported_mode",
        )
    return timestamp.date()


def _feature_catalog_by_key() -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in build_feature_catalog()}


def _observed_at(key: str, source: str, row: pd.Series) -> datetime | None:
    if source in ("sentinel2-surface-reflectance", "sentinel5p-atmosphere"):
        column = "eo_asof_date"
    elif source == "era5-reanalysis":
        column = "feature_end_date"
    elif source == "firms-neighbor-context":
        column = "label_date"
    else:
        return None
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.to_pydatetime()


def build_feature_snapshot(row: pd.Series) -> list[dict[str, Any]]:
    catalog = _feature_catalog_by_key()
    values = []
    for key, metadata in catalog.items():
        if key not in row.index or pd.isna(row[key]):
            continue
        values.append(
            {
                "key": key,
                "display_name": metadata["display_name"],
                "value": float(row[key]),
                "unit": metadata["unit"],
                "source": metadata["source"],
                "observed_at": _observed_at(key, metadata["source"], row),
            }
        )
    return values


def validate_prepared_frame(frame: pd.DataFrame, label_date: date, feature_columns: list[str]) -> pd.DataFrame:
    """Enforce the preparation-to-inference contract without rebuilding features."""
    if frame.empty:
        raise DataUnresolvableError("The prepared parquet is empty.", code="invalid_feature_data")

    contract_features = list(get_feature_contract()["feature_prune"]["kept_features"])
    if len(feature_columns) != 86 or feature_columns != contract_features:
        raise ServiceUnavailableError(
            "The loaded model does not match the frozen 86-feature contract.",
            code="model_contract_mismatch",
        )
    required = {"cell_id", "label_date", "eo_asof_date", "feature_end_date", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataUnresolvableError(
            f"The prepared parquet is missing required columns: {missing[:5]}.",
            code="invalid_feature_data",
        )

    table = frame.copy().reset_index(drop=True)
    for column in ("label_date", "eo_asof_date", "feature_end_date"):
        table[column] = pd.to_datetime(table[column], errors="coerce").dt.normalize()
        if table[column].isna().any():
            raise DataUnresolvableError(
                f"The prepared parquet contains invalid {column} values.",
                code="invalid_feature_data",
            )

    expected_label = pd.Timestamp(label_date)
    expected_eo = expected_label - timedelta(days=1)
    expected_feature_end = expected_label - timedelta(days=6)
    if set(table["label_date"].unique()) != {expected_label}:
        raise DataUnresolvableError(
            "The prepared parquet does not contain exactly the requested label date.",
            code="invalid_feature_data",
        )
    if not table["eo_asof_date"].eq(expected_eo).all():
        raise DataUnresolvableError(
            "The prepared parquet has an invalid EO as-of date.", code="invalid_feature_data"
        )
    if not table["feature_end_date"].eq(expected_feature_end).all():
        raise DataUnresolvableError(
            "The prepared parquet has an invalid ERA5 feature-end date.",
            code="invalid_feature_data",
        )

    table["cell_id"] = table["cell_id"].astype(str)
    if table["cell_id"].duplicated().any():
        raise DataUnresolvableError(
            "The prepared parquet contains duplicate grid cells.", code="invalid_feature_data"
        )
    supported = set(grid_catalog.cell_lookup().keys())
    if not supported or not set(table["cell_id"]).issubset(supported):
        raise DataUnresolvableError(
            "The prepared parquet contains an unsupported grid-cell set.",
            code="invalid_feature_data",
        )
    try:
        numeric = table[feature_columns].apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise DataUnresolvableError(
            "The prepared parquet contains non-numeric model features.",
            code="invalid_feature_data",
        ) from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataUnresolvableError(
            "The prepared parquet contains missing or non-finite model features.",
            code="invalid_feature_data",
        )
    table[feature_columns] = numeric
    return table


def load_scored_day(label_date: date) -> ScoredDay:
    registry = get_model_registry()
    if not registry.is_loaded:
        raise ServiceUnavailableError(
            registry.unavailable_reason or "The model is not loaded.", code="model_unavailable"
        )
    try:
        prepared = load_prepared_day(label_date)
    except PreparedDataAccessError as exc:
        raise ServiceUnavailableError(str(exc), code=exc.code) from exc
    if prepared is None:
        raise DataUnresolvableError(
            f"No prepared feature parquet is available for {label_date.isoformat()}.",
            code="feature_data_unavailable",
            field_errors={"timestamp": "Prepare this prediction date before scoring it."},
        )
    key = (label_date.isoformat(), prepared.identity, registry.version)
    with _score_cache_lock:
        cached = _score_cache.get(key)
        if cached is not None:
            _score_cache.move_to_end(key)
            return cached

    table = validate_prepared_frame(prepared.frame, label_date, registry.feature_columns)
    try:
        scored = registry.score(table)
    except ModelUnavailable as exc:
        raise ServiceUnavailableError(str(exc), code="model_unavailable") from exc
    table = table.assign(
        _raw_probability=scored.raw_probability,
        _probability=scored.probability,
        _rank_score=scored.rank_score,
        _alert_score=scored.alert_score,
        _priority_rank=scored.priority_rank,
        _alert_top_25=scored.alert_top_25,
    )
    result = ScoredDay(table, prepared.identity, registry.version)
    with _score_cache_lock:
        _score_cache[key] = result
        _score_cache.move_to_end(key)
        while len(_score_cache) > _CACHE_LIMIT:
            _score_cache.popitem(last=False)
    return result


def _risk_fields(row: pd.Series) -> dict[str, Any]:
    alert_score = float(row["_alert_score"])
    return {
        "probability": float(row["_probability"]),
        "raw_probability": float(row["_raw_probability"]),
        "alert_score": alert_score,
        "priority_rank": int(row["_priority_rank"]),
        "alert_top_25": bool(row["_alert_top_25"]),
        "risk_class": bucket_risk_class(alert_score),
    }


def build_risk_map(region_id: str, mode: str, timestamp: datetime) -> dict[str, Any]:
    if mode not in RISK_MAP_MODES:
        raise UnsupportedFeatureOverrideError(f"Unsupported mode: {mode!r}", code="unsupported_mode")
    region = grid_catalog.get_region_or_404(region_id)
    label_date = resolve_label_date(mode, timestamp)
    scored_day = load_scored_day(label_date)
    items = []
    for _, row in scored_day.frame.sort_values("_priority_rank").iterrows():
        cell_id = str(row["cell_id"])
        items.append(
            {
                "area_id": cell_id,
                "area_name": f"Grid cell {cell_id}",
                **_risk_fields(row),
                "updated_at": timestamp,
            }
        )
    return {
        "region_id": region["id"],
        "timestamp": timestamp,
        "geometry_version": region["geometry_id"],
        "items": items,
    }


def _require_feature_row(frame: pd.DataFrame, cell_id: str, label_date: date) -> pd.Series:
    subset = frame.loc[frame["cell_id"].astype(str) == cell_id]
    if subset.empty:
        raise NotFoundError(
            f"Grid cell {cell_id!r} is not scored for {label_date.isoformat()}.",
            code="area_not_scored",
        )
    return subset.iloc[0]


def _apply_overrides(frame: pd.DataFrame, cell_id: str, overrides: dict[str, float]) -> pd.DataFrame:
    table = frame.copy()
    catalog = _feature_catalog_by_key()
    field_errors: dict[str, str] = {}
    for key, value in overrides.items():
        metadata = catalog.get(key)
        if metadata is None:
            field_errors[key] = "Unknown feature key."
        elif not metadata["editable_in_scenario"]:
            field_errors[key] = "This feature cannot be overridden in a scenario."
        elif metadata["min"] is not None and value < metadata["min"]:
            field_errors[key] = f"Below minimum ({metadata['min']})."
        elif metadata["max"] is not None and value > metadata["max"]:
            field_errors[key] = f"Above maximum ({metadata['max']})."
        else:
            table.loc[table["cell_id"] == cell_id, key] = float(value)
    if field_errors:
        raise UnsupportedFeatureOverrideError(
            "One or more feature overrides are invalid.",
            code="invalid_feature_override",
            field_errors=field_errors,
        )
    return table


def run_prediction(
    region_id: str,
    mode: str,
    timestamp: datetime,
    *,
    feature_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    if region_id not in grid_catalog.cell_lookup(mode="all"):
        raise NotFoundError(f"Unknown grid cell: {region_id!r}", code="unknown_area")
    label_date = resolve_label_date("live" if mode == "scenario" else mode, timestamp)
    scored_day = load_scored_day(label_date)
    base_row = _require_feature_row(scored_day.frame, region_id, label_date)
    result_row = base_row

    if feature_overrides:
        scenario_frame = _apply_overrides(scored_day.frame, region_id, feature_overrides)
        registry = get_model_registry()
        try:
            scored = registry.score(scenario_frame)
        except ModelUnavailable as exc:
            raise ServiceUnavailableError(str(exc), code="model_unavailable") from exc
        scenario_frame = scenario_frame.assign(
            _raw_probability=scored.raw_probability,
            _probability=scored.probability,
            _rank_score=scored.rank_score,
            _alert_score=scored.alert_score,
            _priority_rank=scored.priority_rank,
            _alert_top_25=scored.alert_top_25,
        )
        result_row = _require_feature_row(scenario_frame, region_id, label_date)

    registry = get_model_registry()
    explanation = None
    try:
        explained = registry.explain(pd.DataFrame([result_row]))
    except ModelUnavailable:
        explained = None
    if explained is not None:
        catalog = _feature_catalog_by_key()
        pairs = sorted(
            zip(explained["feature_columns"], explained["contributions"][0]),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:10]
        explanation = {
            "confidence": float(max(result_row["_probability"], 1 - result_row["_probability"])),
            "feature_importance": [
                {
                    "feature": feature,
                    "display_name": catalog.get(feature, {}).get("display_name", feature),
                    "importance": float(abs(value)),
                }
                for feature, value in pairs
            ],
            "contributions": [
                {
                    "feature": feature,
                    "display_name": catalog.get(feature, {}).get("display_name", feature),
                    "contribution": float(value),
                }
                for feature, value in pairs
            ],
        }

    data_timestamp = pd.Timestamp(result_row["eo_asof_date"])
    if data_timestamp.tzinfo is None:
        data_timestamp = data_timestamp.tz_localize("UTC")
    return {
        "prediction_id": f"pred-{uuid.uuid4().hex[:16]}",
        "region_id": region_id,
        "timestamp": timestamp,
        "inference_mode": mode,
        **_risk_fields(result_row),
        "model_version": registry.version,
        "data_timestamp": data_timestamp.to_pydatetime(),
        "feature_snapshot": {"values": build_feature_snapshot(result_row)},
        "explanation": explanation,
    }
