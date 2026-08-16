"""Backs /validation/events with real actual-vs-predicted records.

This needs two things this repo does not ship, by design (see
Milestone 6/Application/api/README.md):

1. A local copy of the multi-year historical archive
   (``final_processed/2019_2025/2019-2025.parquet``, ~GB-scale) that carries
   real FIRMS-derived ``y_fire`` outcomes -- point
   ``WILDFIRE_HISTORICAL_ARCHIVE`` at a copy fetched per
   ``daily_pipeline/README.md`` "Zero-download 2025 replay".
2. A loaded trained model (``WILDFIRE_MODEL_ARTIFACT``) to produce the
   "predicted" side of each record.

Without both, this endpoint answers 503 rather than inventing predicted
probabilities or outcomes.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from . import grid_catalog
from .config import get_pipeline_config
from .data_access import PreparedDataAccessError, load_historical_archive, load_observed_label_day
from .errors import ServiceUnavailableError
from .model_registry import ModelUnavailable, get_model_registry
from .risk_service import bucket_risk_class
from .risk_service import load_scored_day

def _outcome(actual: bool, predicted_alert: bool) -> str:
    if actual and predicted_alert:
        return "true_positive"
    if actual and not predicted_alert:
        return "false_negative"
    if not actual and predicted_alert:
        return "false_positive"
    return "true_negative"


def _california_today() -> date:
    cfg = get_pipeline_config()
    from config_loader import pipeline_today

    return pipeline_today(cfg)


def daily_validation(label_date: date) -> dict[str, Any]:
    registry = get_model_registry()
    if label_date >= _california_today():
        return {
            "status": "not_mature",
            "date": label_date.isoformat(),
            "model_version": registry.version,
            "message": "Observed labels are available only after the California day has ended.",
        }

    try:
        observed = load_observed_label_day(label_date)
    except PreparedDataAccessError as exc:
        raise ServiceUnavailableError(str(exc), code=exc.code) from exc
    if observed is None:
        return {
            "status": "pending",
            "date": label_date.isoformat(),
            "model_version": registry.version,
            "message": "Completed FIRMS observations are not available for this date yet.",
        }

    scored = load_scored_day(label_date)
    labels = observed.frame.copy()
    labels["cell_id"] = labels["cell_id"].astype(str)
    labels["y_fire"] = pd.to_numeric(labels["y_fire"], errors="coerce").fillna(0)
    labels = labels.sort_values("cell_id").drop_duplicates("cell_id", keep="last").set_index("cell_id")

    items: list[dict[str, Any]] = []
    true_positives = false_positives = false_negatives = 0
    for _, row in scored.frame.iterrows():
        cell_id = str(row["cell_id"])
        label = labels.loc[cell_id] if cell_id in labels.index else None
        actual = bool(label is not None and float(label["y_fire"]) > 0)
        alerted = bool(row["_alert_top_25"])
        outcome = _outcome(actual, alerted)
        true_positives += int(outcome == "true_positive")
        false_positives += int(outcome == "false_positive")
        false_negatives += int(outcome == "false_negative")

        pixel_count = label.get("firms_n_pixels") if label is not None else None
        max_confidence = label.get("firms_max_confidence") if label is not None else None
        items.append(
            {
                "area_id": cell_id,
                "actual_event": actual,
                "firms_pixel_count": None if pixel_count is None or pd.isna(pixel_count) else int(pixel_count),
                "firms_max_confidence": None if max_confidence is None or pd.isna(max_confidence) else float(max_confidence),
                "alert_top_25": alerted,
                "outcome": outcome,
            }
        )

    observed_fire_cells = true_positives + false_negatives
    top_25_count = true_positives + false_positives
    return {
        "status": "available",
        "date": label_date.isoformat(),
        "model_version": scored.model_version,
        "label_source": observed.source,
        "message": "Completed FIRMS observations were compared with this day's model roster.",
        "items": items,
        "summary": {
            "observed_fire_cells": observed_fire_cells,
            "captured_in_top_25": true_positives,
            "recall_at_25": true_positives / observed_fire_cells if observed_fire_cells else None,
            "precision_at_25": true_positives / top_25_count if top_25_count else None,
            "false_alerts": false_positives,
            "top_25_count": top_25_count,
        },
    }


def list_validation_events(
    *,
    region_id: str | None,
    start_date: date | None,
    end_date: date | None,
    model_version: str | None,
    actual_outcome: bool | None,
    predicted_class: str | None,
    cursor: int,
    limit: int,
) -> dict[str, Any]:
    archive = load_historical_archive()
    if archive is None:
        raise ServiceUnavailableError(
            "No historical archive is configured for validation events. Set "
            "WILDFIRE_HISTORICAL_ARCHIVE to a local final_processed/2019_2025/2019-2025.parquet "
            "(see daily_pipeline/README.md 'Zero-download 2025 replay').",
            code="validation_data_unavailable",
        )

    registry = get_model_registry()
    if not registry.is_loaded:
        raise ServiceUnavailableError(
            registry.unavailable_reason or "Model is not loaded.",
            code="model_unavailable",
        )

    frame = archive.copy()
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    if region_id:
        frame = frame.loc[frame["cell_id"].astype(str) == region_id]
    if start_date:
        frame = frame.loc[frame["label_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame.loc[frame["label_date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        return {"items": [], "next_cursor": None, "total": 0}

    if model_version and model_version != registry.version:
        # A different deployed model than the one currently loaded cannot be
        # honestly re-scored here.
        raise ServiceUnavailableError(
            f"Only the currently loaded model ({registry.version!r}) can be scored; "
            f"requested model_version={model_version!r} is not loaded.",
            code="model_unavailable",
        )

    try:
        scored = registry.score(frame)
    except ModelUnavailable as exc:
        raise ServiceUnavailableError(str(exc), code="model_unavailable") from exc

    frame = frame.assign(
        _predicted_probability=scored.probability,
        _alert_score=scored.alert_score,
        _alert_top_25=scored.alert_top_25,
        _actual_event=frame["y_fire"].astype(float).fillna(0) > 0,
    )
    frame["_predicted_risk_class"] = [bucket_risk_class(p) for p in frame["_alert_score"]]

    if actual_outcome is not None:
        frame = frame.loc[frame["_actual_event"] == actual_outcome]
    if predicted_class:
        frame = frame.loc[frame["_predicted_risk_class"] == predicted_class]

    total = len(frame)
    frame = frame.sort_values(["label_date", "cell_id"]).iloc[cursor : cursor + limit]

    cell_names = grid_catalog.cell_lookup(mode="all")
    items = []
    for _, row in frame.iterrows():
        actual = bool(row["_actual_event"])
        predicted_alert = bool(row["_alert_top_25"])
        label_date_str = row["label_date"].date().isoformat()
        items.append(
            {
                "id": f"validation-{row['cell_id']}-{label_date_str}",
                "date": label_date_str,
                "region_id": str(row["cell_id"]),
                "region_name": f"Grid cell {row['cell_id']}",
                "actual_event": actual,
                "actual_acres": None,  # not produced anywhere in this pipeline (FIRMS gives thermal pixels, not acreage)
                "predicted_probability": float(row["_predicted_probability"]),
                "predicted_risk_class": row["_predicted_risk_class"],
                "outcome": _outcome(actual, predicted_alert),
                "prediction_error": float(abs(float(actual) - row["_predicted_probability"])),
                "model_version": registry.version,
            }
        )

    next_cursor = str(cursor + limit) if cursor + limit < total else None
    return {"items": items, "next_cursor": next_cursor, "total": total}
