"""Backs /validation/events with real actual-vs-predicted records.

This needs two things this repo does not ship, by design (see
Milestone 6/api/README.md):

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
from .data_access import load_historical_archive
from .errors import ServiceUnavailableError
from .model_registry import ModelUnavailable, get_model_registry
from .risk_service import bucket_risk_class

# A predicted "alert" (used only to classify true/false positive/negative)
# is any cell bucketed high or very_high by the same fixed bands used for
# /risk-map and /predictions -- see risk_service module docstring.
_ALERT_CLASSES = {"high", "very_high"}


def _outcome(actual: bool, predicted_alert: bool) -> str:
    if actual and predicted_alert:
        return "true_positive"
    if actual and not predicted_alert:
        return "false_negative"
    if not actual and predicted_alert:
        return "false_positive"
    return "true_negative"


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
        _actual_event=frame["y_fire"].astype(float).fillna(0) > 0,
    )
    frame["_predicted_risk_class"] = [bucket_risk_class(p) for p in frame["_predicted_probability"]]

    if actual_outcome is not None:
        frame = frame.loc[frame["_actual_event"] == actual_outcome]
    if predicted_class:
        frame = frame.loc[frame["_predicted_risk_class"] == predicted_class]

    total = len(frame)
    frame = frame.sort_values(["label_date", "cell_id"]).iloc[cursor : cursor + limit]

    cell_names = grid_catalog.cell_lookup(mode="all")
    items = []
    for row in frame.itertuples():
        actual = bool(row._actual_event)
        predicted_alert = row._predicted_risk_class in _ALERT_CLASSES
        label_date_str = row.label_date.date().isoformat()
        items.append(
            {
                "id": f"validation-{row.cell_id}-{label_date_str}",
                "date": label_date_str,
                "region_id": str(row.cell_id),
                "region_name": f"Grid cell {row.cell_id}",
                "actual_event": actual,
                "actual_acres": None,  # not produced anywhere in this pipeline (FIRMS gives thermal pixels, not acreage)
                "predicted_probability": float(row._predicted_probability),
                "predicted_risk_class": row._predicted_risk_class,
                "outcome": _outcome(actual, predicted_alert),
                "prediction_error": float(abs(float(actual) - row._predicted_probability)),
                "model_version": registry.version,
            }
        )

    next_cursor = str(cursor + limit) if cursor + limit < total else None
    return {"items": items, "next_cursor": next_cursor, "total": total}
