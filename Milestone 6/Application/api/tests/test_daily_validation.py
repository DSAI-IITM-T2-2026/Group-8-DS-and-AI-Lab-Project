from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from api.app import validation_service
from api.app.data_access import ObservedLabelDay, PreparedDataAccessError
from api.app.errors import ServiceUnavailableError
from api.app.risk_service import ScoredDay


class Registry:
    version = "model-v1"


def scored_day(actual_date: date = date(2025, 8, 1)) -> ScoredDay:
    frame = pd.DataFrame(
        {
            "cell_id": ["cell-a", "cell-b", "cell-c", "cell-d"],
            "label_date": pd.Timestamp(actual_date),
            "_alert_top_25": [True, True, False, False],
        }
    )
    return ScoredDay(frame=frame, identity="artifact-v1", model_version="model-v1")


def configure(monkeypatch, *, today=date(2025, 8, 3)):
    monkeypatch.setattr(validation_service, "_california_today", lambda: today)
    monkeypatch.setattr(validation_service, "get_model_registry", lambda: Registry())


def test_not_mature_does_not_load_labels_or_score(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(
        validation_service,
        "load_observed_label_day",
        lambda _: (_ for _ in ()).throw(AssertionError("labels must not load")),
    )
    monkeypatch.setattr(
        validation_service,
        "load_scored_day",
        lambda _: (_ for _ in ()).throw(AssertionError("model must not score")),
    )

    result = validation_service.daily_validation(date(2025, 8, 3))
    assert result["status"] == "not_mature"
    assert "items" not in result


def test_missing_matured_labels_are_pending_without_scoring(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(validation_service, "load_observed_label_day", lambda _: None)
    monkeypatch.setattr(
        validation_service,
        "load_scored_day",
        lambda _: (_ for _ in ()).throw(AssertionError("model must not score")),
    )

    result = validation_service.daily_validation(date(2025, 8, 1))
    assert result["status"] == "pending"


def test_daily_capture_metrics_and_all_outcomes_use_scored_day(monkeypatch):
    configure(monkeypatch)
    labels = pd.DataFrame(
        {
            "cell_id": ["cell-a", "cell-c"],
            "y_fire": [1, 1],
            "firms_n_pixels": [3, 1],
            "firms_max_confidence": [88.0, 61.0],
        }
    )
    monkeypatch.setattr(
        validation_service,
        "load_observed_label_day",
        lambda _: ObservedLabelDay(labels, "historical_archive", "archive-day"),
    )
    score_calls = []
    monkeypatch.setattr(
        validation_service,
        "load_scored_day",
        lambda value: score_calls.append(value) or scored_day(value),
    )

    result = validation_service.daily_validation(date(2025, 8, 1))

    assert score_calls == [date(2025, 8, 1)]
    assert result["status"] == "available"
    assert result["summary"] == {
        "observed_fire_cells": 2,
        "captured_in_top_25": 1,
        "recall_at_25": 0.5,
        "precision_at_25": 0.5,
        "false_alerts": 1,
        "top_25_count": 2,
    }
    assert {item["area_id"]: item["outcome"] for item in result["items"]} == {
        "cell-a": "true_positive",
        "cell-b": "false_positive",
        "cell-c": "false_negative",
        "cell-d": "true_negative",
    }
    assert result["items"][0]["firms_pixel_count"] == 3


def test_zero_positive_day_has_undefined_recall(monkeypatch):
    configure(monkeypatch)
    labels = pd.DataFrame(columns=["cell_id", "y_fire", "firms_n_pixels", "firms_max_confidence"])
    monkeypatch.setattr(
        validation_service,
        "load_observed_label_day",
        lambda _: ObservedLabelDay(labels, "firms_daily_geotiff", "gcs-day"),
    )
    monkeypatch.setattr(validation_service, "load_scored_day", scored_day)

    summary = validation_service.daily_validation(date(2025, 8, 1))["summary"]
    assert summary["observed_fire_cells"] == 0
    assert summary["recall_at_25"] is None
    assert summary["precision_at_25"] == 0.0


def test_label_source_failure_is_not_treated_as_zero_fire(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(
        validation_service,
        "load_observed_label_day",
        lambda _: (_ for _ in ()).throw(
            PreparedDataAccessError("invalid_validation_data", "FIRMS raster is corrupt.")
        ),
    )

    with pytest.raises(ServiceUnavailableError) as caught:
        validation_service.daily_validation(date(2025, 8, 1))
    assert caught.value.code == "invalid_validation_data"
