from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from api.app import risk_service
from api.app.data_access import PreparedDay
from api.app.errors import DataUnresolvableError
from api.app.model_registry import ModelRegistry, ScoredBatch, within_day_percentile


FEATURES = [f"feature_{index}" for index in range(86)]


class Classifier:
    def predict_proba(self, frame):
        probability = frame[FEATURES[0]].to_numpy(dtype=float)
        return np.column_stack([1 - probability, probability])


class Calibrator:
    def predict_proba(self, logits):
        raw = 1 / (1 + np.exp(-np.asarray(logits)[:, 0]))
        calibrated = np.clip(raw * 0.5, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


class Ranker:
    def predict(self, frame):
        return frame[FEATURES[1]].to_numpy(dtype=float)


def registry() -> ModelRegistry:
    instance = object.__new__(ModelRegistry)
    instance.artifact_path = None
    instance._error = None
    instance._version_digest = "test"
    instance._artifact = {
        "classifier_pipeline": Classifier(),
        "probability_calibrator": Calibrator(),
        "ranker_pipeline": Ranker(),
        "classifier_weight": 0.3,
        "ranker_weight": 0.7,
        "feature_columns": FEATURES,
    }
    return instance


def model_frame(rows: int = 30) -> pd.DataFrame:
    values = np.linspace(0.02, 0.98, rows)
    frame = pd.DataFrame({feature: np.zeros(rows) for feature in FEATURES})
    frame[FEATURES[0]] = values
    frame[FEATURES[1]] = values[::-1]
    frame["label_date"] = pd.Timestamp("2025-08-01")
    frame["cell_id"] = [f"cell-{index:02d}" for index in range(rows)]
    return frame


def test_scoring_matches_notebook_blend_and_top_25():
    frame = model_frame()
    scored = registry().score(frame)

    expected_classifier = within_day_percentile(frame[FEATURES[0]].to_numpy(), frame["label_date"])
    expected_ranker = within_day_percentile(frame[FEATURES[1]].to_numpy(), frame["label_date"])
    expected_alert = 0.3 * expected_classifier + 0.7 * expected_ranker

    np.testing.assert_allclose(scored.probability, frame[FEATURES[0]].to_numpy() * 0.5)
    np.testing.assert_allclose(scored.alert_score, expected_alert)
    assert sorted(scored.priority_rank.tolist()) == list(range(1, 31))
    assert scored.alert_top_25.sum() == 25


def prepared_contract_frame(feature_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({feature: [1.0, 2.0] for feature in feature_names})
    frame["cell_id"] = ["cell-a", "cell-b"]
    frame["label_date"] = pd.Timestamp("2025-08-01")
    frame["eo_asof_date"] = pd.Timestamp("2025-07-31")
    frame["feature_end_date"] = pd.Timestamp("2025-07-26")
    return frame


def configure_contract(monkeypatch):
    monkeypatch.setattr(
        risk_service,
        "get_feature_contract",
        lambda: {"feature_prune": {"kept_features": FEATURES}},
    )
    monkeypatch.setattr(
        risk_service.grid_catalog,
        "cell_lookup",
        lambda mode=None: {"cell-a": {}, "cell-b": {}},
    )


def test_prepared_contract_rejects_missing_features_and_wrong_dates(monkeypatch):
    configure_contract(monkeypatch)
    with pytest.raises(DataUnresolvableError):
        risk_service.validate_prepared_frame(
            prepared_contract_frame(FEATURES[:-1]), date(2025, 8, 1), FEATURES
        )

    wrong = prepared_contract_frame(FEATURES)
    wrong["eo_asof_date"] = pd.Timestamp("2025-08-01")
    with pytest.raises(DataUnresolvableError):
        risk_service.validate_prepared_frame(wrong, date(2025, 8, 1), FEATURES)


def test_prepared_contract_rejects_duplicate_cells(monkeypatch):
    configure_contract(monkeypatch)
    frame = prepared_contract_frame(FEATURES)
    frame["cell_id"] = ["cell-a", "cell-a"]
    with pytest.raises(DataUnresolvableError):
        risk_service.validate_prepared_frame(frame, date(2025, 8, 1), FEATURES)


def test_prepared_contract_accepts_only_exact_or_one_day_era5_fallback(monkeypatch):
    configure_contract(monkeypatch)
    fallback = prepared_contract_frame(FEATURES)
    fallback["feature_end_date"] = pd.Timestamp("2025-07-25")

    validated = risk_service.validate_prepared_frame(
        fallback, date(2025, 8, 1), FEATURES
    )
    assert validated["feature_end_date"].eq(pd.Timestamp("2025-07-25")).all()

    too_old = prepared_contract_frame(FEATURES)
    too_old["feature_end_date"] = pd.Timestamp("2025-07-24")
    with pytest.raises(DataUnresolvableError):
        risk_service.validate_prepared_frame(
            too_old, date(2025, 8, 1), FEATURES
        )


def test_prepared_contract_rejects_mixed_exact_and_fallback_dates(monkeypatch):
    configure_contract(monkeypatch)
    mixed = prepared_contract_frame(FEATURES)
    mixed["feature_end_date"] = [
        pd.Timestamp("2025-07-26"),
        pd.Timestamp("2025-07-25"),
    ]
    with pytest.raises(DataUnresolvableError):
        risk_service.validate_prepared_frame(
            mixed, date(2025, 8, 1), FEATURES
        )


class CachedRegistry:
    is_loaded = True
    unavailable_reason = None
    version = "model-v1"
    feature_columns = FEATURES

    def __init__(self):
        self.calls = 0

    def score(self, frame):
        self.calls += 1
        rows = len(frame)
        values = np.linspace(0.1, 0.9, rows)
        ranks = np.arange(1, rows + 1)
        return ScoredBatch(values, values, values, values, ranks, ranks <= 25)


def test_scored_day_cache_uses_parquet_identity(monkeypatch):
    configure_contract(monkeypatch)
    fake_registry = CachedRegistry()
    frame = prepared_contract_frame(FEATURES)
    identity = {"value": "artifact-1"}
    monkeypatch.setattr(risk_service, "get_model_registry", lambda: fake_registry)
    monkeypatch.setattr(
        risk_service,
        "load_prepared_day",
        lambda _: PreparedDay(frame, identity["value"]),
    )
    risk_service.clear_score_cache()

    risk_service.load_scored_day(date(2025, 8, 1))
    risk_service.load_scored_day(date(2025, 8, 1))
    assert fake_registry.calls == 1

    identity["value"] = "artifact-2"
    risk_service.load_scored_day(date(2025, 8, 1))
    assert fake_registry.calls == 2


def test_single_cell_result_matches_full_day_risk_map(monkeypatch):
    scored_frame = prepared_contract_frame(FEATURES).assign(
        _raw_probability=[0.2, 0.3],
        _probability=[0.1, 0.15],
        _rank_score=[0.7, 0.5],
        _alert_score=[0.9, 0.6],
        _priority_rank=[1, 2],
        _alert_top_25=[True, True],
    )
    scored_day = risk_service.ScoredDay(scored_frame, "artifact", "model-v1")

    class DetailRegistry:
        version = "model-v1"

        def explain(self, frame):
            return None

    monkeypatch.setattr(risk_service, "load_scored_day", lambda _: scored_day)
    monkeypatch.setattr(risk_service, "get_model_registry", lambda: DetailRegistry())
    monkeypatch.setattr(
        risk_service.grid_catalog,
        "cell_lookup",
        lambda mode=None: {"cell-a": {}, "cell-b": {}},
    )
    monkeypatch.setattr(
        risk_service.grid_catalog,
        "get_region_or_404",
        lambda _: {"id": "california", "geometry_id": "v1"},
    )
    monkeypatch.setattr(risk_service, "build_feature_catalog", lambda: [])

    timestamp = datetime(2025, 8, 1, 12, tzinfo=timezone.utc)
    risk_item = risk_service.build_risk_map("california", "forecast_24h", timestamp)["items"][0]
    prediction = risk_service.run_prediction("cell-a", "forecast_24h", timestamp)

    for key in ("probability", "raw_probability", "alert_score", "priority_rank", "alert_top_25"):
        assert prediction[key] == risk_item[key]
