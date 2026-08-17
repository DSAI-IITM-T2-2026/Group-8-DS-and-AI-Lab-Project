from __future__ import annotations

from datetime import date, datetime, timezone
import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd


UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from preprocess.final_artifact import (  # noqa: E402
    _slice_historical_archive,
    existing_final_artifact,
    write_artifact_provenance,
)


FEATURES = [f"feature_{index}" for index in range(86)]


class FakeBlob:
    def __init__(self, payload: bytes | None):
        self.payload = payload
        self.size = len(payload) if payload is not None else None
        self.updated = datetime(2026, 8, 13, tzinfo=timezone.utc)
        self.download_calls = 0

    def exists(self):
        return self.payload is not None

    def reload(self):
        return None

    def download_as_bytes(self):
        self.download_calls += 1
        assert self.payload is not None
        return self.payload


class FakeBucket:
    def __init__(self, blob):
        self._blob = blob
        self.requested_name = None

    def blob(self, name):
        self.requested_name = name
        return self._blob


class FakeClient:
    def __init__(self, blob):
        self._bucket = FakeBucket(blob)

    def bucket(self, name):
        assert name == "test-bucket"
        return self._bucket


def config(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(
        '{"feature_prune":{"kept_features":['
        + ",".join(f'"{feature}"' for feature in FEATURES)
        + "]}}",
        encoding="utf-8",
    )
    cells = tmp_path / "cells.csv"
    cells.write_text(
        "cell_id,fire_region_category\ncell-a,High\ncell-b,Medium\n",
        encoding="utf-8",
    )
    return {
        "gcs": {
            "bucket": "test-bucket",
            "project": "test-project",
            "prefixes": {"final_processed": "final_processed"},
            "historical_archive": "final_processed/2019_2025/2019-2025.parquet",
        },
        "task": {"era5_lag_days": 5, "lead_days": 1},
        "preprocess": {"cell_subset": "high_medium_fire"},
        "paths": {
            "contracts": str(contract),
            "fire_region_csv": str(cells),
            "local_cache": str(tmp_path / "cache"),
        },
        "_utils_root": str(tmp_path),
    }


def prepared_frame(label_date: date = date(2026, 8, 12)):
    frame = pd.DataFrame({feature: [1.0, 2.0] for feature in FEATURES})
    frame["cell_id"] = ["cell-a", "cell-b"]
    day = pd.Timestamp(label_date)
    frame["label_date"] = day
    frame["eo_asof_date"] = day - pd.Timedelta(days=1)
    frame["feature_end_date"] = day - pd.Timedelta(days=6)
    frame["y_fire"] = [1, 0]
    return frame


def parquet_bytes(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def test_valid_existing_parquet_is_reused_without_local_file(tmp_path):
    blob = FakeBlob(parquet_bytes(prepared_frame()))
    client = FakeClient(blob)

    artifact = existing_final_artifact(
        date(2026, 8, 12), config(tmp_path), storage_client=client
    )

    assert artifact is not None
    assert artifact["objectUri"] == "gs://test-bucket/final_processed/2026-08-12_test.parquet"
    assert artifact["featureCount"] == 86
    assert artifact["cellCount"] == 2
    assert blob.download_calls == 1


def test_live_artifact_without_provenance_requires_reconstruction(tmp_path):
    artifact = existing_final_artifact(
        date(2026, 8, 12),
        config(tmp_path),
        storage_client=FakeClient(FakeBlob(parquet_bytes(prepared_frame()))),
        require_provenance=True,
        expected_cutoff="2026-08-11T06:30:00-07:00",
    )
    assert artifact is None


def test_local_provenance_records_cutoff_and_ordered_features(tmp_path):
    cfg = config(tmp_path)
    cfg["task"]["timezone"] = "America/Los_Angeles"
    cfg["_forecast_context"] = {
        "cutoffAt": "2026-08-11T06:30:00-07:00",
        "timezone": "America/Los_Angeles",
        "forecastMode": "provisional_tomorrow",
        "sourceSnapshots": {"dem": {"ready": True, "mode": "static"}},
    }
    target = tmp_path / "2026-08-12_test.parquet"
    frame = prepared_frame()
    frame.to_parquet(target, index=False)
    artifact, provenance_uri = write_artifact_provenance(
        frame,
        date(2026, 8, 12),
        cfg,
        object_uri=str(target),
        feature_cols=FEATURES,
        created_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )
    payload = __import__("json").loads(Path(provenance_uri).read_text())
    assert artifact["forecastMode"] == "provisional_tomorrow"
    assert payload["orderedFeatures"] == FEATURES
    assert payload["cutoffAt"] == "2026-08-11T06:30:00-07:00"


def test_missing_parquet_does_not_download(tmp_path):
    blob = FakeBlob(None)
    artifact = existing_final_artifact(
        date(2026, 8, 12), config(tmp_path), storage_client=FakeClient(blob)
    )
    assert artifact is None
    assert blob.download_calls == 0


def test_invalid_existing_parquet_requests_rebuild(tmp_path):
    frame = prepared_frame()
    frame.loc[0, FEATURES[0]] = np.nan
    artifact = existing_final_artifact(
        date(2026, 8, 12),
        config(tmp_path),
        storage_client=FakeClient(FakeBlob(parquet_bytes(frame))),
    )
    assert artifact is None


def test_unreadable_existing_object_requests_rebuild(tmp_path):
    artifact = existing_final_artifact(
        date(2026, 8, 12),
        config(tmp_path),
        storage_client=FakeClient(FakeBlob(b"not a parquet file")),
    )
    assert artifact is None


def test_2026_missing_daily_does_not_touch_archive(tmp_path, monkeypatch):
    archive = tmp_path / "archive.parquet"
    prepared_frame(date(2025, 6, 15)).to_parquet(archive, index=False)
    monkeypatch.setenv("WILDFIRE_HISTORICAL_ARCHIVE_URI", str(archive))
    monkeypatch.setenv("WILDFIRE_ALLOW_GCS", "false")

    calls: list[tuple[str, date]] = []

    def _spy(source, label_date):
        calls.append((source, label_date))
        return _slice_historical_archive(source, label_date)

    monkeypatch.setattr(
        "preprocess.final_artifact._slice_historical_archive",
        _spy,
    )

    artifact = existing_final_artifact(
        date(2026, 8, 12),
        config(tmp_path),
        storage_client=FakeClient(FakeBlob(None)),
    )
    assert artifact is None
    assert calls == []


def test_2025_missing_daily_slices_archive(tmp_path, monkeypatch):
    label = date(2025, 6, 15)
    archive = tmp_path / "archive.parquet"
    # Archive holds the requested day plus a distractor day.
    pd.concat(
        [prepared_frame(label), prepared_frame(date(2025, 6, 16))],
        ignore_index=True,
    ).to_parquet(archive, index=False)
    monkeypatch.setenv("WILDFIRE_HISTORICAL_ARCHIVE_URI", str(archive))
    monkeypatch.setenv("WILDFIRE_ALLOW_GCS", "false")

    cfg = config(tmp_path)
    artifact = existing_final_artifact(
        label, cfg, storage_client=FakeClient(FakeBlob(None))
    )

    assert artifact is not None
    assert artifact["labelDate"] == "2025-06-15"
    assert artifact["featureCount"] == 86
    assert artifact["cellCount"] == 2
    cached = Path(cfg["paths"]["local_cache"]) / "final_processed" / "2025-06-15_test.parquet"
    assert cached.is_file()
    cached_frame = pd.read_parquet(cached)
    assert set(cached_frame["label_date"].astype(str)) == {"2025-06-15"}
    assert cached_frame["y_fire"].tolist() == [1, 0]


def test_day_absent_from_archive_returns_none(tmp_path, monkeypatch):
    archive = tmp_path / "archive.parquet"
    prepared_frame(date(2025, 6, 15)).to_parquet(archive, index=False)
    monkeypatch.setenv("WILDFIRE_HISTORICAL_ARCHIVE_URI", str(archive))
    monkeypatch.setenv("WILDFIRE_ALLOW_GCS", "false")

    artifact = existing_final_artifact(
        date(2025, 7, 1),
        config(tmp_path),
        storage_client=FakeClient(FakeBlob(None)),
    )
    assert artifact is None
