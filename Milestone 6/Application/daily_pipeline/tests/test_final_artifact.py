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

from preprocess.final_artifact import existing_final_artifact  # noqa: E402


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
        },
        "task": {"era5_lag_days": 5, "lead_days": 1},
        "preprocess": {"cell_subset": "high_medium_fire"},
        "paths": {"contracts": str(contract), "fire_region_csv": str(cells)},
        "_utils_root": str(tmp_path),
    }


def prepared_frame():
    frame = pd.DataFrame({feature: [1.0, 2.0] for feature in FEATURES})
    frame["cell_id"] = ["cell-a", "cell-b"]
    frame["label_date"] = pd.Timestamp("2026-08-12")
    frame["eo_asof_date"] = pd.Timestamp("2026-08-11")
    frame["feature_end_date"] = pd.Timestamp("2026-08-06")
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
