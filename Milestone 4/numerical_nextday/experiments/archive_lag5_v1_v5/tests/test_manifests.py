from __future__ import annotations

import pandas as pd

from numerical_nextday import io
from numerical_nextday.io import atomic_parquet, write_shard_manifest


def test_manifests_are_partitioned_by_year(tmp_path) -> None:
    manifest_root = tmp_path / "manifests" / "era5"
    outputs = []
    for year in (2024, 2025):
        shard = tmp_path / "cache" / f"year={year}" / "month=01.parquet"
        atomic_parquet(pd.DataFrame({"value": [year]}), shard)
        outputs.append(
            write_shard_manifest(
                shard,
                {"source": "era5", "year": year, "month": 1, "rows": 1},
                "test-config",
                manifest_root,
            )
        )
    assert outputs[0] != outputs[1]
    assert outputs[0].parent.name == "year=2024"
    assert outputs[1].parent.name == "year=2025"
    assert all(path.exists() for path in outputs)


def test_streaming_csv_retries_transient_failures(monkeypatch) -> None:
    attempts = []

    def sometimes_fails(uri, wanted_columns):
        attempts.append(uri)
        if len(attempts) < 3:
            raise RuntimeError("temporary transfer failure")
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(io, "_gcs_read_csv_once", sometimes_fails)
    frame = io.gcs_read_csv("gs://bucket/object.csv", {"value"}, max_attempts=3)
    assert len(attempts) == 3
    assert frame["value"].tolist() == [1]
