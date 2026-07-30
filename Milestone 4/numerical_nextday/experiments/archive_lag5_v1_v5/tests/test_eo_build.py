from __future__ import annotations

import pandas as pd

from numerical_nextday.config import load_config
from numerical_nextday.data import eo


def test_eo_month_streams_multiple_objects_and_aggregates(monkeypatch, tmp_path) -> None:
    cfg = load_config(
        overrides={
            "paths": {"cache_dir": str(tmp_path / "cache")},
            "execution": {"eo_parallel_streams": 2},
        }
    )
    objects = [
        "gs://example/year=2025/month=01/window=1/features.csv",
        "gs://example/year=2025/month=01/window=2/features.csv",
    ]
    feature = cfg["features"]["s2"][0]
    monkeypatch.setattr(eo, "gcs_list", lambda pattern: objects)

    def fake_csv(uri: str, wanted_columns: set[str], max_attempts: int) -> pd.DataFrame:
        offset = 0.0 if "window=1" in uri else 2.0
        return pd.DataFrame(
            {
                "latitude": [38.1],
                "longitude": [-121.1],
                "window_end": ["2025-01-05"],
                feature: [0.4 + offset],
            }
        )

    monkeypatch.setattr(eo, "gcs_read_csv", fake_csv)
    destination = eo.build_eo_month(cfg, "s2", 2025, 1)
    result = pd.read_parquet(destination)

    assert len(result) == 1
    assert result.loc[0, f"s2_{feature.lower()}"] == 1.4
