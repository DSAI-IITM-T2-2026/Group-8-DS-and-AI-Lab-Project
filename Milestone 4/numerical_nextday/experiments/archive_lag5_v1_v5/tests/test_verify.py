from __future__ import annotations

from numerical_nextday.pipeline import verify


def test_probe_flags_incomplete_daily_source(monkeypatch) -> None:
    objects = [
        f"gs://bucket/year=2025/month={month:02d}/window={day:03d}/features.csv"
        for month in range(1, 13)
        for day in (1, 2)
    ]
    monkeypatch.setattr(verify, "gcs_list", lambda uri, timeout: objects)
    result = verify._probe(
        "gs://bucket/year=2025/month=*/window=*/features.csv",
        list(range(1, 13)),
        r"/month=(\d{2})/",
        expected_object_count=365,
    )
    assert result["status"] == "missing"
    assert result["available_months"] == list(range(1, 13))
    assert result["missing_object_count"] == 341
