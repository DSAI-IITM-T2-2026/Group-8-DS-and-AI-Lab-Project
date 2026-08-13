from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

spec = importlib.util.spec_from_file_location(
    "live_knn_build_stage_c_day", UTILS / "preprocess" / "build_stage_c_day.py"
)
build_stage_c_day = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(build_stage_c_day)
_apply_knn = build_stage_c_day._apply_knn


def test_live_window_observations_are_used_when_historical_donors_are_absent(tmp_path):
    source = tmp_path / "stage_c.parquet"
    output = tmp_path / "stage_c_knn"
    frame = pd.DataFrame(
        {
            "cell_id": ["observed-a", "observed-b", "missing"],
            "label_date": pd.to_datetime(["2026-08-10"] * 3),
            "s2n_available": [1, 1, 0],
            "s2n_B2_mean": [0.2, 0.8, np.nan],
            "s2n_B3_mean": [0.3, 0.9, np.nan],
            "temperature": [10.0, 30.0, 12.0],
        }
    )
    frame.to_parquet(source, index=False)

    _apply_knn(source, output, {"preprocess": {"knn_neighbors": 2}})

    result = pd.read_parquet(output / "all.parquet")
    imputed = result.loc[result["cell_id"].eq("missing")].iloc[0]
    assert np.isfinite(imputed[["s2n_B2_mean", "s2n_B3_mean"]].astype(float)).all()
    assert imputed["s2n_available"] == 0
    assert imputed["s2n_knn_imputed"] == 1


def test_live_knn_fails_when_no_observed_s2_donor_exists(tmp_path):
    source = tmp_path / "stage_c.parquet"
    frame = pd.DataFrame(
        {
            "cell_id": ["missing"],
            "label_date": pd.to_datetime(["2026-08-10"]),
            "s2n_available": [0],
            "s2n_B2_mean": [np.nan],
            "temperature": [12.0],
        }
    )
    frame.to_parquet(source, index=False)

    try:
        _apply_knn(source, tmp_path / "output", {"preprocess": {"knn_neighbors": 2}})
    except ValueError as exc:
        assert "no fully observed Sentinel-2 donor" in str(exc)
    else:  # pragma: no cover - documents the required terminal failure
        raise AssertionError("Expected missing donor data to fail")
