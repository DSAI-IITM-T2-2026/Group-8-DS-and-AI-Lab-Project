"""Unit tests that do not require GCS."""

from __future__ import annotations

import numpy as np
import pandas as pd

from numerical_nextday.data.joins import attach_causal_window_end
from numerical_nextday.train.router import bucket_for_month, filter_bucket


def test_causal_attach_no_future_leak():
    samples = pd.DataFrame(
        {
            "cell_id": ["a", "a"],
            "eo_asof_date": pd.to_datetime(["2024-08-10", "2024-08-12"]),
            "y_fire": [0, 1],
        }
    )
    # Window ending Aug 11 covers... but causal must NOT use for Aug 10
    w = pd.DataFrame(
        {
            "cell_id": ["a", "a"],
            "window_end": pd.to_datetime(["2024-08-09", "2024-08-11"]),
            "v": [1.0, 99.0],
        }
    )
    out = attach_causal_window_end(samples, [w], ["v"], date_col="eo_asof_date", prefix="t_")
    assert out.loc[out["eo_asof_date"] == "2024-08-10", "t_v"].iloc[0] == 1.0
    assert out.loc[out["eo_asof_date"] == "2024-08-12", "t_v"].iloc[0] == 99.0


def test_router_months():
    cfg = {
        "model_buckets": {
            "fire_season_months": [4, 5, 6, 7, 8, 9, 10, 11],
            "month_models": [1, 2, 3, 12],
        }
    }
    assert bucket_for_month(1, cfg) == "jan"
    assert bucket_for_month(8, cfg) == "fire_season"
    assert bucket_for_month(4, cfg) == "fire_season"


def test_lag_alignment_math():
    lead, lag = 1, 5
    feature_end = pd.Timestamp("2024-08-01")
    label = feature_end + pd.Timedelta(days=lead + lag)
    assert (label - feature_end).days == 6
    eo = label - pd.Timedelta(days=1)
    assert eo == pd.Timestamp("2024-08-06")
