from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd


UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from preprocess.build_stage_c_day import (
    append_era5_fallback_target_rows,
    stage_c_date_bounds,
)


def test_tomorrow_panel_keeps_d_row_with_causal_source_clips():
    target = date(2026, 8, 18)
    daily_cfg = {
        "task": {"lookback_days": 30},
        "_forecast_context": {"firmsThroughDate": "2026-08-16"},
    }
    m4_cfg = {
        "task": {"era5_lag_days": 5, "lead_days": 1, "history_days": 7},
    }

    bounds = stage_c_date_bounds(
        target,
        daily_cfg,
        m4_cfg,
        current_day=date(2026, 8, 17),
    )

    assert bounds["panel_end"] == target
    assert bounds["source_as_of"] == date(2026, 8, 17)
    assert bounds["era5_end"] == date(2026, 8, 12)
    assert bounds["firms_end"] == date(2026, 8, 16)
    assert bounds["panel_start"] == date(2026, 7, 19)
    assert bounds["era5_start"] == date(2026, 7, 6)


def test_fallback_appends_complete_truthful_prediction_day_rows():
    source_day = pd.Timestamp("2026-08-17")
    selected = pd.Timestamp("2026-08-11")
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "label_date": [source_day, source_day],
            "eo_asof_date": [source_day - pd.Timedelta(days=1)] * 2,
            "feature_end_date": [selected, selected],
            "era5_lag_days": [5, 5],
            "t2m_mean": [300.0, 301.0],
            "y_fire": [1, 1],
        }
    )

    result = append_era5_fallback_target_rows(
        frame, date(2026, 8, 18), date(2026, 8, 11)
    )
    target = result.loc[result["label_date"].eq(pd.Timestamp("2026-08-18"))]

    assert set(target["cell_id"]) == {"a", "b"}
    assert target["eo_asof_date"].eq(pd.Timestamp("2026-08-17")).all()
    assert target["feature_end_date"].eq(selected).all()
    assert target["t2m_mean"].tolist() == [300.0, 301.0]
    assert target["y_fire"].eq(0).all()
    assert target["era5_lag_days"].eq(6).all()
