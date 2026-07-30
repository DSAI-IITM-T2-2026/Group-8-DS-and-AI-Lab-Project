from __future__ import annotations

import numpy as np
import pandas as pd

from numerical_nextday.data.eo import causal_attach


def test_causal_attach_uses_latest_nonfuture_window() -> None:
    base = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b"],
            "feature_end_date": pd.to_datetime(["2025-01-05", "2025-01-09", "2025-01-05"]),
        }
    )
    eo = pd.DataFrame(
        {
            "cell_id": ["a", "b", "a"],
            "window_end": pd.to_datetime(["2025-01-04", "2025-01-05", "2025-01-10"]),
            "s2_ndvi_mean": [0.2, 0.7, 0.9],
        }
    )
    joined = causal_attach(base, eo, "s2")
    assert joined.loc[0, "s2_ndvi_mean"] == 0.2
    assert joined.loc[1, "s2_ndvi_mean"] == 0.2
    assert joined.loc[2, "s2_ndvi_mean"] == 0.7
    assert (joined["s2_window_end"] <= joined["feature_end_date"]).all()


def test_missing_s5p_uses_zero_and_availability_flag() -> None:
    base = pd.DataFrame({"cell_id": ["a"], "feature_end_date": [pd.Timestamp("2021-06-01")]})
    joined = causal_attach(base, pd.DataFrame(columns=["cell_id", "window_end"]), "s5p")
    assert joined.loc[0, "s5p_data_available"] == 0
    assert np.isnan(joined.loc[0, "s5p_age_days"])
