from __future__ import annotations

import pandas as pd
import pytest

from numerical_nextday.contracts import validate_samples


def valid_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["39.00_-121.00"],
            "feature_end_date": [pd.Timestamp("2025-07-10")],
            "label_date": [pd.Timestamp("2025-07-11")],
            "era5_source_date": [pd.Timestamp("2025-07-05")],
            "latitude": [39.0],
            "longitude": [-121.0],
            "model_bucket": ["fire_season"],
            "y_fire": [0],
            "s2_window_end": [pd.Timestamp("2025-07-10")],
        }
    )


def test_timing_contract_accepts_exact_alignment() -> None:
    validate_samples(valid_sample(), lead_days=1, era5_lag_days=5)


def test_timing_contract_rejects_future_satellite_data() -> None:
    frame = valid_sample()
    frame["s2_window_end"] = pd.Timestamp("2025-07-11")
    with pytest.raises(ValueError, match="leakage"):
        validate_samples(frame, lead_days=1, era5_lag_days=5)


def test_timing_contract_rejects_wrong_era5_lag() -> None:
    frame = valid_sample()
    frame["era5_source_date"] = pd.Timestamp("2025-07-06")
    with pytest.raises(ValueError, match="lag"):
        validate_samples(frame, lead_days=1, era5_lag_days=5)
