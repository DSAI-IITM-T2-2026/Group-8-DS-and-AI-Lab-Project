from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

KEY_COLUMNS = ["cell_id", "feature_end_date"]
LABEL_COLUMNS = ["label_date", "y_fire"]

ERA5_DAY_FEATURES = [
    "t2m_mean",
    "t2m_max",
    "t2m_min",
    "d2m_mean",
    "rh_mean",
    "sp_mean",
    "wind_speed_mean",
    "wind_dir_sin",
    "wind_dir_cos",
    "i10fg_max",
    "tp_sum_mm",
    "swvl1_mean",
    "swvl2_mean",
    "soil_moisture_index",
    "cvh_mean",
    "cvl_mean",
    "lai_hv_mean",
    "lai_lv_mean",
    "blh_mean",
]

ERA5_WINDOW_FEATURES = [
    "t2m_max_7d",
    "tp_sum_7d",
    "wind_speed_max_7d",
    "rh_min_7d",
    "swvl1_mean_7d",
    "i10fg_max_7d",
]

DEM_FEATURES = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tri",
    "tpi",
    "orographic_index",
    "hillshade",
]

CALENDAR_FEATURES = ["day_of_year_sin", "day_of_year_cos"]


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def assert_unique_keys(frame: pd.DataFrame, keys: list[str] | tuple[str, ...], name: str) -> None:
    if frame.duplicated(list(keys)).any():
        examples = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)].head(5)
        raise ValueError(f"{name} contains duplicate keys {keys}:\n{examples}")


def validate_samples(frame: pd.DataFrame, lead_days: int, era5_lag_days: int) -> None:
    require_columns(
        frame,
        KEY_COLUMNS + LABEL_COLUMNS + ["era5_source_date", "latitude", "longitude", "model_bucket"],
        "samples",
    )
    assert_unique_keys(frame, KEY_COLUMNS, "samples")
    for col in ("feature_end_date", "label_date", "era5_source_date"):
        if not pd.api.types.is_datetime64_any_dtype(frame[col]):
            raise TypeError(f"samples.{col} must be datetime64")
    expected_label = frame["feature_end_date"] + pd.to_timedelta(lead_days, unit="D")
    if not expected_label.equals(frame["label_date"]):
        raise ValueError("label_date is not exactly feature_end_date + lead_days")
    expected_era5 = frame["feature_end_date"] - pd.to_timedelta(era5_lag_days, unit="D")
    if not expected_era5.equals(frame["era5_source_date"]):
        raise ValueError("era5_source_date violates configured lag")
    if not frame["y_fire"].dropna().isin([0, 1]).all():
        raise ValueError("y_fire must be binary")
    for time_col in ("s2_window_end", "s5p_window_end"):
        if time_col in frame and (frame[time_col] > frame["feature_end_date"]).fillna(False).any():
            raise ValueError(f"Point-in-time leakage: {time_col} exceeds feature_end_date")


def feature_columns(frame: pd.DataFrame, stage: str) -> list[str]:
    columns = [
        c
        for c in ERA5_DAY_FEATURES + ERA5_WINDOW_FEATURES + DEM_FEATURES + CALENDAR_FEATURES
        if c in frame
    ]
    if stage.upper() in {"B", "C"}:
        columns.extend(c for c in frame if c.startswith("s2_") and c != "s2_window_end")
    if stage.upper() == "C":
        columns.extend(c for c in frame if c.startswith("s5p_") and c != "s5p_window_end")
    return list(dict.fromkeys(columns))
