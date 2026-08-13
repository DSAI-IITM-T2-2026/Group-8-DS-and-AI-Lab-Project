"""Month / season model router."""

from __future__ import annotations

import pandas as pd


def bucket_for_month(month: int, cfg: dict) -> str:
    month = int(month)
    if month == 1:
        return "jan"
    if month == 2:
        return "feb"
    if month == 3:
        return "mar"
    if month == 12:
        return "dec"
    # April–November → fire_season (april_bucket default)
    return "fire_season"


def filter_bucket(df: pd.DataFrame, bucket: str, cfg: dict) -> pd.DataFrame:
    months = pd.to_datetime(df["label_date"]).dt.month
    if bucket == "fire_season":
        keep = set(cfg["model_buckets"]["fire_season_months"])
        return df.loc[months.isin(keep)].copy()
    mapping = {"jan": 1, "feb": 2, "mar": 3, "dec": 12}
    m = mapping[bucket]
    return df.loc[months == m].copy()


def route_predict(label_dates: pd.Series, cfg: dict) -> pd.Series:
    return label_dates.map(lambda d: bucket_for_month(pd.Timestamp(d).month, cfg))
