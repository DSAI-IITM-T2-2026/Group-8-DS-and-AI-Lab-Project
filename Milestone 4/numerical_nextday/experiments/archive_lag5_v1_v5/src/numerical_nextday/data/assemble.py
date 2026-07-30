from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import bucket_for_month
from ..contracts import (
    DEM_FEATURES,
    ERA5_WINDOW_FEATURES,
    assert_unique_keys,
    validate_samples,
)
from ..grid import load_dem_cells
from ..io import atomic_parquet
from .eo import causal_attach, configured_feature_columns, read_eo_for_year
from .era5 import read_era5_range
from .firms import read_firms_year

logger = logging.getLogger(__name__)


def stage_path(cfg: dict, stage: str, year: int) -> Path:
    return cfg["paths"]["dataset_dir"] / f"stage_{stage.lower()}" / f"year={year}.parquet"


def add_rolling_weather(frame: pd.DataFrame, history_days: int) -> pd.DataFrame:
    result = frame.sort_values(["cell_id", "date"]).copy()
    grouped = result.groupby("cell_id", sort=False)
    rules = {
        "t2m_max_7d": ("t2m_max", "max"),
        "tp_sum_7d": ("tp_sum_mm", "sum"),
        "wind_speed_max_7d": ("wind_speed_mean", "max"),
        "rh_min_7d": ("rh_mean", "min"),
        "swvl1_mean_7d": ("swvl1_mean", "mean"),
        "i10fg_max_7d": ("i10fg_max", "max"),
    }
    for output, (source, reducer) in rules.items():
        result[output] = grouped[source].transform(
            lambda values, reducer=reducer: getattr(
                values.rolling(history_days, min_periods=history_days), reducer
            )()
        )
    return result


def build_stage_a_year(
    cfg: dict, year: int, era5_lag_days: int | None = None, force: bool = False
) -> Path:
    lag = int(cfg["task"]["era5_lag_days"]) if era5_lag_days is None else int(era5_lag_days)
    destination = stage_path(cfg, f"a_lag{lag}", year)
    if destination.exists() and not force:
        return destination
    history = int(cfg["task"]["history_days"])
    lead = int(cfg["task"]["lead_days"])
    label_start = pd.Timestamp(year=year, month=1, day=1)
    label_end = pd.Timestamp(year=year, month=12, day=31)
    feature_start = label_start - pd.Timedelta(days=lead)
    feature_end = label_end - pd.Timedelta(days=lead)
    era5_start = feature_start - pd.Timedelta(days=lag + history - 1)
    era5_end = feature_end - pd.Timedelta(days=lag)

    era5 = read_era5_range(cfg, era5_start, era5_end)
    era5 = add_rolling_weather(era5, history)
    era5 = era5.dropna(subset=ERA5_WINDOW_FEATURES).copy()
    era5 = era5.rename(columns={"date": "era5_source_date"})
    era5["feature_end_date"] = era5["era5_source_date"] + pd.Timedelta(days=lag)
    era5["label_date"] = era5["feature_end_date"] + pd.Timedelta(days=lead)
    era5 = era5.loc[era5["label_date"].dt.year == year]

    dem = load_dem_cells(cfg["paths"]["dem_cells"])
    dem_columns = ["cell_id", *[column for column in DEM_FEATURES if column in dem]]
    samples = era5.merge(dem[dem_columns], on="cell_id", how="inner")
    labels = read_firms_year(cfg, year).rename(columns={"date": "label_date"})
    assert_unique_keys(labels, ["cell_id", "label_date"], "FIRMS labels")
    samples = samples.merge(labels, on=["cell_id", "label_date"], how="left")
    samples["y_fire"] = samples["y_fire"].fillna(0).astype("int8")
    samples["firms_n_pixels"] = samples["firms_n_pixels"].fillna(0).astype("int32")
    day_of_year = samples["feature_end_date"].dt.dayofyear
    samples["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    samples["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    samples["model_bucket"] = samples["label_date"].dt.month.map(
        lambda month: bucket_for_month(cfg, int(month))
    )
    samples["era5_lag_days"] = lag
    samples["region"] = [
        f"cell:{cell_id} ({latitude:.2f},{longitude:.2f})"
        for cell_id, latitude, longitude in zip(
            samples["cell_id"], samples["latitude"], samples["longitude"]
        )
    ]
    samples = samples.sort_values(["label_date", "cell_id"]).reset_index(drop=True)
    validate_samples(samples, lead_days=lead, era5_lag_days=lag)
    atomic_parquet(samples, destination)
    logger.info(
        "Built Stage A lag=%d year=%d: rows=%d positives=%d",
        lag,
        year,
        len(samples),
        int(samples["y_fire"].sum()),
    )
    return destination


def attach_eo_year(
    cfg: dict,
    source: str,
    year: int,
    era5_lag_days: int | None = None,
    force: bool = False,
) -> Path:
    lag = int(cfg["task"]["era5_lag_days"]) if era5_lag_days is None else int(era5_lag_days)
    input_stage = f"a_lag{lag}" if source == "s2" else f"b_lag{lag}"
    output_stage = f"b_lag{lag}" if source == "s2" else f"c_lag{lag}"
    source_path = stage_path(cfg, input_stage, year)
    destination = stage_path(cfg, output_stage, year)
    if destination.exists() and not force:
        return destination
    if not source_path.exists():
        raise FileNotFoundError(f"Missing input stage: {source_path}")
    base = pd.read_parquet(source_path)
    eo = read_eo_for_year(cfg, source, year)
    joined = causal_attach(
        base,
        eo,
        source,
        max_age_days=int(cfg["features"]["max_age_days"][source]),
    )
    for column in configured_feature_columns(cfg, source):
        if column not in joined:
            joined[column] = 0.0 if source == "s5p" else np.nan
        elif source == "s5p":
            joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
    validate_samples(joined, int(cfg["task"]["lead_days"]), lag)
    atomic_parquet(joined, destination)
    logger.info("Built Stage %s year=%d rows=%d", output_stage, year, len(joined))
    return destination
