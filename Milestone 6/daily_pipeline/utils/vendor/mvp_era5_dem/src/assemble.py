from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

WINDOW_FEATURES = [
    "t2m_max_7d",
    "tp_sum_7d",
    "wind_speed_max_7d",
    "rh_min_7d",
    "swvl1_mean_7d",
    "i10fg_max_7d",
]


def _add_rolling(era5: pd.DataFrame, history_days: int) -> pd.DataFrame:
    era5 = era5.sort_values(["cell_id", "date"]).copy()
    g = era5.groupby("cell_id", sort=False)
    era5["t2m_max_7d"] = g["t2m_max"].transform(
        lambda s: s.rolling(history_days, min_periods=history_days).max()
    )
    era5["tp_sum_7d"] = g["tp_sum_mm"].transform(
        lambda s: s.rolling(history_days, min_periods=history_days).sum()
    )
    era5["wind_speed_max_7d"] = g["wind_speed_mean"].transform(
        lambda s: s.rolling(history_days, min_periods=history_days).max()
    )
    era5["rh_min_7d"] = g["rh_mean"].transform(
        lambda s: s.rolling(history_days, min_periods=history_days).min()
    )
    era5["swvl1_mean_7d"] = g["swvl1_mean"].transform(
        lambda s: s.rolling(history_days, min_periods=history_days).mean()
    )
    era5["i10fg_max_7d"] = g["i10fg_max"].transform(
        lambda s: s.rolling(history_days, min_periods=history_days).max()
    )
    return era5


def assemble_samples(
    dem: pd.DataFrame,
    era5_daily: pd.DataFrame,
    firms_cells: pd.DataFrame,
    history_days: int,
    lead_days: int,
) -> pd.DataFrame:
    """
    One row per (cell_id, feature_end_date):
      features from day D (+ 7d window ending D)
      label y_fire from FIRMS on day D+lead_days
    """
    era5 = era5_daily.copy()
    era5["date"] = pd.to_datetime(era5["date"]).dt.normalize()

    keep_cols = ["date", "cell_id", "latitude", "longitude"] + [
        c for c in ERA5_DAY_FEATURES if c in era5.columns
    ]
    era5 = era5[keep_cols]
    era5 = _add_rolling(era5, history_days=history_days)

    # Drop rows without full history window
    era5 = era5.dropna(subset=WINDOW_FEATURES).reset_index(drop=True)

    dem_cols = ["cell_id"] + [c for c in DEM_FEATURES if c in dem.columns]
    dem_small = dem[dem_cols].drop_duplicates("cell_id")
    samples = era5.merge(dem_small, on="cell_id", how="inner")

    samples = samples.rename(columns={"date": "feature_end_date"})
    samples["label_date"] = samples["feature_end_date"] + pd.Timedelta(days=lead_days)

    labels = firms_cells.copy()
    if len(labels):
        labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
        labels = labels.rename(
            columns={
                "date": "label_date",
                "firms_n_pixels": "firms_n_pixels",
                "firms_max_confidence": "firms_max_confidence",
                "y_fire": "y_fire",
            }
        )
        samples = samples.merge(
            labels[
                ["label_date", "cell_id", "firms_n_pixels", "firms_max_confidence", "y_fire"]
            ],
            on=["label_date", "cell_id"],
            how="left",
        )
    else:
        samples["firms_n_pixels"] = 0
        samples["firms_max_confidence"] = np.nan
        samples["y_fire"] = 0

    samples["y_fire"] = samples["y_fire"].fillna(0).astype("int8")
    samples["firms_n_pixels"] = samples["firms_n_pixels"].fillna(0).astype("int32")

    # Region string for alerts (cell centroid)
    samples["region"] = [
        f"cell:{cid} ({lat:.2f},{lon:.2f})"
        for cid, lat, lon in zip(
            samples["cell_id"], samples["latitude"], samples["longitude"]
        )
    ]

    logger.info(
        "Assembled %d samples | positives=%d (%.4f%%)",
        len(samples),
        int(samples["y_fire"].sum()),
        100.0 * samples["y_fire"].mean() if len(samples) else 0.0,
    )
    return samples


def feature_columns(samples: pd.DataFrame) -> list[str]:
    cols = (
        [c for c in ERA5_DAY_FEATURES if c in samples.columns]
        + [c for c in WINDOW_FEATURES if c in samples.columns]
        + [c for c in DEM_FEATURES if c in samples.columns]
    )
    return cols


def apply_split(
    samples: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> dict[str, pd.DataFrame]:
    """Temporal split on label_date."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    d = samples["label_date"]

    splits = {
        "train": samples.loc[d <= train_end_ts].copy(),
        "val": samples.loc[(d > train_end_ts) & (d <= val_end_ts)].copy(),
        "test": samples.loc[d > val_end_ts].copy(),
    }
    for name, df in splits.items():
        logger.info(
            "split %-5s  n=%d  pos=%d  label_dates=[%s → %s]",
            name,
            len(df),
            int(df["y_fire"].sum()) if len(df) else 0,
            df["label_date"].min().date() if len(df) else "n/a",
            df["label_date"].max().date() if len(df) else "n/a",
        )
    return splits


def write_dataset(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    output_dir: Path,
    meta: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        out = output_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(df))

    with (meta_dir / "feature_columns.json").open("w") as f:
        json.dump(feature_cols, f, indent=2)
    with (meta_dir / "dataset_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)
