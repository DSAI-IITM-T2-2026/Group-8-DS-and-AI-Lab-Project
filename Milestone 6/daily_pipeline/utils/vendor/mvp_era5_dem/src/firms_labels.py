from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

from .cells import lonlat_to_cell_ids

logger = logging.getLogger(__name__)

_EMPTY_COLS = [
    "date",
    "cell_id",
    "firms_n_pixels",
    "firms_max_confidence",
    "y_fire",
]


def _firms_path(date: pd.Timestamp, vsigs_prefix: str) -> str:
    return f"{vsigs_prefix.rstrip('/')}/{date.strftime('%Y-%m-%d')}.tif"


def label_day_to_cells(
    date: pd.Timestamp,
    vsigs_prefix: str,
    confidence_min: float,
    resolution: float = 0.25,
) -> pd.DataFrame:
    """Map FIRMS fire pixels (confidence >= threshold) onto ERA5 cell_ids for one day."""
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    path = _firms_path(date, vsigs_prefix)

    try:
        with rasterio.open(path) as src:
            descriptions = list(src.descriptions) if src.descriptions else []
            if descriptions and all(descriptions):
                band_map = {name: i + 1 for i, name in enumerate(descriptions)}
            else:
                band_map = {"firms_confidence": 1, "firms_t21": 2, "label": 3}

            conf = src.read(band_map["firms_confidence"]).astype("float32")
            fire = np.isfinite(conf) & (conf >= confidence_min)
            if not fire.any():
                return pd.DataFrame(columns=_EMPTY_COLS)

            rows, cols = np.where(fire)
            conf_vals = conf[rows, cols]
            xs, ys = xy(src.transform, rows, cols, offset="center")
            lons = np.asarray(xs, dtype="float64")
            lats = np.asarray(ys, dtype="float64")
            cell_ids = lonlat_to_cell_ids(lons, lats, resolution=resolution)
    except Exception as exc:
        logger.warning("FIRMS read failed for %s (%s): %s", date.date(), path, exc)
        return pd.DataFrame(columns=_EMPTY_COLS)

    tmp = pd.DataFrame({"cell_id": cell_ids, "confidence": conf_vals})
    agg = tmp.groupby("cell_id", as_index=False).agg(
        firms_n_pixels=("confidence", "size"),
        firms_max_confidence=("confidence", "max"),
    )
    agg["date"] = pd.Timestamp(date).normalize()
    agg["y_fire"] = 1
    return agg[["date", "cell_id", "firms_n_pixels", "firms_max_confidence", "y_fire"]]


def build_firms_cell_labels(
    start: pd.Timestamp,
    end: pd.Timestamp,
    vsigs_prefix: str,
    confidence_min: float,
    cache_dir: Path,
    resolution: float = 0.25,
    months: list[int] | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Build (and cache) daily cell-level FIRMS labels for the date range."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    periods = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    for period in periods:
        if months is not None and period.month not in months:
            continue

        cache_path = cache_dir / f"firms_cells_{period.year}_{period.month:02d}.parquet"
        month_start = max(start, period.to_timestamp(how="start"))
        month_end = min(end, period.to_timestamp(how="end").normalize())

        if cache_path.exists():
            logger.info("FIRMS cell cache hit %s", cache_path.name)
            month_df = pd.read_parquet(cache_path)
        else:
            days = list(pd.date_range(month_start, month_end, freq="D"))
            day_frames: list[pd.DataFrame] = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {
                    pool.submit(
                        label_day_to_cells,
                        day,
                        vsigs_prefix,
                        confidence_min,
                        resolution,
                    ): day
                    for day in days
                }
                for fut in as_completed(futs):
                    day_frames.append(fut.result())

            month_df = (
                pd.concat(day_frames, ignore_index=True)
                if day_frames
                else pd.DataFrame(columns=_EMPTY_COLS)
            )
            month_df.to_parquet(cache_path, index=False)
            logger.info("Wrote %s (%d fire-cell rows)", cache_path.name, len(month_df))

        mask = (month_df["date"] >= start) & (month_df["date"] <= end)
        frames.append(month_df.loc[mask])

    if not frames:
        return pd.DataFrame(columns=_EMPTY_COLS)
    return pd.concat(frames, ignore_index=True)
