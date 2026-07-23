"""Loaders for new S2 / S5P Hive-partitioned features.csv stores."""
from __future__ import annotations

import io
import re
from typing import Optional

import pandas as pd

from src import config
from src.data.gcs import get_fs

_WINDOW_RE = re.compile(r"window=(\d+)")
_YEAR_RE = re.compile(r"year=(\d{4})")
_MONTH_RE = re.compile(r"month=(\d{2})")


def s2_window_path(year: int, month: int, window: int) -> str:
    return (
        f"{config.S2_BUCKET}/{config.S2_PREFIX}/"
        f"year={year:04d}/month={month:02d}/window={window:03d}/features.csv"
    )


def s5p_window_path(year: int, month: int, window: int) -> str:
    return (
        f"{config.S5P_BUCKET}/{config.S5P_PREFIX}/"
        f"year={year:04d}/month={month:02d}/window={window:03d}/features.csv"
    )


def list_partition_dirs(bucket: str, prefix: str, year: Optional[int] = None) -> list[str]:
    fs = get_fs()
    base = f"{bucket}/{prefix}"
    if year is not None:
        base = f"{base}/year={year:04d}"
    try:
        return sorted(fs.ls(base))
    except FileNotFoundError:
        return []


def list_feature_csv_paths(
    bucket: str,
    prefix: str,
    year: Optional[int] = None,
) -> list[dict]:
    """
    Walk year=/month=/window=/features.csv and return metadata rows.
    """
    fs = get_fs()
    years = list_partition_dirs(bucket, prefix)
    if year is not None:
        years = [y for y in years if y.endswith(f"year={year:04d}")]

    out = []
    for ypath in years:
        ym = _YEAR_RE.search(ypath)
        y = int(ym.group(1)) if ym else None
        try:
            months = sorted(fs.ls(ypath))
        except Exception:
            continue
        for mpath in months:
            mm = _MONTH_RE.search(mpath)
            m = int(mm.group(1)) if mm else None
            try:
                windows = sorted(fs.ls(mpath))
            except Exception:
                continue
            for wpath in windows:
                wm = _WINDOW_RE.search(wpath)
                w = int(wm.group(1)) if wm else None
                csv_path = f"{wpath}/features.csv"
                if not fs.exists(csv_path):
                    # sometimes ls already points at file
                    if wpath.endswith("features.csv"):
                        csv_path = wpath
                    else:
                        continue
                info = fs.info(csv_path)
                out.append(
                    {
                        "path": csv_path,
                        "year": y,
                        "month": m,
                        "window": w,
                        "size": info.get("size", 0),
                    }
                )
    return out


def read_features_csv(
    gcs_path: str,
    columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    fs = get_fs()
    with fs.open(gcs_path, "rb") as f:
        return pd.read_csv(f, usecols=columns, nrows=nrows)


def peek_window_dates(gcs_path: str) -> dict:
    df = read_features_csv(
        gcs_path, columns=["window_start", "window_end", "latitude", "longitude"], nrows=5
    )
    return {
        "window_start": str(df["window_start"].iloc[0]) if len(df) else None,
        "window_end": str(df["window_end"].iloc[0]) if len(df) else None,
        "sample_lat": float(df["latitude"].iloc[0]) if len(df) else None,
        "sample_lon": float(df["longitude"].iloc[0]) if len(df) else None,
    }


def latlon_bounds_from_csv(gcs_path: str, chunksize: int = 150_000) -> dict:
    fs = get_fs()
    lat_min = lon_min = float("inf")
    lat_max = lon_max = float("-inf")
    nrows = 0
    with fs.open(gcs_path, "rb") as f:
        for chunk in pd.read_csv(f, usecols=["latitude", "longitude"], chunksize=chunksize):
            nrows += len(chunk)
            lat_min = min(lat_min, float(chunk["latitude"].min()))
            lat_max = max(lat_max, float(chunk["latitude"].max()))
            lon_min = min(lon_min, float(chunk["longitude"].min()))
            lon_max = max(lon_max, float(chunk["longitude"].max()))
    return {
        "west": lon_min,
        "south": lat_min,
        "east": lon_max,
        "north": lat_max,
        "nrows": nrows,
    }
