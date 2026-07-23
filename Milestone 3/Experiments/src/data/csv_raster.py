"""Rasterize S2/S5P features.csv (~413k CA cells) onto the FIRMS reference grid."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from src.data.feature_csv import read_features_csv
from src.data.gcs import get_fs


def firms_hw_transform(reference_grid_da):
    """Return (H, W, affine transform) for the FIRMS reference DataArray."""
    H = int(reference_grid_da.sizes["y"])
    W = int(reference_grid_da.sizes["x"])
    transform = reference_grid_da.rio.transform()
    return H, W, transform


def scatter_latlon_to_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    reference_grid_da,
) -> np.ndarray:
    """
    Nearest-cell scatter of point values onto the FIRMS grid.
    Multiple hits on one cell keep the last finite value (grids are ~1 km matched).
    """
    H, W, transform = firms_hw_transform(reference_grid_da)
    out = np.full((H, W), np.nan, dtype=np.float32)
    rows, cols = rowcol(transform, lons, lats)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float32)
    valid = (
        (rows >= 0)
        & (rows < H)
        & (cols >= 0)
        & (cols < W)
        & np.isfinite(vals)
        & np.isfinite(lats)
        & np.isfinite(lons)
    )
    if not np.any(valid):
        return out
    out[rows[valid], cols[valid]] = vals[valid]
    return out


def rasterize_feature_columns(
    gcs_path: str,
    columns: list[str],
    reference_grid_da,
) -> dict[str, np.ndarray]:
    """Load selected value columns + lat/lon from CSV and scatter onto FIRMS grid."""
    usecols = ["latitude", "longitude", *columns]
    df = read_features_csv(gcs_path, columns=usecols)
    lats = df["latitude"].to_numpy(dtype=np.float64)
    lons = df["longitude"].to_numpy(dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for col in columns:
        out[col] = scatter_latlon_to_grid(
            lats, lons, df[col].to_numpy(dtype=np.float64), reference_grid_da
        )
    return out


def find_s5p_csv_for_date(date_str: str) -> Optional[str]:
    """
    S5P window id ≈ day-of-year. Path:
      year=YYYY/month=MM/window=DDD/features.csv
    """
    from src import config

    ts = pd.Timestamp(date_str)
    doy = int(ts.dayofyear)
    path = (
        f"{config.S5P_BUCKET}/{config.S5P_PREFIX}/"
        f"year={ts.year:04d}/month={ts.month:02d}/window={doy:03d}/features.csv"
    )
    fs = get_fs()
    if fs.exists(path):
        return path
    return None
