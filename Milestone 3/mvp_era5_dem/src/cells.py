from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_cell_id(lat: float, lon: float) -> str:
    """Match dem fusion IDs: '32.75_-117.25', '33.00_-117.00'."""
    return f"{float(lat):.2f}_{float(lon):.2f}"


def load_dem_cells(path: Path) -> pd.DataFrame:
    dem = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"cell_id", "latitude", "longitude", "elevation", "slope", "aspect", "tri", "tpi"}
    missing = required - set(dem.columns)
    if missing:
        raise ValueError(f"DEM cells missing columns: {sorted(missing)}")

    dem = dem.copy()
    dem["aspect_sin"] = np.sin(np.deg2rad(dem["aspect"]))
    dem["aspect_cos"] = np.cos(np.deg2rad(dem["aspect"]))
    dem["orographic_index"] = dem["elevation"] * dem["slope"]
    # Keep land cells with valid elevation (ocean/nodata already mostly filtered)
    dem = dem.dropna(subset=["elevation"]).reset_index(drop=True)
    return dem


def lonlat_to_cell_ids(lons: np.ndarray, lats: np.ndarray, resolution: float = 0.25) -> np.ndarray:
    """Snap lon/lat to ERA5 0.25° centers used in this project."""
    cell_lon = np.round(lons / resolution) * resolution
    cell_lat = np.round(lats / resolution) * resolution
    return np.array([make_cell_id(la, lo) for la, lo in zip(cell_lat, cell_lon)], dtype=object)
