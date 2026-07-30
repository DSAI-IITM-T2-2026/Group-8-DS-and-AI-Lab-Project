from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import require_columns


def make_cell_id(latitude: float, longitude: float) -> str:
    return f"{float(latitude):.2f}_{float(longitude):.2f}"


def coordinates_to_cell_ids(
    longitudes: np.ndarray, latitudes: np.ndarray, resolution: float
) -> np.ndarray:
    cell_lon = np.round(np.asarray(longitudes, dtype="float64") / resolution) * resolution
    cell_lat = np.round(np.asarray(latitudes, dtype="float64") / resolution) * resolution
    return np.asarray(
        [make_cell_id(lat, lon) for lat, lon in zip(cell_lat, cell_lon)], dtype=object
    )


def load_dem_cells(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"DEM cell table not found: {path}. Copy the Milestone 3 DEM parquet here or "
            "set paths.dem_cells in config.yaml."
        )
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    require_columns(
        frame,
        ["cell_id", "latitude", "longitude", "elevation", "slope", "aspect", "tri", "tpi"],
        "DEM",
    )
    frame = frame.copy()
    frame["aspect_sin"] = np.sin(np.deg2rad(frame["aspect"]))
    frame["aspect_cos"] = np.cos(np.deg2rad(frame["aspect"]))
    frame["orographic_index"] = frame["elevation"] * frame["slope"]
    frame = frame.dropna(subset=["elevation"]).drop_duplicates("cell_id")
    return frame.reset_index(drop=True)
