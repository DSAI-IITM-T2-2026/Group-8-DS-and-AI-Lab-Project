"""Region / area catalog backed by the daily_pipeline's real, static assets.

The daily_pipeline does not model U.S. counties -- it scores a fixed 0.25-deg
ERA5-aligned grid over California (``cell_id`` like ``"37.75_-119.25"``,
confirmed by the neighbor-distance logic in
``daily_pipeline/utils/preprocess/build_champion_features.py``). This module
turns that grid into the ``/regions`` family of responses using two files
that ship with the repo and need no network access or credentials:

- ``utils/vendor/mvp_era5_dem/data/era5_grid_dem_features.parquet`` -- every
  cell in the AOI with real elevation/slope/etc.
- ``utils/vendor/fire_analysis2.csv`` -- the fire-region category
  (High Outlier / High / Medium / Low) used to build the modeled subset.

No county boundaries are fabricated: each "area" is the real grid cell the
model actually scores, rendered as its own 0.25 x 0.25 degree square. This is
flagged to the frontend team in the API README as the one semantic
difference from the contract's county example (see contract section 7/8).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from .config import ensure_pipeline_on_path, get_pipeline_config
from .errors import NotFoundError

GRID_STEP_DEG = 0.25
HALF_STEP_DEG = GRID_STEP_DEG / 2
GEOMETRY_VERSION = "california-grid-0.25deg-v1"
REGION_ID = "california"
STATE_FIPS = "06"

_DEFAULT_PRESETS = {
    "high_medium_fire": ["High Outlier", "High", "Medium"],
    "high_only": ["High Outlier", "High"],
    "all": None,
}


@lru_cache(maxsize=1)
def _dem_frame() -> pd.DataFrame:
    ensure_pipeline_on_path()
    from paths import resolve_path

    cfg = get_pipeline_config()
    path = resolve_path(cfg, "dem_local")
    if not path.is_file():
        raise NotFoundError(
            f"Grid definition file missing: {path}. It ships with the repo under "
            "daily_pipeline/utils/vendor/mvp_era5_dem/data/; check your checkout.",
            code="grid_definition_missing",
            status_code=503,
        )
    frame = pd.read_parquet(path, columns=["cell_id", "latitude", "longitude", "elevation"])
    return frame.astype({"cell_id": str})


@lru_cache(maxsize=1)
def _fire_category_frame() -> pd.DataFrame:
    ensure_pipeline_on_path()
    from paths import resolve_path

    cfg = get_pipeline_config()
    path = resolve_path(cfg, "fire_region_csv")
    if not path.is_file():
        return pd.DataFrame(columns=["cell_id", "fire_region_category"])
    frame = pd.read_csv(path)
    cat_col = "Category" if "Category" in frame.columns else "fire_region_category"
    frame = frame.rename(columns={cat_col: "fire_region_category"})
    return frame[["cell_id", "fire_region_category"]].astype({"cell_id": str})


def default_cell_subset_mode() -> str:
    cfg = get_pipeline_config()
    return str(cfg.get("preprocess", {}).get("cell_subset", "high_medium_fire"))


def _resolve_categories(mode: str) -> list[str] | None:
    mode_norm = (mode or "").strip().lower().replace("-", "_")
    if mode_norm in {"", "all", "none", "off"}:
        return None
    return _DEFAULT_PRESETS.get(mode_norm, _DEFAULT_PRESETS["high_medium_fire"])


def list_cells(mode: str | None = None) -> pd.DataFrame:
    """All cells (mode=all) or the modeled subset, each with lat/lon/category."""
    mode = mode or default_cell_subset_mode()
    dem = _dem_frame()
    categories = _fire_category_frame()
    merged = dem.merge(categories, on="cell_id", how="left")
    merged["fire_region_category"] = merged["fire_region_category"].fillna("Unclassified")

    cats = _resolve_categories(mode)
    if cats is not None:
        merged = merged.loc[merged["fire_region_category"].isin(cats)].copy()
    return merged.reset_index(drop=True)


def cell_lookup(mode: str | None = None) -> dict[str, dict[str, Any]]:
    frame = list_cells(mode)
    return {
        row.cell_id: {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "category": row.fire_region_category,
        }
        for row in frame.itertuples()
    }


def region_summary() -> dict[str, Any]:
    return {
        "id": REGION_ID,
        "name": "California",
        "country": "US",
        "region_type": "state",
        "geometry_id": GEOMETRY_VERSION,
        "availability": "supported",
    }


def region_detail() -> dict[str, Any]:
    all_cells = list_cells(mode="all")
    lat = all_cells["latitude"]
    lon = all_cells["longitude"]
    bounds = (
        float(lon.min() - HALF_STEP_DEG),
        float(lat.min() - HALF_STEP_DEG),
        float(lon.max() + HALF_STEP_DEG),
        float(lat.max() + HALF_STEP_DEG),
    )
    center = (float(lon.mean()), float(lat.mean()))
    return {**region_summary(), "center": center, "bounds": bounds}


def get_region_or_404(region_id: str) -> dict[str, Any]:
    if region_id != REGION_ID:
        raise NotFoundError(f"Unknown region: {region_id!r}. Only {REGION_ID!r} is served.")
    return region_detail()


def _cell_square(latitude: float, longitude: float) -> list[list[float]]:
    west, east = longitude - HALF_STEP_DEG, longitude + HALF_STEP_DEG
    south, north = latitude - HALF_STEP_DEG, latitude + HALF_STEP_DEG
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def region_geometry(region_id: str, mode: str | None = None) -> dict[str, Any]:
    get_region_or_404(region_id)
    frame = list_cells(mode)
    features = []
    for row in frame.itertuples():
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_cell_square(float(row.latitude), float(row.longitude))],
                },
                "properties": {
                    "id": row.cell_id,
                    "name": f"Grid cell {row.cell_id}",
                    "stateFips": STATE_FIPS,
                    "fireRegionCategory": row.fire_region_category,
                    "elevation": float(row.elevation),
                },
            }
        )
    return {
        "region_id": REGION_ID,
        "geometry_version": GEOMETRY_VERSION,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
