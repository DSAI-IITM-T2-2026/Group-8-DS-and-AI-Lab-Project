#!/usr/bin/env python3
"""
Build an ERA5-resolution static DEM table for California and document fusion strategy.

DEM is static; ERA5 is spatiotemporal. Fusion = attach DEM covariates once per grid cell,
then join to every ERA5 timestep by (lat, lon) / cell_id.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eda.era5")

EDA_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (EDA_ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def era5_grid_points(boundary: gpd.GeoDataFrame, resolution: float) -> pd.DataFrame:
    """Create ERA5-like cell centers that fall inside California."""
    minx, miny, maxx, maxy = boundary.total_bounds
    # Snap to ERA5-like grid
    lons = np.arange(np.floor(minx / resolution) * resolution, maxx + resolution, resolution)
    lats = np.arange(np.floor(miny / resolution) * resolution, maxy + resolution, resolution)
    records = []
    geom = boundary.unary_union
    for lat in lats:
        for lon in lons:
            pt = Point(lon, lat)
            if geom.contains(pt) or geom.touches(pt):
                records.append(
                    {
                        "cell_id": f"{lat:.2f}_{lon:.2f}",
                        "latitude": float(lat),
                        "longitude": float(lon),
                    }
                )
    return pd.DataFrame(records)


def sample_dem_at_points(points: pd.DataFrame, clipped_dir: Path, layers: list[str], nodata: float) -> pd.DataFrame:
    coords = list(zip(points["longitude"], points["latitude"]))
    out = points.copy()
    for layer in layers:
        path = clipped_dir / f"{layer}_ca.tif"
        if not path.exists():
            out[layer] = np.nan
            continue
        with rasterio.open(path) as src:
            vals = [v[0] for v in src.sample(coords)]
        series = np.array(vals, dtype=np.float64)
        series[(series == nodata) | ~np.isfinite(series)] = np.nan
        out[layer] = series
        logger.info("Sampled %s at %d ERA5 cells (%d valid)", layer, len(out), np.isfinite(series).sum())
    return out


def write_fusion_report(cfg: dict, grid_df: pd.DataFrame, report_path: Path) -> None:
    era5_vars = cfg["era5"]["variables"]
    dem_layers = cfg["extraction"]["layers"]

    # Classify ERA5 vars for modeling notes
    weather_dynamic = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "instantaneous_10m_wind_gust",
        "total_precipitation",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "boundary_layer_height",
    ]
    vegetation_slow = [
        "high_vegetation_cover",
        "low_vegetation_cover",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation",
    ]

    lines = [
        "# DEM × ERA5 Fusion Analysis (California)",
        "",
        "## Goal",
        "Use **static Copernicus DEM terrain features** together with your **ERA5 time series**",
        "for wildfire modeling over California (2021–2025).",
        "",
        "## Key idea",
        "DEM does not change with time. ERA5 does. So:",
        "1. Build a **static lookup table** of DEM features on the ERA5 grid (one row per cell).",
        "2. For every ERA5 timestep `(time, lat, lon)`, **join** DEM features by cell id / lat-lon.",
        "3. Train models on the combined feature vector.",
        "",
        "## Your ERA5 variables",
        "",
        "### Dynamic weather / atmosphere (change hourly–daily)",
        *[f"- `{v}`" for v in weather_dynamic if v in era5_vars],
        "",
        "### Vegetation / land cover (slow / near-static in ERA5)",
        *[f"- `{v}`" for v in vegetation_slow if v in era5_vars],
        "",
        "### DEM static covariates (this pipeline)",
        *[f"- `{v}`" for v in dem_layers],
        "",
        "## Recommended join schema",
        "",
        "```",
        "ERA5 row:  time | lat | lon | 2m_temperature | ... | boundary_layer_height",
        "DEM row:         lat | lon | elevation | slope | aspect | hillshade | tri | tpi",
        "Joined:    time | lat | lon | [all ERA5] | [all DEM]",
        "```",
        "",
        f"- ERA5 resolution assumed: **{cfg['era5']['resolution_deg']}°**",
        f"- California ERA5 cells with DEM samples: **{len(grid_df)}**",
        f"- Cells with valid elevation: **{int(grid_df['elevation'].notna().sum()) if 'elevation' in grid_df else 0}**",
        "",
        "## How DEM helps with your ERA5 features",
        "",
        "| ERA5 feature group | How DEM adds value |",
        "|--------------------|--------------------|",
        "| Temperature / dewpoint | Elevation explains cold air drainage & lapse rate; slope/aspect drive local heating |",
        "| Wind U/V + gust | TRI / TPI / slope capture channeling, ridges, and exposure |",
        "| Precipitation | Orographic lift controlled by elevation + slope facing moisture flow |",
        "| Soil water L1/L2 | Slope & TPI relate to runoff vs retention; valleys hold moisture |",
        "| Vegetation cover / LAI | Elevation & aspect control vegetation zones; DEM is prior for fuel structure |",
        "| Boundary layer height | Terrain roughness (TRI) and elevation modulate BLH patterns |",
        "| Surface pressure | Strongly tied to elevation (hydrostatic); DEM is a strong covariate |",
        "",
        "## Practical fusion recipes",
        "",
        "### A. Grid-cell tabular model (recommended start)",
        "- Resample / sample DEM to ERA5 0.25° centers (done in `era5_grid_dem_features.parquet`).",
        "- Join DEM columns onto every ERA5 timestep.",
        "- Optional derived DEM features: `wind_exposure = slope * sin(aspect)`, `vpd` from T + dewpoint, `wind_speed = hypot(u,v)`.",
        "",
        "### B. Patch / multimodal model",
        "- Keep DEM at higher resolution (30–300 m) as spatial context around each ERA5 cell.",
        "- ERA5 provides the temporal weather sequence; DEM provides local terrain patch.",
        "",
        "### C. Derived physics-inspired features",
        "```python",
        "wind_speed = hypot(u10, v10)",
        "vpd ≈ es(T) - es(Td)   # from 2m_temperature & 2m_dewpoint",
        "orographic_index = elevation * slope",
        "ridge_indicator = (tpi > threshold)",
        "```",
        "",
        "## Suggested feature matrix for ML",
        "",
        "**Static (DEM):** elevation, slope, aspect (or sin/cos aspect), tri, tpi  ",
        "**Slow ERA5:** vegetation cover + LAI  ",
        "**Dynamic ERA5:** T, Td, pressure, winds, gust, precip, soil water, BLH  ",
        "**Labels:** FIRMS / fire occurrence at matching space-time",
        "",
        "## Files produced",
        "- `outputs/numerical/era5_grid_dem_features.parquet` — static DEM @ ERA5 cells",
        "- `outputs/figures/era5_grid_elevation.png` — map of elevation on ERA5 grid",
        "- this report",
        "",
    ]
    report_path.write_text("\n".join(lines))
    logger.info("Wrote fusion report %s", report_path)


def main() -> None:
    cfg = load_config()
    clipped_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "clipped_ca"
    num_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "numerical"
    fig_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "figures"
    report_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "reports"
    for d in (num_dir, fig_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    boundary = gpd.read_file(EDA_ROOT / cfg["paths"]["boundary"]).to_crs("EPSG:4326")
    points = era5_grid_points(boundary, cfg["era5"]["resolution_deg"])
    logger.info("ERA5-like cells inside California: %d", len(points))

    grid_df = sample_dem_at_points(
        points,
        clipped_dir,
        cfg["extraction"]["layers"],
        cfg["clip"]["nodata"],
    )
    out_parquet = num_dir / "era5_grid_dem_features.parquet"
    grid_df.to_parquet(out_parquet, index=False)
    grid_df.to_csv(num_dir / "era5_grid_dem_features.csv", index=False)
    logger.info("Wrote %s", out_parquet)

    # Quick map: elevation on ERA5 grid
    if "elevation" in grid_df.columns:
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(
            grid_df["longitude"],
            grid_df["latitude"],
            c=grid_df["elevation"],
            s=18,
            cmap="terrain",
            alpha=0.9,
        )
        plt.colorbar(sc, label="Elevation (m)")
        boundary.boundary.plot(ax=plt.gca(), color="black", linewidth=0.8)
        plt.title("Static DEM elevation sampled on ERA5 0.25° grid (CA)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(fig_dir / "era5_grid_elevation.png", dpi=140)
        plt.close()

    write_fusion_report(cfg, grid_df, report_dir / "dem_era5_fusion.md")

    meta = {
        "era5_resolution_deg": cfg["era5"]["resolution_deg"],
        "era5_variables": cfg["era5"]["variables"],
        "dem_layers": cfg["extraction"]["layers"],
        "n_cells": len(grid_df),
        "n_cells_with_elevation": int(grid_df["elevation"].notna().sum()) if "elevation" in grid_df else 0,
    }
    (num_dir / "era5_fusion_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("ERA5 fusion analysis complete")


if __name__ == "__main__":
    main()
