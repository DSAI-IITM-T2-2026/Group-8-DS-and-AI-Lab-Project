#!/usr/bin/env python3
"""Clip DEM terrain layers to the official California state boundary (not bbox)."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eda.clip")

EDA_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (EDA_ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def clip_layer(
    src_path: Path,
    geometries: list,
    out_path: Path,
    downsample: int,
    nodata: float,
) -> Path:
    with rasterio.open(src_path) as src:
        clipped, transform = mask(
            src,
            geometries,
            crop=True,
            nodata=nodata,
            filled=True,
        )
        profile = src.profile.copy()
        height, width = clipped.shape[1], clipped.shape[2]

        if downsample > 1:
            out_h = max(1, height // downsample)
            out_w = max(1, width // downsample)
            data = np.empty((1, out_h, out_w), dtype=np.float32)
            reproject(
                source=clipped.astype(np.float32),
                destination=data,
                src_transform=transform,
                src_crs=src.crs,
                dst_transform=rasterio.transform.from_bounds(
                    *rasterio.transform.array_bounds(height, width, transform),
                    out_w,
                    out_h,
                ),
                dst_crs=src.crs,
                src_nodata=nodata,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )
            transform = rasterio.transform.from_bounds(
                *rasterio.transform.array_bounds(height, width, transform),
                out_w,
                out_h,
            )
            height, width = out_h, out_w
            clipped = data

        profile.update(
            {
                "height": height,
                "width": width,
                "transform": transform,
                "dtype": "float32",
                "nodata": nodata,
                "compress": "lzw",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
                "count": 1,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clipped.astype(np.float32))

    logger.info("Wrote %s (%dx%d)", out_path.name, width, height)
    return out_path


def main() -> None:
    cfg = load_config()
    boundary = gpd.read_file(EDA_ROOT / cfg["paths"]["boundary"]).to_crs("EPSG:4326")
    geoms = [geom.__geo_interface__ for geom in boundary.geometry]
    terrain_dir = EDA_ROOT / cfg["paths"]["terrain_dir"]
    out_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "clipped_ca"
    downsample = cfg["clip"]["downsample_factor"]
    nodata = cfg["clip"]["nodata"]

    logger.info(
        "Clipping to California state polygon (not bbox). downsample=%dx",
        downsample,
    )
    for layer in cfg["extraction"]["layers"]:
        src = terrain_dir / f"{layer}.tif"
        if not src.exists():
            logger.warning("Missing %s — skip", src)
            continue
        clip_layer(src, geoms, out_dir / f"{layer}_ca.tif", downsample, nodata)

    boundary.to_file(out_dir / "california_boundary.geojson", driver="GeoJSON")
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
