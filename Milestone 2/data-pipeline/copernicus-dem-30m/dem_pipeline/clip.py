"""Clip merged DEM to the study area bounding box."""

from __future__ import annotations

import logging
from pathlib import Path

import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping

from dem_pipeline.config import PipelineConfig

logger = logging.getLogger("dem_pipeline.clip")


def clip_dem(merged_path: Path, config: PipelineConfig) -> Path:
    """Clip the merged DEM to the configured study area."""
    config.clipped_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.clipped_dir / "dem_clipped.tif"

    study = config.study_area
    clip_geom = box(study.west, study.south, study.east, study.north)

    with rasterio.open(merged_path) as src:
        clipped, transform = mask(src, [mapping(clip_geom)], crop=True, nodata=config.nodata_value)
        profile = src.profile.copy()
        profile.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "nodata": config.nodata_value,
                "compress": "lzw",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)

    logger.info("Clipped DEM -> %s", output_path)
    return output_path
