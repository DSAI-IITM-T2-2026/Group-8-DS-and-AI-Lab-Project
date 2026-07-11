"""Merge DEM tiles into a single seamless raster."""

from __future__ import annotations

import logging
from pathlib import Path

import rasterio
from rasterio.merge import merge

from dem_pipeline.config import PipelineConfig

logger = logging.getLogger("dem_pipeline.merge")

_BATCH_SIZE = 12


def _merge_batch(tile_paths: list[Path], output_path: Path, nodata: float) -> Path:
    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(datasets, nodata=nodata)
        profile = datasets[0].profile.copy()
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "nodata": nodata,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            }
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()
    return output_path


def merge_tiles(tile_paths: list[Path], config: PipelineConfig) -> Path:
    """Merge validated tiles into one GeoTIFF using batched merging."""
    config.merged_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.merged_dir / "dem_merged.tif"

    if not tile_paths:
        raise ValueError("No tiles available to merge")

    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with rasterio.open(output_path) as src:
                if src.width > 0 and src.height > 0:
                    logger.info("Using existing merged DEM: %s", output_path)
                    return output_path
        except rasterio.RasterioIOError:
            output_path.unlink(missing_ok=True)

    current_paths = list(tile_paths)
    batch_index = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for start in range(0, len(current_paths), _BATCH_SIZE):
            batch = current_paths[start : start + _BATCH_SIZE]
            if len(current_paths) <= _BATCH_SIZE and len(batch) == len(current_paths):
                logger.info("Final merge of %d rasters", len(batch))
                return _merge_batch(batch, output_path, config.nodata_value)

            intermediate = config.merged_dir / f"batch_{batch_index:03d}.tif"
            logger.info("Merging batch %d (%d tiles)", batch_index, len(batch))
            _merge_batch(batch, intermediate, config.nodata_value)
            next_paths.append(intermediate)
            batch_index += 1

        for old in current_paths:
            if old.name.startswith("batch_") and old.exists():
                old.unlink()
        current_paths = next_paths

    return _merge_batch(current_paths, output_path, config.nodata_value)
