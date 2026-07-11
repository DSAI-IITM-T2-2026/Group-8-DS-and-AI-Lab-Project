"""Metadata and dataset summary generation."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from dem_pipeline.config import PipelineConfig
from dem_pipeline.tiles import DemTile

logger = logging.getLogger("dem_pipeline.metadata")


def _raster_stats(path: Path, nodata: float) -> dict:
    with rasterio.open(path) as src:
        # Use overviews for large rasters to avoid loading full arrays into memory.
        if src.width * src.height > 50_000_000 and src.overviews(1):
            overview_idx = len(src.overviews(1))
            data = src.read(1, out_shape=(
                src.height // (2 ** overview_idx),
                src.width // (2 ** overview_idx),
            ), masked=True)
        else:
            data = src.read(1, masked=True)

        valid = data.compressed() if hasattr(data, "compressed") else data[~data.mask]
        if valid.size == 0:
            valid = data.flatten()
            valid = valid[valid != nodata]

        bounds = src.bounds
        return {
            "file": path.name,
            "crs": src.crs.to_string() if src.crs else None,
            "width": src.width,
            "height": src.height,
            "bounds": {
                "west": bounds.left,
                "south": bounds.bottom,
                "east": bounds.right,
                "north": bounds.top,
            },
            "min": float(np.min(valid)) if valid.size else None,
            "max": float(np.max(valid)) if valid.size else None,
            "mean": float(np.mean(valid)) if valid.size else None,
            "std": float(np.std(valid)) if valid.size else None,
        }


def write_metadata(
    config: PipelineConfig,
    tiles: list[DemTile],
    terrain_outputs: dict[str, Path],
    validation_summary: dict,
    qa_summary: dict,
) -> tuple[Path, Path]:
    """Write metadata.json and dataset_summary.csv."""
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    layer_stats = {
        name: _raster_stats(path, config.nodata_value)
        for name, path in terrain_outputs.items()
    }

    metadata = {
        "dataset": config.dataset_name,
        "resolution_m": config.resolution_m,
        "generated_at_utc": timestamp,
        "study_area": {
            "north": config.study_area.north,
            "south": config.study_area.south,
            "west": config.study_area.west,
            "east": config.study_area.east,
        },
        "tiles": {
            "total": len(tiles),
            "tile_names": [t.name for t in tiles],
        },
        "validation": validation_summary,
        "quality_assurance": qa_summary,
        "outputs": {
            name: str(path.relative_to(config.base_dir))
            for name, path in terrain_outputs.items()
        },
        "layers": layer_stats,
    }

    metadata_path = config.metadata_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    summary_path = config.metadata_dir / "dataset_summary.csv"
    fieldnames = [
        "layer",
        "file",
        "crs",
        "width",
        "height",
        "west",
        "south",
        "east",
        "north",
        "min",
        "max",
        "mean",
        "std",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer_name, stats in layer_stats.items():
            bounds = stats["bounds"]
            writer.writerow(
                {
                    "layer": layer_name,
                    "file": stats["file"],
                    "crs": stats["crs"],
                    "width": stats["width"],
                    "height": stats["height"],
                    "west": bounds["west"],
                    "south": bounds["south"],
                    "east": bounds["east"],
                    "north": bounds["north"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
            )

    logger.info("Wrote metadata -> %s", metadata_path)
    logger.info("Wrote summary -> %s", summary_path)
    return metadata_path, summary_path
