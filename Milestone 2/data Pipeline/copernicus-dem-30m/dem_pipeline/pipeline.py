"""End-to-end Copernicus DEM pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from dem_pipeline.clip import clip_dem
from dem_pipeline.config import PipelineConfig, setup_logging
from dem_pipeline.download import download_tiles
from dem_pipeline.merge import merge_tiles
from dem_pipeline.metadata import write_metadata
from dem_pipeline.qa import generate_preview_plots, run_quality_checks
from dem_pipeline.terrain import generate_terrain_features
from dem_pipeline.tiles import identify_tiles
from dem_pipeline.validate import validate_tiles

logger = logging.getLogger("dem_pipeline")


def run_pipeline(
    config_path: Path,
    project_root: Path | None = None,
    *,
    skip_download: bool = False,
    skip_merge: bool = False,
    skip_clip: bool = False,
    skip_terrain: bool = False,
) -> dict:
    """Execute the full DEM acquisition and processing workflow."""
    root = project_root or config_path.parent
    config = PipelineConfig.from_yaml(config_path, root)
    config.ensure_directories()
    setup_logging(config)

    logger.info("Starting Copernicus DEM pipeline")
    logger.info(
        "Study area: N=%.2f S=%.2f W=%.2f E=%.2f",
        config.study_area.north,
        config.study_area.south,
        config.study_area.west,
        config.study_area.east,
    )

    tiles = identify_tiles(config.study_area)
    logger.info("Identified %d tiles intersecting study area", len(tiles))

    if skip_download:
        downloaded = {
            p.stem.replace(".tif", ""): p
            for p in config.raw_dir.glob("*.tif")
            if p.stat().st_size > 0
        }
        logger.info("Skipping download; found %d existing raw tiles", len(downloaded))
    else:
        downloaded = download_tiles(tiles, config)
    validation = validate_tiles(tiles, downloaded, config)
    if not validation.valid_tiles:
        raise RuntimeError("No valid tiles available after validation")

    merged_path = config.merged_dir / "dem_merged.tif"
    clipped_path = config.clipped_dir / "dem_clipped.tif"

    if skip_merge and merged_path.exists():
        logger.info("Skipping merge; using %s", merged_path)
    else:
        merged_path = merge_tiles(validation.valid_tiles, config)

    if skip_clip and clipped_path.exists():
        logger.info("Skipping clip; using %s", clipped_path)
    else:
        clipped_path = clip_dem(merged_path, config)

    terrain_layers = (
        "elevation",
        "slope",
        "aspect",
        "hillshade",
        "tri",
        "tpi",
    )
    if skip_terrain and all((config.terrain_dir / f"{name}.tif").exists() for name in terrain_layers):
        terrain_outputs = {
            name: config.terrain_dir / f"{name}.tif" for name in terrain_layers
        }
        logger.info("Skipping terrain generation; using existing outputs")
    else:
        terrain_outputs = generate_terrain_features(clipped_path, config)

    qa_summary = run_quality_checks(terrain_outputs, config)
    generate_preview_plots(terrain_outputs, config)

    validation_summary = {
        "valid_count": len(validation.valid_tiles),
        "invalid_count": len(validation.invalid_tiles),
        "missing_count": len(validation.missing_tiles),
        "warning_count": len(validation.warnings),
        "passed": validation.passed,
    }

    metadata_path, summary_path = write_metadata(
        config,
        tiles,
        terrain_outputs,
        validation_summary,
        qa_summary,
    )

    logger.info("Pipeline completed successfully")
    return {
        "tiles": len(tiles),
        "downloaded": len(downloaded),
        "merged": str(merged_path),
        "clipped": str(clipped_path),
        "terrain_outputs": {k: str(v) for k, v in terrain_outputs.items()},
        "metadata": str(metadata_path),
        "summary": str(summary_path),
        "qa_passed": qa_summary["passed"],
    }
