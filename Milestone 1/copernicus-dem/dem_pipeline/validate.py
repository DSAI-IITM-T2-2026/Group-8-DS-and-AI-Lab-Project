"""Downloaded tile validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from dem_pipeline.config import PipelineConfig
from dem_pipeline.tiles import DemTile

logger = logging.getLogger("dem_pipeline.validate")


@dataclass
class ValidationResult:
    valid_tiles: list[Path] = field(default_factory=list)
    invalid_tiles: list[str] = field(default_factory=list)
    missing_tiles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.valid_tiles) and not self.invalid_tiles


def validate_tiles(
    tiles: list[DemTile],
    downloaded: dict[str, Path],
    config: PipelineConfig,
) -> ValidationResult:
    """Validate CRS, resolution, integrity, NoData, and missing tiles."""
    result = ValidationResult()

    for tile in tiles:
        path = downloaded.get(tile.name)
        if path is None or not path.exists():
            result.missing_tiles.append(tile.name)
            continue

        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    result.invalid_tiles.append(f"{tile.name}: missing CRS")
                    continue

                if src.crs.to_string() != config.expected_crs:
                    result.warnings.append(
                        f"{tile.name}: CRS is {src.crs.to_string()}, "
                        f"expected {config.expected_crs}"
                    )

                if src.count < 1:
                    result.invalid_tiles.append(f"{tile.name}: no bands")
                    continue

                data = src.read(1, masked=True)
                if data.size == 0:
                    result.invalid_tiles.append(f"{tile.name}: empty raster")
                    continue

                if src.nodata is not None and np.any(data == src.nodata):
                    result.warnings.append(
                        f"{tile.name}: contains NoData values ({src.nodata})"
                    )

                pixel_size = abs(src.transform.a)
                expected_deg = config.arc_seconds / 3600.0
                if not np.isclose(pixel_size, expected_deg, rtol=0.1):
                    result.warnings.append(
                        f"{tile.name}: pixel size {pixel_size:.6f}° "
                        f"differs from expected ~{expected_deg:.6f}°"
                    )

                result.valid_tiles.append(path)

        except rasterio.RasterioIOError as exc:
            result.invalid_tiles.append(f"{tile.name}: corrupt file ({exc})")

    logger.info(
        "Validation: %d valid, %d invalid, %d missing, %d warnings",
        len(result.valid_tiles),
        len(result.invalid_tiles),
        len(result.missing_tiles),
        len(result.warnings),
    )
    for warning in result.warnings:
        logger.warning(warning)

    return result
