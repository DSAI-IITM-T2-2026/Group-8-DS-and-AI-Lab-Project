"""Quality assurance checks and optional visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from dem_pipeline.config import PipelineConfig

logger = logging.getLogger("dem_pipeline.qa")


def run_quality_checks(
    terrain_outputs: dict[str, Path],
    config: PipelineConfig,
) -> dict:
    """Verify dimensions, spatial extent, CRS, and raster statistics."""
    study = config.study_area
    results: dict = {"passed": True, "checks": []}

    for layer_name, path in terrain_outputs.items():
        check: dict = {"layer": layer_name, "file": str(path), "issues": []}

        with rasterio.open(path) as src:
            if src.width == 0 or src.height == 0:
                check["issues"].append("zero dimensions")
                results["passed"] = False

            if src.crs is None:
                check["issues"].append("missing CRS")
                results["passed"] = False
            elif src.crs.to_string() != config.expected_crs:
                check["issues"].append(
                    f"CRS mismatch: {src.crs.to_string()} != {config.expected_crs}"
                )

            bounds = src.bounds
            if bounds.left > study.west + 0.01 or bounds.right < study.east - 0.01:
                check["issues"].append("longitude extent does not cover study area")
            if bounds.bottom > study.south + 0.01 or bounds.top < study.north - 0.01:
                check["issues"].append("latitude extent does not cover study area")

            if src.width * src.height > 50_000_000:
                scale = max(src.width, src.height) / 2000
                data = src.read(
                    1,
                    out_shape=(int(src.height / scale), int(src.width / scale)),
                    resampling=rasterio.enums.Resampling.nearest,
                    masked=True,
                )
            else:
                data = src.read(1, masked=True)

            valid = data.compressed() if hasattr(data, "compressed") else data.flatten()
            if valid.size == 0:
                check["issues"].append("no valid pixels")
                results["passed"] = False
            else:
                check["statistics"] = {
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                }

        if check["issues"]:
            results["passed"] = False
            for issue in check["issues"]:
                logger.warning("QA [%s]: %s", layer_name, issue)

        results["checks"].append(check)

    status = "PASSED" if results["passed"] else "FAILED"
    logger.info("Quality assurance: %s", status)
    return results


def generate_preview_plots(
    terrain_outputs: dict[str, Path],
    config: PipelineConfig,
) -> Path | None:
    """Generate a quick-look visualization of terrain layers (downsampled)."""
    elevation_path = terrain_outputs.get("elevation")
    if elevation_path is None:
        return None

    preview_path = config.metadata_dir / "terrain_preview.png"
    config.metadata_dir.mkdir(parents=True, exist_ok=True)

    layers = ["elevation", "slope", "aspect", "hillshade", "tri", "tpi"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Copernicus DEM Terrain Features — California Study Area")

    max_preview_pixels = 2000

    for ax, layer_name in zip(axes.flat, layers):
        path = terrain_outputs.get(layer_name)
        if path is None:
            ax.set_visible(False)
            continue

        with rasterio.open(path) as src:
            scale = max(src.width, src.height) / max_preview_pixels
            if scale > 1:
                out_h = int(src.height / scale)
                out_w = int(src.width / scale)
                data = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=rasterio.enums.Resampling.bilinear,
                    masked=True,
                )
            else:
                data = src.read(1, masked=True)

            cmap = "terrain" if layer_name == "elevation" else "viridis"
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(layer_name.upper())
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved terrain preview -> %s", preview_path)
    return preview_path
