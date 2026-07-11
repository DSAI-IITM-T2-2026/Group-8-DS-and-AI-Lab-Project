"""Terrain derivative generation: slope, aspect, hillshade, TRI, TPI."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from dem_pipeline.config import PipelineConfig

logger = logging.getLogger("dem_pipeline.terrain")


def _pixel_size_meters(transform: rasterio.Affine, latitude: float) -> tuple[float, float]:
    """Approximate pixel size in meters for geographic CRS."""
    lat_rad = math.radians(latitude)
    dx = abs(transform.a) * 111320.0 * math.cos(lat_rad)
    dy = abs(transform.e) * 111320.0
    return dx, dy


def _write_raster(
    output_path: Path,
    data: np.ndarray,
    profile: dict,
    nodata: float,
) -> None:
    out_profile = profile.copy()
    out_profile.update({
        "dtype": "float32",
        "nodata": nodata,
        "compress": "lzw",
        "count": 1,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    })
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _compute_slope_aspect(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    latitude: float,
    nodata: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx_m, dy_m = _pixel_size_meters(transform, latitude)
    valid = elevation != nodata

    elev = np.where(valid, elevation, np.nan)
    dz_dy, dz_dx = np.gradient(elev, dy_m, dx_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (360.0 + aspect_deg) % 360.0
    aspect_deg = np.where(~np.isfinite(aspect_deg), nodata, aspect_deg)

    slope_deg = np.where(~np.isfinite(slope_deg), nodata, slope_deg)
    slope_deg = np.where(~valid, nodata, slope_deg)
    aspect_deg = np.where(~valid, nodata, aspect_deg)

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def _compute_hillshade(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    latitude: float,
    nodata: float,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    dx_m, dy_m = _pixel_size_meters(transform, latitude)
    valid = elevation != nodata
    elev = np.where(valid, elevation, np.nan)
    dz_dy, dz_dx = np.gradient(elev, dy_m, dx_m)

    zenith_rad = math.radians(90.0 - altitude)
    azimuth_rad = math.radians(360.0 - azimuth + 90.0)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(dz_dx, -dz_dy)

    hillshade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade = (hillshade * 255.0).clip(0, 255)
    hillshade = np.where(~valid | ~np.isfinite(hillshade), nodata, hillshade)
    return hillshade.astype(np.float32)


def _compute_tri(elevation: np.ndarray, nodata: float) -> np.ndarray:
    """Terrain Ruggedness Index: RMS difference from 8 neighbors."""
    valid = elevation != nodata
    elev = np.where(valid, elevation, 0.0)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    neighbor_sum = ndimage.convolve(elev, kernel, mode="nearest")
    neighbor_count = ndimage.convolve(valid.astype(np.float32), kernel, mode="nearest")
    neighbor_mean = np.divide(
        neighbor_sum,
        neighbor_count,
        out=np.zeros_like(neighbor_sum),
        where=neighbor_count > 0,
    )

    diff_sq = np.zeros_like(elev, dtype=np.float32)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        shifted = np.roll(np.roll(elev, dy, axis=0), dx, axis=1)
        diff_sq += (elev - shifted) ** 2

    tri = np.sqrt(diff_sq / 8.0)
    tri = np.where(~valid, nodata, tri)
    return tri.astype(np.float32)


def _compute_tpi(elevation: np.ndarray, nodata: float, radius: int) -> np.ndarray:
    """Topographic Position Index: elevation minus neighborhood mean."""
    valid = elevation != nodata
    elev = np.where(valid, elevation, 0.0)
    size = 2 * radius + 1

    neighbor_sum = ndimage.uniform_filter(elev, size=size, mode="nearest") * (size**2)
    neighbor_count = ndimage.uniform_filter(valid.astype(np.float32), size=size, mode="nearest") * (
        size**2
    )
    neighbor_mean = np.divide(
        neighbor_sum,
        neighbor_count,
        out=np.zeros_like(neighbor_sum),
        where=neighbor_count > 0,
    )

    tpi = elev - neighbor_mean
    tpi = np.where(~valid, nodata, tpi)
    return tpi.astype(np.float32)


def generate_terrain_features(clipped_path: Path, config: PipelineConfig) -> dict[str, Path]:
    """Generate all terrain derivative rasters from the clipped DEM."""
    config.terrain_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    with rasterio.open(clipped_path) as src:
        elevation = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = config.nodata_value
        center_lat = (config.study_area.north + config.study_area.south) / 2.0

        elevation_path = config.terrain_dir / "elevation.tif"
        _write_raster(elevation_path, elevation, profile, nodata)
        outputs["elevation"] = elevation_path

        slope, aspect = _compute_slope_aspect(elevation, src.transform, center_lat, nodata)
        slope_path = config.terrain_dir / "slope.tif"
        aspect_path = config.terrain_dir / "aspect.tif"
        _write_raster(slope_path, slope, profile, nodata)
        _write_raster(aspect_path, aspect, profile, nodata)
        outputs["slope"] = slope_path
        outputs["aspect"] = aspect_path

        hillshade = _compute_hillshade(
            elevation,
            src.transform,
            center_lat,
            nodata,
            config.hillshade_azimuth,
            config.hillshade_altitude,
        )
        hillshade_path = config.terrain_dir / "hillshade.tif"
        _write_raster(hillshade_path, hillshade, profile, nodata)
        outputs["hillshade"] = hillshade_path

        tri = _compute_tri(elevation, nodata)
        tri_path = config.terrain_dir / "tri.tif"
        _write_raster(tri_path, tri, profile, nodata)
        outputs["tri"] = tri_path

        tpi = _compute_tpi(elevation, nodata, config.tpi_radius_pixels)
        tpi_path = config.terrain_dir / "tpi.tif"
        _write_raster(tpi_path, tpi, profile, nodata)
        outputs["tpi"] = tpi_path

    logger.info("Generated %d terrain layers in %s", len(outputs), config.terrain_dir)
    return outputs
