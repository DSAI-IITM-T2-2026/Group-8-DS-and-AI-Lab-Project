"""FIRMS 1 km reference grid; regrid ERA5/DEM; rasterize numerical tables."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine, rowcol, xy
from rasterio.warp import reproject

from . import config
from .gcs_fetch import ensure_dem, ensure_firms

logger = logging.getLogger(__name__)

GRID_META_PATH = config.CACHE_GRID / "reference_meta.json"
LAND_MASK_PATH = config.CACHE_GRID / "land_mask.npy"


def _clip_aoi(da: xr.DataArray) -> xr.DataArray:
    b = config.AOI_BOUNDS
    return da.rio.clip_box(minx=b["west"], miny=b["south"], maxx=b["east"], maxy=b["north"])


def build_reference_grid(ref_date: str | None = None) -> xr.DataArray:
    """Load FIRMS GeoTIFF, clip to CA AOI, cache metadata."""
    ref_date = ref_date or config.FIRMS_REF_DATE
    path = ensure_firms(ref_date)
    da = rxr.open_rasterio(path, masked=True)
    if int(da.sizes.get("band", da.shape[0])) >= 3:
        da = da.isel(band=slice(0, 3))
    da = da.rio.write_crs("EPSG:4326")
    da = _clip_aoi(da)
    # single-band template for reproject_match
    template = da.isel(band=0).drop_vars("band", errors="ignore")
    template = template.rio.write_crs("EPSG:4326")

    transform = template.rio.transform()
    height, width = int(template.sizes["y"]), int(template.sizes["x"])
    meta = {
        "ref_date": ref_date,
        "height": height,
        "width": width,
        "transform": list(transform)[:6],
        "crs": "EPSG:4326",
        "bounds": config.AOI_BOUNDS,
    }
    GRID_META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Reference grid %sx%s from FIRMS %s", height, width, ref_date)
    return template


def load_grid_meta() -> dict:
    if not GRID_META_PATH.exists():
        build_reference_grid()
    return json.loads(GRID_META_PATH.read_text())


def reference_da() -> xr.DataArray:
    meta = load_grid_meta()
    h, w = meta["height"], meta["width"]
    transform = Affine(*meta["transform"])
    # rebuild empty DA with coords
    rows = np.arange(h)
    cols = np.arange(w)
    xs, _ = xy(transform, np.zeros(w), cols)
    _, ys = xy(transform, rows, np.zeros(h))
    data = np.zeros((h, w), dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
    )
    da = da.rio.write_crs(meta["crs"])
    da = da.rio.write_transform(transform)
    return da


def build_land_mask(ref: xr.DataArray | None = None) -> np.ndarray:
    if LAND_MASK_PATH.exists():
        return np.load(LAND_MASK_PATH)
    ref = ref if ref is not None else reference_da()
    import geopandas as gpd

    if not config.CA_GEOJSON.exists():
        logger.warning("CA geojson missing — land mask all True")
        mask = np.ones((ref.sizes["y"], ref.sizes["x"]), dtype=bool)
        np.save(LAND_MASK_PATH, mask)
        return mask
    gdf = gpd.read_file(config.CA_GEOJSON).to_crs("EPSG:4326")
    transform = ref.rio.transform()
    # geometry_mask expects bare geometries (not rasterize-style (geom, value) pairs)
    geoms = [geom for geom in gdf.geometry if geom is not None]
    # invert=True → True inside polygon (= land)
    land = geometry_mask(
        geoms,
        out_shape=(int(ref.sizes["y"]), int(ref.sizes["x"])),
        transform=transform,
        invert=True,
    )
    if int(land.sum()) == 0:
        raise RuntimeError(
            f"Land mask is empty after rasterizing {config.CA_GEOJSON}; check CRS/transform"
        )
    np.save(LAND_MASK_PATH, land)
    logger.info("Land mask: %d / %d cells", int(land.sum()), land.size)
    return land


def regrid_array_to_reference(
    arr_2d: np.ndarray,
    src_transform: Affine,
    src_crs: str,
    reference: xr.DataArray,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    dst = np.full((int(reference.sizes["y"]), int(reference.sizes["x"])), np.nan, dtype=np.float32)
    reproject(
        source=arr_2d.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=reference.rio.transform(),
        dst_crs=reference.rio.crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def regrid_era5_var_to_reference(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    reference: xr.DataArray,
) -> np.ndarray:
    """ERA5 lat/lon centers → FIRMS grid via xarray reproject_match."""
    # Ensure ascending lat for rioxarray
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[::-1, :]
    da = xr.DataArray(
        values.astype(np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    out = da.rio.reproject_match(reference, resampling=Resampling.bilinear)
    return np.asarray(out.values, dtype=np.float32)


def load_dem_on_grid(reference: xr.DataArray | None = None) -> dict[str, np.ndarray]:
    reference = reference if reference is not None else reference_da()
    paths = ensure_dem()
    out = {}
    for key, path in paths.items():
        da = rxr.open_rasterio(path, masked=True).squeeze(drop=True)
        da = da.rio.write_crs("EPSG:4326")
        matched = da.rio.reproject_match(reference, resampling=Resampling.bilinear)
        out[key] = np.asarray(matched.values, dtype=np.float32)
    return out


def rasterize_points_to_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    reference: xr.DataArray,
    agg: str = "mean",
) -> np.ndarray:
    """Scatter fine-grid numerical points onto FIRMS cells."""
    transform = reference.rio.transform()
    h, w = int(reference.sizes["y"]), int(reference.sizes["x"])
    rows, cols = rowcol(transform, lons, lats)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w) & np.isfinite(values)
    rows, cols, values = rows[valid], cols[valid], values[valid].astype(np.float64)
    sums = np.zeros((h, w), dtype=np.float64)
    counts = np.zeros((h, w), dtype=np.float64)
    np.add.at(sums, (rows, cols), values)
    np.add.at(counts, (rows, cols), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / np.maximum(counts, 1.0)
    out[counts == 0] = np.nan
    return out.astype(np.float32)


def grid_latlon_mesh(reference: xr.DataArray | None = None) -> tuple[np.ndarray, np.ndarray]:
    reference = reference if reference is not None else reference_da()
    transform = reference.rio.transform()
    h, w = int(reference.sizes["y"]), int(reference.sizes["x"])
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xs, ys = xy(transform, rows, cols)
    return np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)
