"""Per-source loaders ported from M2 Kaggle notebook (Landsat dropped)."""
from __future__ import annotations

import calendar
import gc
import io
import os
import re
import zipfile
from typing import Optional

import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.warp import reproject
from rasterio.warp import Resampling as WarpResampling

from src import config
from src.data.gcs import get_fs, list_bucket_files


# ---------------------------------------------------------------------------
# FIRMS
# ---------------------------------------------------------------------------
def load_firms_raster(date_str: str):
    path = f"gs://{config.FIRMS_BUCKET}/{config.FIRMS_PREFIX}/{date_str}.tif"
    da = rxr.open_rasterio(path, masked=True)
    # Most days are 3-band (confidence, BT, detection). Some GCS tiles are 6-band;
    # keep the first 3 which match the historical schema used by firms_to_binary_label.
    n_band = int(da.sizes.get("band", da.shape[0]))
    if n_band < 3:
        raise ValueError(f"FIRMS {date_str}: expected >=3 bands, got {n_band}")
    if n_band != 3:
        print(f"  FIRMS {date_str}: {n_band} bands — using first 3", flush=True)
        da = da.isel(band=slice(0, 3))
    # Drop existing band coord before renaming (avoids xarray size conflicts)
    da = da.assign_coords(band=("band", ["confidence", "brightness_temp_K", "detection_flag"]))
    return da


def list_firms_dates() -> list[str]:
    files = list_bucket_files(config.FIRMS_BUCKET, config.FIRMS_PREFIX, suffix=".tif")
    dates = [f.split("/")[-1].replace(".tif", "") for f in files]
    return sorted(dates)


def firms_to_binary_label(firms_da, confidence_threshold: int = 30) -> np.ndarray:
    detection_flag = firms_da.sel(band="detection_flag").values
    confidence = firms_da.sel(band="confidence").values
    binary_label = (~np.isnan(detection_flag)).astype("uint8")
    if confidence_threshold is not None:
        low_conf = confidence < confidence_threshold
        low_conf_filled = np.where(np.isnan(low_conf), False, low_conf)
        binary_label = np.where(low_conf_filled, 0, binary_label)
    return binary_label


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------
_S2_NAME_RE = re.compile(r"s2_(\d{4})_(\d{2})")


def list_sentinel2_tiles() -> list[str]:
    return list_bucket_files(config.GCS_BUCKET, config.PREFIX_S2, suffix=".tif")


def parse_s2_year_month(path: str) -> Optional[tuple[int, int]]:
    name = path.split("/")[-1]
    m = _S2_NAME_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def s2_scene_date(year: int, month: int) -> str:
    """Use last calendar day of the month as the scene acquisition date."""
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last:02d}"


def load_sentinel2_tile(gcs_path: str, band_names=None):
    if band_names is None:
        band_names = ["B2_blue", "B3_green", "B4_red", "B8_nir", "B11_swir1", "B12_swir2"]
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}"
    da = rxr.open_rasterio(gcs_path, masked=True)
    da = da.assign_coords(band=band_names[: da.sizes["band"]])
    return da


def compute_sentinel2_indices(s2_da) -> dict[str, np.ndarray]:
    """CPU/numpy indices (avoids pushing huge tiles onto unified memory)."""
    red = np.asarray(s2_da.sel(band="B4_red").values, dtype=np.float32)
    nir = np.asarray(s2_da.sel(band="B8_nir").values, dtype=np.float32)
    swir2 = np.asarray(s2_da.sel(band="B12_swir2").values, dtype=np.float32)
    green = np.asarray(s2_da.sel(band="B3_green").values, dtype=np.float32)
    return {
        "NDVI_S2": (nir - red) / (nir + red + 1e-8),
        "NBR_S2": (nir - swir2) / (nir + swir2 + 1e-8),
        "NDWI_S2": (green - nir) / (green + nir + 1e-8),
    }


# ---------------------------------------------------------------------------
# Sentinel-5P
# ---------------------------------------------------------------------------
def list_sentinel5p_files() -> list[str]:
    return list_bucket_files(config.GCS_BUCKET, config.PREFIX_S5P, suffix=".tif")


def load_sentinel5p_file(gcs_path: str, band_names=None):
    """Defensive band-count handling — some files have 2+ bands unexpectedly."""
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}"
    da = rxr.open_rasterio(gcs_path, masked=True)
    n_bands = da.sizes["band"]
    if band_names is None:
        if n_bands == 1:
            band_names = ["aerosol_index"]
        else:
            band_names = ["aerosol_index"] + [f"unknown_band_{i}" for i in range(2, n_bands + 1)]
            print(
                f"WARNING: {gcs_path.split('/')[-1]}: found {n_bands} bands "
                f"(expected 1) — extras labeled generically."
            )
    da = da.assign_coords(band=band_names[:n_bands])
    return da


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------
def list_era5_year_folder(year: int) -> list[str]:
    return list_bucket_files(config.GCS_BUCKET, f"{config.PREFIX_ERA5}/{year}", suffix=".nc")


def open_era5_file(gcs_path_relative: str):
    """ERA5 .nc files are ZIP archives containing instant + accum NetCDFs."""
    fs = get_fs()
    with fs.open(gcs_path_relative, "rb") as f:
        raw_bytes = f.read()
    zip_buffer = io.BytesIO(raw_bytes)
    datasets = []
    with zipfile.ZipFile(zip_buffer) as zf:
        for inner_name in zf.namelist():
            inner_bytes = zf.read(inner_name)
            ds = xr.open_dataset(io.BytesIO(inner_bytes), engine="h5netcdf")
            datasets.append(ds)
    merged = xr.merge(datasets, compat="override", join="outer")
    if "valid_time" in merged.dims or "valid_time" in merged.coords:
        merged = merged.rename({"valid_time": "time"})
    return merged


def aggregate_era5_to_daily(ds) -> dict[str, np.ndarray]:
    """Per-variable daily aggregation (numpy — small ERA5 grids)."""
    daily: dict[str, np.ndarray] = {}

    if "t2m" in ds:
        t2m_c = np.asarray(ds["t2m"].values, dtype=np.float32) - 273.15
        daily["t2m_max_C"] = t2m_c.max(axis=0)
        daily["t2m_min_C"] = t2m_c.min(axis=0)
        daily["t2m_mean_C"] = t2m_c.mean(axis=0)

    if "d2m" in ds:
        d2m_c = np.asarray(ds["d2m"].values, dtype=np.float32) - 273.15
        daily["d2m_mean_C"] = d2m_c.mean(axis=0)

    if "sp" in ds:
        daily["sp_mean"] = np.asarray(ds["sp"].values, dtype=np.float32).mean(axis=0)

    if "u10" in ds and "v10" in ds:
        u = np.asarray(ds["u10"].values, dtype=np.float32)
        v = np.asarray(ds["v10"].values, dtype=np.float32)
        u_mean, v_mean = u.mean(axis=0), v.mean(axis=0)
        daily["wind_speed_mean"] = np.sqrt(u_mean**2 + v_mean**2)
        wind_dir_rad = np.arctan2(v_mean, u_mean)
        daily["wind_dir_sin"] = np.sin(wind_dir_rad)
        daily["wind_dir_cos"] = np.cos(wind_dir_rad)

    if "i10fg" in ds:
        daily["wind_gust_max"] = np.asarray(ds["i10fg"].values, dtype=np.float32).max(axis=0)

    if "tp" in ds:
        daily["precip_sum_mm"] = np.asarray(ds["tp"].values, dtype=np.float32).sum(axis=0) * 1000.0

    for var, out_name in [("swvl1", "soil_water_l1_mean"), ("swvl2", "soil_water_l2_mean")]:
        if var in ds:
            daily[out_name] = np.asarray(ds[var].values, dtype=np.float32).mean(axis=0)

    for var, out_name in [
        ("cvh", "high_veg_cover"),
        ("cvl", "low_veg_cover"),
        ("lai_hv", "lai_high_veg"),
        ("lai_lv", "lai_low_veg"),
    ]:
        if var in ds:
            daily[out_name] = np.asarray(ds[var].values, dtype=np.float32).mean(axis=0)

    if "blh" in ds:
        blh = np.asarray(ds["blh"].values, dtype=np.float32)
        daily["blh_mean"] = blh.mean(axis=0)
        daily["blh_max"] = blh.max(axis=0)

    if "t2m_mean_C" in daily and "d2m_mean_C" in daily:
        t = daily["t2m_mean_C"]
        td = daily["d2m_mean_C"]
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
        ea = 0.6108 * np.exp(17.27 * td / (td + 237.3))
        daily["vpd"] = es - ea

    return daily


# ---------------------------------------------------------------------------
# DEM — memory-safe download + warp (from M2 cell 78)
# ---------------------------------------------------------------------------
def download_full_resolution(gcs_relative_path: str, local_filename: str) -> str:
    """Download GCS object to data/cache/dem/ (skip if non-empty file already exists)."""
    local_path = config.CACHE_DEM_DIR / local_filename
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"  DEM disk-cache hit: {local_path.name}")
        return str(local_path)
    fs = get_fs()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(gcs_relative_path, "rb") as src_f:
        with open(local_path, "wb") as dst_f:
            chunk_size = 50 * 1024 * 1024
            while True:
                chunk = src_f.read(chunk_size)
                if not chunk:
                    break
                dst_f.write(chunk)
    return str(local_path)


def load_and_regrid_dem_features(reference_da) -> dict[str, np.ndarray]:
    """
    Reproject each DEM layer onto the FIRMS reference grid.
    Prefer full local download under data/cache/dem/ (kept for rebuilds).
    Also caches the FIRMS-grid arrays as dem_regrid_{H}x{W}.npz so rebuilds skip warp.
    """
    dst_crs = reference_da.rio.crs
    dst_transform = reference_da.rio.transform()
    dst_height = int(reference_da.sizes["y"])
    dst_width = int(reference_da.sizes["x"])

    regrid_npz = config.CACHE_DEM_DIR / f"dem_regrid_{dst_height}x{dst_width}.npz"
    if regrid_npz.exists() and regrid_npz.stat().st_size > 0:
        print(f"  DEM regrid cache hit: {regrid_npz.name}", flush=True)
        with np.load(regrid_npz) as z:
            return {name: z[name].astype(np.float32, copy=False) for name in config.DEM_TERRAIN_FILES}

    regridded: dict[str, np.ndarray] = {}
    for feature_name, filename in config.DEM_TERRAIN_FILES.items():
        rel = f"{config.GCS_BUCKET}/{config.PREFIX_DEM_TERRAIN}/{filename}"
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                local_path = download_full_resolution(rel, filename)
                print(
                    f"  DEM {feature_name}: warping {filename} → "
                    f"{dst_height}x{dst_width} (attempt {attempt})...",
                    flush=True,
                )
                with rasterio.open(local_path) as src:
                    dst_array = np.empty((dst_height, dst_width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst_array,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=WarpResampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=np.nan,
                    )
                regridded[feature_name] = dst_array
                print(
                    f"  DEM {feature_name}: done (attempt {attempt}), "
                    f"range [{np.nanmin(dst_array):.2f}, {np.nanmax(dst_array):.2f}]",
                    flush=True,
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print(f"  DEM {feature_name}: attempt {attempt} failed: {exc}")
                # Corrupt cache file — remove and retry download
                cached = config.CACHE_DEM_DIR / filename
                if cached.exists():
                    try:
                        os.remove(cached)
                    except OSError:
                        pass
                gc.collect()
        if last_err is not None:
            raise RuntimeError(f"DEM {feature_name} failed after retries") from last_err

    try:
        np.savez_compressed(regrid_npz, **regridded)
        print(f"  DEM regrid cache saved: {regrid_npz.name}", flush=True)
    except OSError as exc:
        print(f"  DEM regrid cache save skipped: {exc}", flush=True)
    return regridded

def clip_da_to_bounds(da, bounds: dict):
    return da.rio.clip_box(
        minx=bounds["west"],
        miny=bounds["south"],
        maxx=bounds["east"],
        maxy=bounds["north"],
    )


def get_firms_reference(date_str: str = "2024-08-15", bounds: Optional[dict] = None):
    da = load_firms_raster(date_str)
    if bounds is not None:
        da = clip_da_to_bounds(da, bounds)
    return da
