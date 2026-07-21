from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .cells import make_cell_id

logger = logging.getLogger(__name__)


def _relative_humidity(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    t_c = t2m_k - 273.15
    d_c = d2m_k - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * d_c) / (d_c + 243.5))
    return np.clip((e / es) * 100.0, 0.0, 100.0)


def _open_era5_zip(path: Path) -> xr.Dataset:
    """ERA5 monthly files on GCS are ZIP archives containing instant + accum NetCDFs."""
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as td:
        zf.extractall(td)
        members = sorted(Path(td).glob("*.nc"))
        if not members:
            raise FileNotFoundError(f"No .nc members inside {path}")
        datasets = [xr.open_dataset(m) for m in members]
        ds = xr.merge(datasets, compat="override", join="exact")
        # Load into memory before temp dir cleanup
        ds = ds.load()
        for d in datasets:
            d.close()
        return ds


def download_era5_month(
    year: int,
    month: int,
    gcs_prefix: str,
    cache_dir: Path,
) -> Path:
    """Download one monthly ERA5 zip/.nc into cache if missing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"era5_{year}_{month:02d}.nc"
    if local.exists() and local.stat().st_size > 0:
        return local

    uri = f"{gcs_prefix.rstrip('/')}/{year}/era5_{year}_{month:02d}.nc"
    logger.info("Downloading %s → %s", uri, local)
    try:
        from google.cloud.storage import Client

        # gs://bucket/path...
        without = uri.replace("gs://", "", 1)
        bucket_name, blob_name = without.split("/", 1)
        client = Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).download_to_filename(str(local))
    except Exception:
        # Fallback: gsutil
        import subprocess

        subprocess.check_call(["gsutil", "-q", "cp", uri, str(local)])
    return local


def month_to_daily_frame(ds: xr.Dataset) -> pd.DataFrame:
    """Aggregate hourly ERA5 to daily cell features (Report §3.5 rules)."""
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("ERA5 dataset missing time coordinate")

    mean_ds = ds.resample(time="1D").mean()
    max_ds = ds.resample(time="1D").max()
    min_ds = ds.resample(time="1D").min()
    sum_ds = ds.resample(time="1D").sum()

    merged = xr.Dataset(
        {
            "t2m_mean": mean_ds["t2m"],
            "t2m_max": max_ds["t2m"],
            "t2m_min": min_ds["t2m"],
            "d2m_mean": mean_ds["d2m"],
            "sp_mean": mean_ds["sp"],
            "u10_mean": mean_ds["u10"],
            "v10_mean": mean_ds["v10"],
            "i10fg_max": max_ds["i10fg"],
            "swvl1_mean": mean_ds["swvl1"],
            "swvl2_mean": mean_ds["swvl2"],
            "cvh_mean": mean_ds["cvh"],
            "cvl_mean": mean_ds["cvl"],
            "lai_hv_mean": mean_ds["lai_hv"],
            "lai_lv_mean": mean_ds["lai_lv"],
            "blh_mean": mean_ds["blh"],
            # precip: daily sum of hourly accumulations, metres → mm
            "tp_sum_mm": sum_ds["tp"] * 1000.0,
        }
    )
    # Vector-mean wind from daily-mean u/v
    merged["wind_speed_mean"] = np.sqrt(merged["u10_mean"] ** 2 + merged["v10_mean"] ** 2)
    direction = (180.0 / np.pi) * np.arctan2(merged["u10_mean"], merged["v10_mean"]) + 180.0
    merged["wind_dir_sin"] = np.sin(np.deg2rad(direction))
    merged["wind_dir_cos"] = np.cos(np.deg2rad(direction))
    merged["rh_mean"] = xr.apply_ufunc(
        _relative_humidity,
        merged["t2m_mean"],
        merged["d2m_mean"],
        dask="forbidden",
    )
    merged["soil_moisture_index"] = (merged["swvl1_mean"] + merged["swvl2_mean"]) / 2.0

    df = merged.to_dataframe().reset_index()
    df = df.rename(columns={"time": "date", "latitude": "latitude", "longitude": "longitude"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["cell_id"] = [
        make_cell_id(la, lo) for la, lo in zip(df["latitude"], df["longitude"])
    ]
    # Drop helper components not needed downstream (keep means for debugging)
    return df


def build_era5_daily_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    gcs_prefix: str,
    raw_cache: Path,
    daily_cache: Path,
) -> pd.DataFrame:
    """Return daily ERA5 features for [start, end], caching per month."""
    daily_cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    for period in months:
        year, month = period.year, period.month
        cache_path = daily_cache / f"era5_daily_{year}_{month:02d}.parquet"
        if cache_path.exists():
            logger.info("ERA5 daily cache hit %s", cache_path.name)
            month_df = pd.read_parquet(cache_path)
        else:
            raw = download_era5_month(year, month, gcs_prefix, raw_cache)
            ds = _open_era5_zip(raw)
            month_df = month_to_daily_frame(ds)
            month_df.to_parquet(cache_path, index=False)
            logger.info("Wrote %s (%d rows)", cache_path.name, len(month_df))

        mask = (month_df["date"] >= start) & (month_df["date"] <= end)
        frames.append(month_df.loc[mask])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "cell_id"]).reset_index(drop=True)
