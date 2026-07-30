from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..grid import make_cell_id
from ..io import atomic_parquet, gcs_copy

logger = logging.getLogger(__name__)


def relative_humidity(t2m_kelvin: np.ndarray, d2m_kelvin: np.ndarray) -> np.ndarray:
    temperature_c = t2m_kelvin - 273.15
    dewpoint_c = d2m_kelvin - 273.15
    saturation = 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    vapour = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    return np.clip((vapour / saturation) * 100.0, 0.0, 100.0)


def open_era5_archive(path: Path):
    """Open the project's monthly ZIP-disguised-as-.nc archive or a normal NetCDF."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Building raw ERA5 requires xarray and netCDF4. Install the project "
            "requirements before running the era5_firms stage."
        ) from exc
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path).load()
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as temp_dir:
        archive.extractall(temp_dir)
        members = sorted(Path(temp_dir).glob("*.nc"))
        if not members:
            raise FileNotFoundError(f"No NetCDF members inside {path}")
        datasets = [xr.open_dataset(member) for member in members]
        try:
            return xr.merge(datasets, compat="override", join="exact").load()
        finally:
            for dataset in datasets:
                dataset.close()


def month_to_daily_frame(dataset) -> pd.DataFrame:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Aggregating ERA5 requires xarray. Install the project requirements."
        ) from exc
    if "valid_time" in dataset.coords:
        dataset = dataset.rename({"valid_time": "time"})
    if "time" not in dataset.coords and "time" not in dataset.dims:
        raise ValueError("ERA5 dataset is missing its time coordinate")
    required = {
        "t2m",
        "d2m",
        "sp",
        "u10",
        "v10",
        "i10fg",
        "swvl1",
        "swvl2",
        "cvh",
        "cvl",
        "lai_hv",
        "lai_lv",
        "blh",
        "tp",
    }
    missing = required - set(dataset.data_vars)
    if missing:
        raise ValueError(f"ERA5 archive missing variables: {sorted(missing)}")

    daily_mean = dataset.resample(time="1D").mean()
    daily_max = dataset.resample(time="1D").max()
    daily_min = dataset.resample(time="1D").min()
    daily_sum = dataset.resample(time="1D").sum()
    merged = xr.Dataset(
        {
            "t2m_mean": daily_mean["t2m"],
            "t2m_max": daily_max["t2m"],
            "t2m_min": daily_min["t2m"],
            "d2m_mean": daily_mean["d2m"],
            "sp_mean": daily_mean["sp"],
            "u10_mean": daily_mean["u10"],
            "v10_mean": daily_mean["v10"],
            "i10fg_max": daily_max["i10fg"],
            "swvl1_mean": daily_mean["swvl1"],
            "swvl2_mean": daily_mean["swvl2"],
            "cvh_mean": daily_mean["cvh"],
            "cvl_mean": daily_mean["cvl"],
            "lai_hv_mean": daily_mean["lai_hv"],
            "lai_lv_mean": daily_mean["lai_lv"],
            "blh_mean": daily_mean["blh"],
            "tp_sum_mm": daily_sum["tp"] * 1000.0,
        }
    )
    merged["wind_speed_mean"] = np.sqrt(merged["u10_mean"] ** 2 + merged["v10_mean"] ** 2)
    direction = np.rad2deg(np.arctan2(merged["u10_mean"], merged["v10_mean"])) + 180.0
    merged["wind_dir_sin"] = np.sin(np.deg2rad(direction))
    merged["wind_dir_cos"] = np.cos(np.deg2rad(direction))
    merged["rh_mean"] = xr.apply_ufunc(
        relative_humidity,
        merged["t2m_mean"],
        merged["d2m_mean"],
        dask="forbidden",
    )
    merged["soil_moisture_index"] = (merged["swvl1_mean"] + merged["swvl2_mean"]) / 2.0

    frame = merged.to_dataframe().reset_index()
    frame = frame.rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["cell_id"] = [
        make_cell_id(latitude, longitude)
        for latitude, longitude in zip(frame["latitude"], frame["longitude"])
    ]
    if "number" in frame:
        frame = frame.drop(columns=["number"])
    if frame.duplicated(["cell_id", "date"]).any():
        raise ValueError("ERA5 daily aggregation produced duplicate cell-day rows")
    return frame.sort_values(["date", "cell_id"]).reset_index(drop=True)


def raw_cache_path(cfg: dict, year: int, month: int) -> Path:
    return cfg["paths"]["cache_dir"] / "era5_raw" / f"era5_{year}_{month:02d}.nc"


def daily_cache_path(cfg: dict, year: int, month: int) -> Path:
    return cfg["paths"]["cache_dir"] / "era5_daily" / f"year={year}" / f"month={month:02d}.parquet"


def _seed_file(cfg: dict, year: int, month: int) -> Path | None:
    names = [
        f"era5_daily_{year}_{month:02d}.parquet",
        f"year={year}/month={month:02d}.parquet",
    ]
    for directory in cfg["paths"].get("era5_daily_seed_dirs", []):
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def build_era5_month(cfg: dict, year: int, month: int, force: bool = False) -> Path:
    destination = daily_cache_path(cfg, year, month)
    if destination.exists() and not force:
        return destination
    seed = _seed_file(cfg, year, month)
    if seed is not None:
        frame = pd.read_parquet(seed)
        atomic_parquet(frame, destination)
        logger.info("Seeded ERA5 %04d-%02d from %s", year, month, seed)
        return destination

    raw = raw_cache_path(cfg, year, month)
    if not raw.exists() or force:
        uri = f"{cfg['gcs']['era5_prefix'].rstrip('/')}/{year}/era5_{year}_{month:02d}.nc"
        logger.info("Downloading %s", uri)
        gcs_copy(uri, raw)
    dataset = open_era5_archive(raw)
    try:
        frame = month_to_daily_frame(dataset)
    finally:
        dataset.close()
    atomic_parquet(frame, destination)
    logger.info("Built ERA5 daily shard %s (%d rows)", destination, len(frame))
    return destination


def read_era5_range(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    periods = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    frames = []
    for period in periods:
        path = daily_cache_path(cfg, period.year, period.month)
        if not path.exists():
            raise FileNotFoundError(f"Missing ERA5 daily shard: {path}")
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    mask = combined["date"].between(start.normalize(), end.normalize())
    return combined.loc[mask].sort_values(["cell_id", "date"]).reset_index(drop=True)
