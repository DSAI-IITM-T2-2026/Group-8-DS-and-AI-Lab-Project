from __future__ import annotations

import calendar
import logging
import tempfile
import zipfile
from datetime import date, timedelta
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


def _is_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def _open_era5_file(path: Path) -> xr.Dataset:
    """Open monthly ZIP-of-NCs or a plain daily/monthly NetCDF."""
    if _is_zip(path):
        with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            members = sorted(Path(td).glob("*.nc"))
            if not members:
                raise FileNotFoundError(f"No .nc members inside {path}")
            datasets = [xr.open_dataset(m) for m in members]
            ds = xr.merge(datasets, compat="override", join="exact")
            ds = ds.load()
            for d in datasets:
                d.close()
            return ds
    ds = xr.open_dataset(path)
    try:
        return ds.load()
    finally:
        ds.close()


def _open_era5_zip(path: Path) -> xr.Dataset:
    """Back-compat alias."""
    return _open_era5_file(path)


def _gcs_blob_exists(uri: str) -> bool:
    if not uri.startswith("gs://"):
        return Path(uri).exists()
    try:
        from google.cloud.storage import Client

        without = uri.replace("gs://", "", 1)
        bucket_name, blob_name = without.split("/", 1)
        client = Client()
        return client.bucket(bucket_name).blob(blob_name).exists()
    except Exception:
        import subprocess

        proc = subprocess.run(
            ["gsutil", "-q", "stat", uri],
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0


def _gcs_download(uri: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        from google.cloud.storage import Client

        without = uri.replace("gs://", "", 1)
        bucket_name, blob_name = without.split("/", 1)
        Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local))
    except Exception:
        import subprocess

        subprocess.check_call(["gsutil", "-q", "cp", uri, str(local)])


def monthly_candidate_uris(year: int, month: int, gcs_prefix: str) -> list[str]:
    """Resolve monthly ERA5 object paths used by training + daily ops.

    Layouts tried (in order):
      gs://…/era5/YYYY/era5_YYYY_MM.nc
      gs://…/era5/raw/YYYY/era5_YYYY_MM.nc   ← existing wildfire-detection-first
    """
    stem = f"era5_{year}_{month:02d}.nc"
    base = gcs_prefix.rstrip("/")
    # If prefix already ends with /raw, don't double it.
    if base.endswith("/raw"):
        return [f"{base}/{year}/{stem}", f"{base[:-4]}/{year}/{stem}"]
    return [
        f"{base}/{year}/{stem}",
        f"{base}/raw/{year}/{stem}",
    ]


def daily_candidate_uri(day: date, gcs_prefix: str) -> str:
    stem = f"era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
    base = gcs_prefix.rstrip("/")
    if base.endswith("/raw"):
        base = base[:-4]
    return f"{base}/{day.year:04d}/{stem}"


def download_era5_month(
    year: int,
    month: int,
    gcs_prefix: str,
    cache_dir: Path,
    *,
    legacy_prefix: str | None = None,
) -> Path:
    """Download one monthly ERA5 zip/.nc into cache if missing.

    Raises FileNotFoundError if no monthly object exists (caller may stitch dailies).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"era5_{year}_{month:02d}.nc"
    if local.exists() and local.stat().st_size > 0:
        return local

    candidates = monthly_candidate_uris(year, month, gcs_prefix)
    if legacy_prefix:
        candidates.extend(monthly_candidate_uris(year, month, legacy_prefix))

    for uri in candidates:
        if not _gcs_blob_exists(uri):
            continue
        logger.info("Downloading monthly ERA5 %s → %s", uri, local)
        _gcs_download(uri, local)
        return local

    raise FileNotFoundError(
        f"No monthly ERA5 for {year}-{month:02d}. Tried: {candidates}"
    )


def download_era5_day_nc(
    day: date,
    gcs_prefix: str,
    cache_dir: Path,
    *,
    legacy_prefix: str | None = None,
) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
    if local.exists() and local.stat().st_size > 0:
        return local

    candidates = [daily_candidate_uri(day, gcs_prefix)]
    if legacy_prefix:
        candidates.append(daily_candidate_uri(day, legacy_prefix))
        # legacy monthly-style daily under raw/
        candidates.append(
            f"{legacy_prefix.rstrip('/')}/{day.year:04d}/"
            f"era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
        )

    for uri in candidates:
        if not _gcs_blob_exists(uri):
            continue
        logger.info("Downloading daily ERA5 %s → %s", uri, local)
        _gcs_download(uri, local)
        return local
    return None


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
    return df


def _days_in_month(year: int, month: int) -> list[date]:
    n = calendar.monthrange(year, month)[1]
    return [date(year, month, d) for d in range(1, n + 1)]


def _build_month_from_daily_files(
    year: int,
    month: int,
    gcs_prefix: str,
    raw_cache: Path,
    *,
    legacy_prefix: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stitch daily NetCDFs into a month frame (only days that exist)."""
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for day in _days_in_month(year, month):
        ts = pd.Timestamp(day)
        if start is not None and ts < start.normalize():
            continue
        if end is not None and ts > end.normalize():
            continue
        path = download_era5_day_nc(
            day, gcs_prefix, raw_cache, legacy_prefix=legacy_prefix
        )
        if path is None:
            missing.append(day.isoformat())
            continue
        ds = _open_era5_file(path)
        frames.append(month_to_daily_frame(ds))

    if not frames:
        raise FileNotFoundError(
            f"No daily ERA5 files for {year}-{month:02d} under {gcs_prefix} "
            f"(missing example days: {missing[:5]}). "
            f"Run download for those dates, or place monthly "
            f"era5/raw/{year}/era5_{year}_{month:02d}.nc in the bucket."
        )
    if missing:
        logger.warning(
            "ERA5 %04d-%02d: %d day(s) missing in GCS (using %d available): %s%s",
            year,
            month,
            len(missing),
            len(frames),
            ", ".join(missing[:8]),
            "…" if len(missing) > 8 else "",
        )
    return pd.concat(frames, ignore_index=True)


def _unique_dates(df: pd.DataFrame) -> set[date]:
    if df.empty:
        return set()
    return {pd.Timestamp(x).date() for x in pd.to_datetime(df["date"]).unique()}


def _needed_month_days(
    year: int, month: int, start: pd.Timestamp, end: pd.Timestamp
) -> list[date]:
    start_d = pd.Timestamp(start).normalize().date()
    end_d = pd.Timestamp(end).normalize().date()
    return [d for d in _days_in_month(year, month) if start_d <= d <= end_d]


def _fill_month_from_dailies(
    month_df: pd.DataFrame | None,
    year: int,
    month: int,
    gcs_prefix: str,
    raw_cache: Path,
    *,
    legacy_prefix: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Append daily NetCDFs for days missing from a (possibly partial) monthly frame."""
    needed = _needed_month_days(year, month, start, end)
    have = _unique_dates(month_df) if month_df is not None else set()
    missing = [d for d in needed if d not in have]
    if not missing:
        if month_df is None:
            raise FileNotFoundError(f"No ERA5 for {year}-{month:02d}")
        return month_df

    if month_df is not None:
        logger.info(
            "Monthly ERA5 %04d-%02d missing %d needed day(s); stitching dailies: %s%s",
            year,
            month,
            len(missing),
            ", ".join(d.isoformat() for d in missing[:8]),
            "…" if len(missing) > 8 else "",
        )
    else:
        logger.info(
            "No monthly ERA5 for %04d-%02d; stitching daily NetCDFs",
            year,
            month,
        )

    try:
        daily_df = _build_month_from_daily_files(
            year,
            month,
            gcs_prefix,
            raw_cache,
            legacy_prefix=legacy_prefix,
            start=pd.Timestamp(min(missing)),
            end=pd.Timestamp(max(missing)),
        )
    except FileNotFoundError:
        if month_df is None:
            raise
        logger.warning(
            "ERA5 %04d-%02d: monthly file is partial and no daily NetCDFs for missing days",
            year,
            month,
        )
        return month_df

    if month_df is None:
        return daily_df
    out = pd.concat([month_df, daily_df], ignore_index=True)
    return out.drop_duplicates(subset=["date", "cell_id"], keep="first")


def build_era5_daily_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    gcs_prefix: str,
    raw_cache: Path,
    daily_cache: Path,
    *,
    legacy_prefix: str | None = None,
) -> pd.DataFrame:
    """Return daily ERA5 features for [start, end], caching per month.

    Resolution order per month:
      1. Cached parquet (rebuilt if it is missing days in [start, end])
      2. Monthly object (era5/YYYY/… or era5/raw/YYYY/…)
      3. Stitch daily era5_YYYY_MM_DD.nc for any days the monthly file does not cover
    """
    daily_cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    for period in months:
        year, month = period.year, period.month
        cache_path = daily_cache / f"era5_daily_{year}_{month:02d}.parquet"
        needed = set(_needed_month_days(year, month, start, end))
        month_df: pd.DataFrame | None = None

        if cache_path.exists():
            month_df = pd.read_parquet(cache_path)
            have = _unique_dates(month_df)
            if needed <= have:
                logger.info("ERA5 daily cache hit %s", cache_path.name)
                mask = (month_df["date"] >= start) & (month_df["date"] <= end)
                frames.append(month_df.loc[mask])
                continue
            logger.info(
                "ERA5 daily cache %s missing %d needed day(s); filling from daily NetCDFs",
                cache_path.name,
                len(needed - have),
            )
        else:
            try:
                raw = download_era5_month(
                    year,
                    month,
                    gcs_prefix,
                    raw_cache,
                    legacy_prefix=legacy_prefix,
                )
                ds = _open_era5_file(raw)
                month_df = month_to_daily_frame(ds)
                logger.info("Built ERA5 month from monthly file %04d-%02d", year, month)
            except FileNotFoundError:
                month_df = None

        month_df = _fill_month_from_dailies(
            month_df,
            year,
            month,
            gcs_prefix,
            raw_cache,
            legacy_prefix=legacy_prefix,
            start=start,
            end=end,
        )
        month_df.to_parquet(cache_path, index=False)
        logger.info("Wrote %s (%d rows)", cache_path.name, len(month_df))

        mask = (month_df["date"] >= start) & (month_df["date"] <= end)
        frames.append(month_df.loc[mask])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "cell_id"]).reset_index(drop=True)
