"""Local-first GCS fetch: reuse teammate caches, download only gaps."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_HIVE_RE = re.compile(r"year=(\d+)/month=(\d+)/window=(\d+)/features\.csv$")
_MM_S2_RE = re.compile(r"y(\d{4})_m(\d{2})_w(\d+)\.parquet$")
_MM_S5P_RE = re.compile(r"y(\d{4})_m(\d{2})_w(\d+)\.parquet$")


def _gcs_client(anonymous: bool = True):
    from google.cloud.storage import Client

    if anonymous:
        try:
            return Client.create_anonymous_client()
        except Exception:
            pass
    try:
        return Client()
    except Exception:
        return Client.create_anonymous_client()


def _parse_gs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), uri
    path = uri[5:]
    bucket, _, blob = path.partition("/")
    return bucket, blob


def download_gs_file(gs_uri: str, dest: Path, skip_existing: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return dest
    bucket_name, blob_name = _parse_gs(gs_uri)
    client = _gcs_client(anonymous=True)
    blob = client.bucket(bucket_name).blob(blob_name)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    logger.info("Downloading %s → %s", gs_uri, dest)
    try:
        blob.download_to_filename(str(tmp))
    except Exception:
        client = _gcs_client(anonymous=False)
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(str(tmp))
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# FIRMS
# ---------------------------------------------------------------------------
def firms_local_path(date_str: str) -> Path:
    return config.CACHE_FIRMS / f"{date_str}.tif"


def ensure_firms(date_str: str) -> Path:
    dest = firms_local_path(date_str)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    uri = f"gs://{config.FIRMS_BUCKET}/{config.FIRMS_PREFIX}/{date_str}.tif"
    return download_gs_file(uri, dest)


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------
def era5_month_filename(year: int, month: int) -> str:
    return f"era5_{year}_{month:02d}.nc"


def era5_local_path(year: int, month: int) -> Path:
    return config.CACHE_ERA5 / str(year) / era5_month_filename(year, month)


def ensure_era5_month(year: int, month: int) -> Path:
    dest = era5_local_path(year, month)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # Teammate cache
    mvp = config.MVP_ERA5_RAW / era5_month_filename(year, month)
    if mvp.exists() and mvp.stat().st_size > 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(mvp.resolve())
        except OSError:
            shutil.copy2(mvp, dest)
        logger.info("Linked ERA5 from mvp cache: %s", mvp.name)
        return dest
    uri = (
        f"gs://{config.ERA5_BUCKET}/{config.ERA5_PREFIX}/{year}/"
        f"{era5_month_filename(year, month)}"
    )
    return download_gs_file(uri, dest)


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------
def dem_local_path(name: str) -> Path:
    return config.CACHE_DEM / config.DEM_FILES[name]


def ensure_dem() -> dict[str, Path]:
    out = {}
    for key, fname in config.DEM_FILES.items():
        dest = dem_local_path(key)
        if not (dest.exists() and dest.stat().st_size > 0):
            uri = f"gs://{config.DEM_BUCKET}/{config.DEM_PREFIX}/{fname}"
            download_gs_file(uri, dest)
        out[key] = dest
    return out


# ---------------------------------------------------------------------------
# S2 / S5P numerical Hive → parquet cache
# ---------------------------------------------------------------------------
def _hive_parquet_name(year: int, month: int, window: int) -> str:
    return f"y{year}_m{month:02d}_w{window:03d}.parquet"


def _mm_cache_lookup(cache_dir: Path, year: int, month: int, window: int) -> Path | None:
    cand = cache_dir / _hive_parquet_name(year, month, window)
    if cand.exists():
        return cand
    # multimodal sometimes uses unpadded window
    for p in cache_dir.glob(f"y{year}_m{month:02d}_w*.parquet"):
        m = _MM_S2_RE.match(p.name) or _MM_S5P_RE.match(p.name)
        if m and int(m.group(3)) == window:
            return p
    return None


def _cache_features_csv(
    bucket: str,
    blob_name: str,
    dest: Path,
    columns: list[str],
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    client = _gcs_client(anonymous=True)
    blob = client.bucket(bucket).blob(blob_name)
    tmp_csv = dest.with_suffix(".csv.partial")
    logger.info("Downloading gs://%s/%s", bucket, blob_name)
    try:
        blob.download_to_filename(str(tmp_csv))
    except Exception:
        client = _gcs_client(anonymous=False)
        blob = client.bucket(bucket).blob(blob_name)
        blob.download_to_filename(str(tmp_csv))
    base = ["grid_id", "latitude", "longitude", "window_start", "window_end"]
    usecols = list(dict.fromkeys(base + columns))
    try:
        df = pd.read_csv(tmp_csv, usecols=lambda c: c in usecols)
    except ValueError:
        df = pd.read_csv(tmp_csv)
        df = df[[c for c in usecols if c in df.columns]]
    df["window_start"] = pd.to_datetime(df["window_start"]).dt.normalize()
    df["window_end"] = pd.to_datetime(df["window_end"]).dt.normalize()
    df.to_parquet(dest, index=False)
    try:
        tmp_csv.unlink()
    except OSError:
        pass
    return dest


def list_hive_windows(bucket: str, prefix: str, year: int) -> list[tuple[int, int, int, str]]:
    """Return (year, month, window, blob_name) for one year."""
    client = _gcs_client(anonymous=True)
    full = prefix.rstrip("/") + f"/year={year}/"
    rows = []
    try:
        iterator = client.list_blobs(bucket, prefix=full)
    except Exception:
        client = _gcs_client(anonymous=False)
        iterator = client.list_blobs(bucket, prefix=full)
    for blob in iterator:
        m = _HIVE_RE.search(blob.name)
        if not m:
            continue
        y, mo, w = map(int, m.groups())
        rows.append((y, mo, w, blob.name))
    rows.sort()
    return rows


def ensure_s2_window(year: int, month: int, window: int) -> Path | None:
    dest = config.CACHE_S2 / _hive_parquet_name(year, month, window)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    mm = _mm_cache_lookup(config.MM_S2_CACHE, year, month, window)
    if mm is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(mm.resolve())
        except OSError:
            shutil.copy2(mm, dest)
        return dest
    if year not in config.S2_BUCKET_BY_YEAR:
        logger.warning("No S2 bucket mapping for year %s", year)
        return None
    bucket, prefix = config.S2_BUCKET_BY_YEAR[year]
    blob = f"{prefix}/year={year}/month={month:02d}/window={window:03d}/features.csv"
    try:
        return _cache_features_csv(bucket, blob, dest, config.S2_COLUMNS)
    except Exception as e:
        logger.warning("S2 fetch failed %s: %s", blob, e)
        return None


def ensure_s5p_window(year: int, month: int, window: int) -> Path | None:
    dest = config.CACHE_S5P / _hive_parquet_name(year, month, window)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    mm = _mm_cache_lookup(config.MM_S5P_CACHE, year, month, window)
    if mm is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(mm.resolve())
        except OSError:
            shutil.copy2(mm, dest)
        return dest
    if year not in config.S5P_BUCKET_BY_YEAR:
        logger.warning("No S5P bucket mapping for year %s", year)
        return None
    bucket, prefix = config.S5P_BUCKET_BY_YEAR[year]
    blob = f"{prefix}/year={year}/month={month:02d}/window={window:03d}/features.csv"
    try:
        return _cache_features_csv(bucket, blob, dest, config.S5P_COLUMNS)
    except Exception as e:
        logger.warning("S5P fetch failed %s: %s", blob, e)
        return None


def index_local_s2() -> pd.DataFrame:
    rows = []
    for p in sorted(config.CACHE_S2.glob("y*_m*_w*.parquet")):
        m = _MM_S2_RE.match(p.name)
        if not m:
            continue
        rows.append({"year": int(m.group(1)), "month": int(m.group(2)), "window": int(m.group(3)), "path": str(p)})
    # also index multimodal cache without copying
    if config.MM_S2_CACHE.exists():
        for p in sorted(config.MM_S2_CACHE.glob("y*_m*_w*.parquet")):
            m = _MM_S2_RE.match(p.name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if any(r["year"] == key[0] and r["month"] == key[1] and r["window"] == key[2] for r in rows):
                continue
            rows.append({"year": key[0], "month": key[1], "window": key[2], "path": str(p)})
    return pd.DataFrame(rows)


def index_local_s5p() -> pd.DataFrame:
    rows = []
    for cache in (config.CACHE_S5P, config.MM_S5P_CACHE):
        if not cache.exists():
            continue
        for p in sorted(cache.glob("y*_m*_w*.parquet")):
            m = _MM_S5P_RE.match(p.name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if any(r["year"] == key[0] and r["month"] == key[1] and r["window"] == key[2] for r in rows):
                continue
            rows.append({"year": key[0], "month": key[1], "window": key[2], "path": str(p)})
    return pd.DataFrame(rows)


def prefetch_era5_range(start: str, end: str) -> None:
    dates = pd.date_range(start, end, freq="D")
    months = sorted({(d.year, d.month) for d in dates})
    # also need lag lookback before start
    lookback = pd.Timestamp(start) - pd.Timedelta(days=config.FEATURE_LAG_DAYS + config.HISTORY_DAYS + 2)
    for d in pd.date_range(lookback, start, freq="MS"):
        months.append((d.year, d.month))
    for y, m in sorted(set(months)):
        try:
            ensure_era5_month(y, m)
        except Exception as e:
            logger.warning("ERA5 %s-%02d: %s", y, m, e)
