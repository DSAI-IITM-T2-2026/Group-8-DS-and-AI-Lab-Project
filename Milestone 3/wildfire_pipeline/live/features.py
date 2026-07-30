"""Build lag-consistent daily feature stacks for label day D."""
from __future__ import annotations

import io
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .gcs_fetch import (
    ensure_era5_month,
    ensure_s2_window,
    ensure_s5p_window,
    index_local_s2,
    index_local_s5p,
)
from .regrid import (
    build_land_mask,
    load_dem_on_grid,
    rasterize_points_to_grid,
    reference_da,
    regrid_era5_var_to_reference,
)

logger = logging.getLogger(__name__)


def _as_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, pd.Timestamp):
        return d
    return pd.Timestamp(d).date()


def feature_end_for_label(label_day) -> date:
    """Newest allowed feature day for predicting label_day (= D - 2)."""
    return _as_date(label_day) - timedelta(days=config.FEATURE_LAG_DAYS)


def history_dates(label_day) -> list[date]:
    """D-8 … D-2 inclusive (7 days)."""
    end = feature_end_for_label(label_day)
    return [end - timedelta(days=i) for i in range(config.HISTORY_DAYS - 1, -1, -1)]


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------
def _aggregate_era5_day(ds: xr.Dataset, day_idx: int) -> dict[str, np.ndarray]:
    """Hourly slice → daily means/sums for configured vars."""
    sl = ds.isel(time=slice(day_idx * 24, (day_idx + 1) * 24))
    out = {}
    for var in config.ERA5_DAILY_VARS:
        if var not in sl:
            continue
        arr = sl[var]
        if var == "tp":
            # precip: sum over hours (meters) → mm
            daily = arr.sum(dim="time") * 1000.0
        else:
            daily = arr.mean(dim="time")
            if var in ("t2m", "d2m"):
                daily = daily - 273.15
        out[var] = np.asarray(daily.values, dtype=np.float32)
    return out


_era5_month_cache: dict[tuple[int, int], xr.Dataset] = {}


def _open_netcdf_bytes(data: bytes) -> xr.Dataset:
    """Open one NetCDF member; try h5netcdf → netcdf4 → default."""
    last_err: Exception | None = None
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            kwargs = {"engine": engine} if engine else {}
            return xr.open_dataset(io.BytesIO(data), **kwargs)
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


def _open_era5_file(path: Path) -> xr.Dataset:
    """
    ERA5 monthly objects on GCS are ZIP archives of instant+accum NetCDFs
    (same as mvp_era5_dem / Experiments), despite the ``.nc`` suffix.
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        datasets: list[xr.Dataset] = []
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".nc")]
            if not names:
                raise FileNotFoundError(f"No .nc members inside {path}")
            for name in names:
                try:
                    datasets.append(_open_netcdf_bytes(zf.read(name)))
                except Exception:
                    # path-based fallback after extract
                    pass
        if not datasets:
            with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as td:
                zf.extractall(td)
                members = sorted(Path(td).glob("**/*.nc"))
                for m in members:
                    opened = False
                    for engine in ("h5netcdf", "netcdf4", None):
                        try:
                            kwargs = {"engine": engine} if engine else {}
                            datasets.append(xr.open_dataset(m, **kwargs))
                            opened = True
                            break
                        except Exception:
                            continue
                    if not opened:
                        raise RuntimeError(f"Could not open NetCDF member {m}")
                ds = xr.merge(datasets, compat="override", join="outer").load()
                for d in datasets:
                    d.close()
        else:
            ds = xr.merge(datasets, compat="override", join="outer").load()
            for d in datasets:
                d.close()
    else:
        last_err: Exception | None = None
        ds = None
        for engine in ("h5netcdf", "netcdf4", None):
            try:
                kwargs = {"engine": engine} if engine else {}
                ds = xr.open_dataset(path, **kwargs).load()
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise RuntimeError(f"Failed to open ERA5 {path}: {last_err}")

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def _load_era5_month(year: int, month: int) -> xr.Dataset:
    key = (year, month)
    if key in _era5_month_cache:
        return _era5_month_cache[key]
    path = ensure_era5_month(year, month)
    ds = _open_era5_file(Path(path))
    _era5_month_cache[key] = ds
    if len(_era5_month_cache) > 3:
        old = next(iter(_era5_month_cache))
        if old != key:
            try:
                _era5_month_cache[old].close()
            except Exception:
                pass
            del _era5_month_cache[old]
    return ds


def era5_day_on_grid(day: date, reference=None) -> dict[str, np.ndarray] | None:
    reference = reference if reference is not None else reference_da()
    try:
        ds = _load_era5_month(day.year, day.month)
    except Exception as e:
        logger.warning("ERA5 open failed %s: %s", day, e)
        return None
    if "time" not in ds.dims and "time" not in ds.coords:
        logger.warning("ERA5 %s: no time dimension (dims=%s)", day, list(ds.dims))
        return None
    n_hours = int(ds.sizes.get("time", 0))
    n_days = n_hours // 24
    day_idx = day.day - 1
    if day_idx < 0 or day_idx >= n_days:
        return None
    daily = _aggregate_era5_day(ds, day_idx)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = np.asarray(ds[lat_name].values)
    lon = np.asarray(ds[lon_name].values)
    out = {}
    for k, arr in daily.items():
        out[k] = regrid_era5_var_to_reference(arr, lat, lon, reference)
    return out


def build_era5_stack(label_day, reference=None, land_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      stack: (T=7, C=7, H, W)
      valid: (T,) bool — True if day present
    """
    reference = reference if reference is not None else reference_da()
    land_mask = land_mask if land_mask is not None else build_land_mask(reference)
    h, w = int(reference.sizes["y"]), int(reference.sizes["x"])
    T, C = config.HISTORY_DAYS, config.ERA5_CHANNELS
    stack = np.zeros((T, C, h, w), dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    var_order = config.ERA5_DAILY_VARS
    prev = None
    for ti, d in enumerate(history_dates(label_day)):
        day_map = era5_day_on_grid(d, reference)
        if day_map is None:
            if prev is not None:
                stack[ti] = prev
                valid[ti] = False  # filled
            continue
        chans = []
        for v in var_order:
            arr = day_map.get(v)
            if arr is None:
                arr = np.zeros((h, w), dtype=np.float32)
            arr = np.where(land_mask, arr, 0.0)
            arr = np.nan_to_num(arr, nan=0.0)
            chans.append(arr)
        frame = np.stack(chans, axis=0)
        stack[ti] = frame
        valid[ti] = True
        prev = frame
    return stack, valid


# ---------------------------------------------------------------------------
# S5P with LOCF + age/valid
# ---------------------------------------------------------------------------
_s5p_index: pd.DataFrame | None = None
_s5p_day_cache: dict[str, pd.DataFrame] = {}


def _get_s5p_index() -> pd.DataFrame:
    global _s5p_index
    if _s5p_index is None:
        _s5p_index = index_local_s5p()
    return _s5p_index


def _s5p_path_for_day(day: date) -> Path | None:
    """S5P daily windows: window number ≈ day-of-year (1-indexed) in Hive layout."""
    # Prefer local index match by reading parquet window_end if available;
    # fallback: year/month/window=day_of_year style used in team buckets.
    idx = _get_s5p_index()
    # try ensure from GCS using day-of-year window
    doy = day.timetuple().tm_yday
    path = ensure_s5p_window(day.year, day.month, doy)
    if path is not None:
        return path
    # search local index by year/month and check window_end in file (expensive) — skip
    if not idx.empty:
        sub = idx[(idx["year"] == day.year) & (idx["month"] == day.month)]
        for _, row in sub.iterrows():
            p = Path(row["path"])
            try:
                meta = pd.read_parquet(p, columns=["window_start", "window_end"])
                if meta.empty:
                    continue
                ws = pd.Timestamp(meta["window_start"].iloc[0]).date()
                we = pd.Timestamp(meta["window_end"].iloc[0]).date()
                if ws <= day <= we:
                    return p
            except Exception:
                continue
    return None


def _load_s5p_day_frame(day: date) -> pd.DataFrame | None:
    key = day.isoformat()
    if key in _s5p_day_cache:
        return _s5p_day_cache[key]
    path = _s5p_path_for_day(day)
    if path is None:
        return None
    try:
        cols = ["latitude", "longitude"] + [c for c in config.S5P_COLUMNS]
        df = pd.read_parquet(path)
        keep = [c for c in cols if c in df.columns]
        df = df[keep]
    except Exception as e:
        logger.warning("S5P read %s: %s", path, e)
        return None
    _s5p_day_cache[key] = df
    if len(_s5p_day_cache) > 14:
        _s5p_day_cache.pop(next(iter(_s5p_day_cache)))
    return df


def build_s5p_stack(label_day, reference=None, land_mask=None) -> np.ndarray:
    """
    (T=7, C=4, H, W): AAI, CO, valid, age_days/S5P_LOCF_MAX
    LOCF within S5P_LOCF_MAX_DAYS.
    """
    reference = reference if reference is not None else reference_da()
    land_mask = land_mask if land_mask is not None else build_land_mask(reference)
    h, w = int(reference.sizes["y"]), int(reference.sizes["x"])
    T = config.HISTORY_DAYS
    stack = np.zeros((T, config.S5P_CHANNELS, h, w), dtype=np.float32)
    dates = history_dates(label_day)

    # raw maps per day (or None)
    raw_aai: list[np.ndarray | None] = []
    raw_co: list[np.ndarray | None] = []
    for d in dates:
        df = _load_s5p_day_frame(d)
        if df is None or df.empty:
            raw_aai.append(None)
            raw_co.append(None)
            continue
        aai_col = "s5p_aai_mean" if "s5p_aai_mean" in df.columns else None
        co_col = "s5p_co_mean" if "s5p_co_mean" in df.columns else None
        if aai_col is None:
            raw_aai.append(None)
            raw_co.append(None)
            continue
        aai = rasterize_points_to_grid(
            df["latitude"].to_numpy(),
            df["longitude"].to_numpy(),
            df[aai_col].to_numpy(dtype=np.float32),
            reference,
        )
        if co_col:
            co = rasterize_points_to_grid(
                df["latitude"].to_numpy(),
                df["longitude"].to_numpy(),
                df[co_col].to_numpy(dtype=np.float32),
                reference,
            )
        else:
            co = np.full_like(aai, np.nan)
        raw_aai.append(aai)
        raw_co.append(co)

    last_aai = None
    last_co = None
    last_age = None
    for ti in range(T):
        aai = raw_aai[ti]
        co = raw_co[ti]
        if aai is not None and np.isfinite(aai).any():
            last_aai = np.nan_to_num(aai, nan=0.0)
            last_co = np.nan_to_num(co if co is not None else aai * 0, nan=0.0)
            last_age = 0
            valid = 1.0
            age = 0.0
        elif last_aai is not None and last_age is not None and last_age < config.S5P_LOCF_MAX_DAYS:
            last_age += 1
            valid = 1.0
            age = float(last_age)
            # keep last_aai/co
        else:
            last_aai = np.zeros((h, w), dtype=np.float32)
            last_co = np.zeros((h, w), dtype=np.float32)
            last_age = None
            valid = 0.0
            age = float(config.S5P_LOCF_MAX_DAYS)
        age_norm = age / float(config.S5P_LOCF_MAX_DAYS)
        valid_map = np.full((h, w), valid, dtype=np.float32)
        age_map = np.full((h, w), age_norm, dtype=np.float32)
        frame = np.stack([last_aai, last_co, valid_map, age_map], axis=0)
        frame = np.where(land_mask[None], frame, 0.0)
        stack[ti] = frame
    return stack


# ---------------------------------------------------------------------------
# S2 snapshot + lag
# ---------------------------------------------------------------------------
_s2_index: pd.DataFrame | None = None


def _get_s2_index() -> pd.DataFrame:
    global _s2_index
    if _s2_index is None:
        df = index_local_s2()
        # attach window_end by sampling files (cached later)
        ends = []
        for _, row in df.iterrows():
            try:
                meta = pd.read_parquet(row["path"], columns=["window_end"])
                ends.append(pd.Timestamp(meta["window_end"].iloc[0]).date())
            except Exception:
                # approximate from year/month/window for 5-day cadence
                ends.append(date(int(row["year"]), int(row["month"]), 1))
        df = df.copy()
        df["window_end"] = ends
        _s2_index = df
    return _s2_index


def build_s2_static(label_day, reference=None, land_mask=None) -> np.ndarray:
    """(4, H, W): NDVI, NBR, NDWI, lag_days/S2_MAX_LAG."""
    reference = reference if reference is not None else reference_da()
    land_mask = land_mask if land_mask is not None else build_land_mask(reference)
    h, w = int(reference.sizes["y"]), int(reference.sizes["x"])
    feat_end = feature_end_for_label(label_day)

    idx = _get_s2_index()
    best = None
    best_end = None
    if not idx.empty:
        cand = idx[idx["window_end"].apply(lambda d: d <= feat_end)]
        if not cand.empty:
            # prefer latest window_end
            cand = cand.sort_values("window_end")
            row = cand.iloc[-1]
            lag = (feat_end - row["window_end"]).days
            if lag <= config.S2_MAX_LAG_DAYS:
                best = Path(row["path"])
                best_end = row["window_end"]

    out = np.zeros((config.S2_CHANNELS, h, w), dtype=np.float32)
    if best is None:
        return out

    df = pd.read_parquet(best)
    for i, col in enumerate(config.S2_COLUMNS):
        if col not in df.columns:
            continue
        arr = rasterize_points_to_grid(
            df["latitude"].to_numpy(),
            df["longitude"].to_numpy(),
            df[col].to_numpy(dtype=np.float32),
            reference,
        )
        out[i] = np.nan_to_num(arr, nan=0.0)
    lag = (feat_end - best_end).days if best_end else config.S2_MAX_LAG_DAYS
    out[3] = np.full((h, w), lag / float(config.S2_MAX_LAG_DAYS), dtype=np.float32)
    out = np.where(land_mask[None], out, 0.0)
    return out


# ---------------------------------------------------------------------------
# DEM static
# ---------------------------------------------------------------------------
_dem_cache: dict[str, np.ndarray] | None = None


def build_dem_static(reference=None, land_mask=None) -> np.ndarray:
    global _dem_cache
    reference = reference if reference is not None else reference_da()
    land_mask = land_mask if land_mask is not None else build_land_mask(reference)
    if _dem_cache is None:
        _dem_cache = load_dem_on_grid(reference)
    elev = np.nan_to_num(_dem_cache["elevation"], nan=0.0)
    slope = np.nan_to_num(_dem_cache["slope"], nan=0.0)
    aspect = np.nan_to_num(_dem_cache["aspect"], nan=0.0)
    out = np.stack([elev, slope, aspect], axis=0).astype(np.float32)
    return np.where(land_mask[None], out, 0.0)


@dataclass
class DayFeatures:
    era5: np.ndarray  # T,7,H,W
    s5p: np.ndarray   # T,4,H,W
    s2: np.ndarray    # 4,H,W
    dem: np.ndarray   # 3,H,W
    era5_valid: np.ndarray  # T


def build_features_for_label_day(label_day) -> DayFeatures | None:
    reference = reference_da()
    land = build_land_mask(reference)
    era5, era5_valid = build_era5_stack(label_day, reference, land)
    if era5_valid.sum() < config.HISTORY_DAYS - config.ERA5_MAX_MISSING_DAYS:
        logger.warning("Too many ERA5 gaps for %s", label_day)
        return None
    s5p = build_s5p_stack(label_day, reference, land)
    # count bad S5P days: valid channel all ~0
    bad = 0
    for ti in range(config.HISTORY_DAYS):
        if s5p[ti, 2].mean() < 0.1:
            bad += 1
    if bad > config.S5P_MAX_BAD_DAYS:
        logger.debug("S5P sparse for %s (%d bad days) — continuing with masks", label_day, bad)
    s2 = build_s2_static(label_day, reference, land)
    dem = build_dem_static(reference, land)
    return DayFeatures(era5=era5, s5p=s5p, s2=s2, dem=dem, era5_valid=era5_valid)
