"""Source caches — CSV S2/S5P (primary) + legacy GeoTIFF helpers + ERA5/FIRMS.

Durable files under data/cache/ skip GCS on rebuild (Milestone 3 teammate pattern).
"""
from __future__ import annotations

import gc
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from src import config
from src.data import gcs as gcs_mod
from src.data.csv_raster import find_s5p_csv_for_date, rasterize_feature_columns
from src.data.disk_cache import load_array, load_arrays, save_array, save_arrays
from src.data.feature_csv import list_feature_csv_paths, peek_window_dates
from src.data.loaders import (
    aggregate_era5_to_daily,
    compute_sentinel2_indices,
    firms_to_binary_label,
    load_firms_raster,
    load_sentinel2_tile,
    load_sentinel5p_file,
    open_era5_file,
    parse_s2_year_month,
    s2_scene_date,
)


class ForwardFillSourceCache:
    """Caches regridded S2 monthly mosaics; serves most recent scene ≤ target date."""

    def __init__(self, reference_grid_da):
        self.reference_grid_da = reference_grid_da
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self._sorted_dates: list[str] = []

    def add_scene_arrays(self, scene_date_str: str, feature_dict: dict[str, np.ndarray]):
        self._cache[scene_date_str] = feature_dict
        self._sorted_dates = sorted(self._cache.keys())

    def get_most_recent(self, target_date_str: str) -> Optional[dict[str, np.ndarray]]:
        candidates = [d for d in self._sorted_dates if d <= target_date_str]
        if not candidates:
            return None
        return self._cache[candidates[-1]]


def _regrid_array(arr_2d, source_da_like, reference_da) -> np.ndarray:
    da_wrapped = xr.DataArray(
        arr_2d,
        dims=source_da_like.dims[-2:],
        coords={d: source_da_like.coords[d] for d in source_da_like.dims[-2:]},
    )
    da_wrapped = da_wrapped.rio.write_crs(source_da_like.rio.crs)
    da_wrapped = da_wrapped.rio.write_transform(source_da_like.rio.transform())
    return da_wrapped.rio.reproject_match(
        reference_da, resampling=Resampling.bilinear
    ).values


def build_s2_monthly_cache(
    reference_grid_da,
    year: int = 2024,
    tile_paths: Optional[list[str]] = None,
    months: Optional[list[int]] = None,
    max_tiles_per_month: Optional[int] = None,
) -> ForwardFillSourceCache:
    """
    Group S2 tiles by (year, month), mosaic indices onto the FIRMS grid
    (nanmean across overlapping tiles), store under last day of month.
    """
    from src.data.loaders import list_sentinel2_tiles

    if tile_paths is None:
        tile_paths = list_sentinel2_tiles()

    by_month: dict[tuple[int, int], list[str]] = defaultdict(list)
    for path in tile_paths:
        ym = parse_s2_year_month(path)
        if ym is None:
            continue
        y, m = ym
        if y != year:
            continue
        if months is not None and m not in months:
            continue
        by_month[(y, m)].append(path)

    cache = ForwardFillSourceCache(reference_grid_da)
    H = reference_grid_da.sizes["y"]
    W = reference_grid_da.sizes["x"]

    for (y, m) in sorted(by_month.keys()):
        paths = sorted(by_month[(y, m)])
        if max_tiles_per_month is not None:
            paths = paths[:max_tiles_per_month]
        sums = {k: np.zeros((H, W), dtype=np.float64) for k in ("NDVI_S2", "NBR_S2", "NDWI_S2")}
        counts = {k: np.zeros((H, W), dtype=np.float64) for k in sums}
        print(f"  S2 {y}-{m:02d}: mosaicking {len(paths)} tile(s)...")
        for path in paths:
            try:
                raw = load_sentinel2_tile(path)
                indices = compute_sentinel2_indices(raw)
                for name, arr in indices.items():
                    regridded = _regrid_array(arr, raw, reference_grid_da)
                    valid = np.isfinite(regridded)
                    sums[name][valid] += regridded[valid]
                    counts[name][valid] += 1
                del raw, indices
                gc.collect()
            except Exception as exc:
                print(f"    skip {path.split('/')[-1]}: {exc}")
        mosaic = {}
        for name in sums:
            with np.errstate(invalid="ignore", divide="ignore"):
                avg = sums[name] / np.maximum(counts[name], 1)
            avg[counts[name] == 0] = np.nan
            mosaic[name] = avg.astype(np.float32)
        cache.add_scene_arrays(s2_scene_date(y, m), mosaic)
        print(f"    stored as scene date {s2_scene_date(y, m)}")
    return cache


def regrid_array_to_grid(arr_2d, source_lat, source_lon, reference_da) -> np.ndarray:
    da = xr.DataArray(
        arr_2d,
        dims=("latitude", "longitude"),
        coords={"latitude": source_lat, "longitude": source_lon},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    return da.rio.reproject_match(reference_da, resampling=Resampling.bilinear).values


def _era5_day_path(year: int, month: int, day: int):
    return config.CACHE_ERA5_DIR / f"{year}" / f"{month:02d}" / f"{day:02d}.npz"


def _era5_month_marker(year: int, month: int):
    return config.CACHE_ERA5_DIR / f"{year}" / f"{month:02d}" / "_complete.json"


class ERA5DailyCache:
    """Loads one month at a time; persists daily grids under data/cache/era5/."""

    def __init__(self, reference_grid_da):
        self.reference_grid_da = reference_grid_da
        self._month_cache: dict[tuple[int, int], dict] = {}

    def _try_load_month_from_disk(self, year: int, month: int) -> Optional[dict]:
        import json

        if not config.USE_DISK_CACHE:
            return None
        marker = _era5_month_marker(year, month)
        if not marker.exists():
            return None
        try:
            meta = json.loads(marker.read_text())
            n_days = int(meta["n_days"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        daily_by_day = {}
        for day in range(1, n_days + 1):
            arrays = load_arrays(_era5_day_path(year, month, day))
            if arrays is None:
                return None
            daily_by_day[day] = arrays
        print(f"  ERA5 {year}-{month:02d}: loaded from disk cache ({n_days} days)")
        return daily_by_day

    def _load_month(self, year: int, month: int):
        import json

        from_disk = self._try_load_month_from_disk(year, month)
        if from_disk is not None:
            self._month_cache[(year, month)] = from_disk
            self._evict_old_months(year, month)
            return

        path = f"{config.GCS_BUCKET}/{config.PREFIX_ERA5}/{year}/era5_{year}_{month:02d}.nc"
        ds = open_era5_file(path)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        n_days = int(ds.sizes["time"] // 24)
        daily_by_day = {}
        for day_idx in range(n_days):
            day_slice = ds.isel(time=slice(day_idx * 24, (day_idx + 1) * 24))
            daily_vars = aggregate_era5_to_daily(day_slice)
            day_num = day_idx + 1
            arrays = {
                name: regrid_array_to_grid(arr, lat, lon, self.reference_grid_da)
                for name, arr in daily_vars.items()
            }
            daily_by_day[day_num] = arrays
            if config.USE_DISK_CACHE:
                save_arrays(_era5_day_path(year, month, day_num), arrays)
        if config.USE_DISK_CACHE:
            marker = _era5_month_marker(year, month)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"n_days": n_days, "year": year, "month": month}))
        self._month_cache[(year, month)] = daily_by_day
        del ds
        gc.collect()
        mode = "GCS + cached" if config.USE_DISK_CACHE else "GCS stream (no disk cache)"
        print(
            f"  ERA5 {year}-{month:02d}: loaded from {mode} "
            f"({len(self._month_cache)} months in RAM)"
        )
        self._evict_old_months(year, month)

    def _evict_old_months(self, current_year: int, current_month: int):
        current_ordinal = current_year * 12 + current_month
        to_evict = [
            k
            for k in self._month_cache
            if (current_ordinal - (k[0] * 12 + k[1])) > 1
        ]
        for k in to_evict:
            del self._month_cache[k]
        if to_evict:
            gc.collect()
            print(f"    evicted {len(to_evict)} old month(s), {len(self._month_cache)} remain")

    def get_daily(self, date_str: str) -> dict[str, np.ndarray]:
        year, month, day = (int(x) for x in date_str.split("-"))
        if (year, month) not in self._month_cache:
            self._load_month(year, month)
        return self._month_cache[(year, month)][day]


class S5PMonthlyCache:
    """Legacy monthly GeoTIFF S5P (kept for back-compat). Prefer S5PCSVDailyCache."""

    def __init__(self, reference_grid_da):
        self.reference_grid_da = reference_grid_da
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def get_daily(self, date_str: str) -> dict[str, np.ndarray]:
        year, month, _ = date_str.split("-")
        key = (year, month)
        if key not in self._cache:
            path = f"gs://{config.GCS_BUCKET}/{config.PREFIX_S5P}/s5p_{year}_{month}.tif"
            da = load_sentinel5p_file(path)
            aerosol = da.sel(band="aerosol_index")
            regridded = aerosol.rio.reproject_match(
                self.reference_grid_da, resampling=Resampling.bilinear
            )
            self._cache[key] = regridded.values
        return {"aerosol_index": self._cache[key]}


def _s2_scene_cache_path(year: int, scene_date: str):
    return config.CACHE_S2_DIR / str(year) / f"{scene_date}.npz"


def build_s2_csv_forward_fill_cache(
    reference_grid_da,
    year: int,
    months: Optional[list[int]] = None,
) -> ForwardFillSourceCache:
    """
    Load S2 Hive features.csv windows, rasterize NDVI/NBR/NDWI means onto FIRMS,
    store under window_end for forward-fill via get_most_recent(date).
    Disk-caches each scene under data/cache/s2_csv/{year}/{end}.npz.
    """
    rows = list_feature_csv_paths(config.S2_BUCKET, config.S2_PREFIX, year=year)
    if months is not None:
        months_set = set(months)
        rows = [r for r in rows if r.get("month") in months_set]

    cache = ForwardFillSourceCache(reference_grid_da)
    timeout_s = float(os.environ.get("M3_S2_WINDOW_TIMEOUT_S", "300"))
    print(f"  S2 CSV: {len(rows)} window(s) for year={year} (timeout={timeout_s:.0f}s/window)", flush=True)

    def _load_one_window(path: str):
        peek = peek_window_dates(path)
        end = peek.get("window_end") or peek.get("window_start")
        if not end:
            raise ValueError("missing window_end")
        scene_date = str(end)[:10]
        disk_path = _s2_scene_cache_path(year, scene_date)
        arrays = load_arrays(disk_path) if config.USE_DISK_CACHE else None
        if arrays is not None and all(k in arrays for k in ("NDVI_S2", "NBR_S2", "NDWI_S2")):
            return scene_date, arrays, "disk-cache hit"
        grids = rasterize_feature_columns(
            path,
            columns=["NDVI_mean", "NBR_mean", "NDWI_mean"],
            reference_grid_da=reference_grid_da,
        )
        feature_dict = {
            "NDVI_S2": grids["NDVI_mean"],
            "NBR_S2": grids["NBR_mean"],
            "NDWI_S2": grids["NDWI_mean"],
        }
        if config.USE_DISK_CACHE:
            save_arrays(disk_path, feature_dict)
        tag = "ok (cached)" if config.USE_DISK_CACHE else "ok (GCS stream)"
        return scene_date, feature_dict, tag

    for i, row in enumerate(sorted(rows, key=lambda r: (r["month"] or 0, r["window"] or 0))):
        path = row["path"]
        print(
            f"    [{i+1}/{len(rows)}] window={row.get('window')} fetching...",
            flush=True,
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_load_one_window, path)
                scene_date, feature_dict, tag = fut.result(timeout=timeout_s)
            cache.add_scene_arrays(scene_date, feature_dict)
            print(
                f"    [{i+1}/{len(rows)}] window={row.get('window')} "
                f"end={scene_date} {tag}",
                flush=True,
            )
        except FuturesTimeout:
            print(
                f"    [{i+1}/{len(rows)}] TIMEOUT after {timeout_s:.0f}s — skip {path}",
                flush=True,
            )
            gcs_mod._fs = None  # force fresh GCS client next window
        except Exception as exc:
            print(f"    skip {path}: {exc}", flush=True)
            gcs_mod._fs = None
        gc.collect()
    return cache


class S5PCSVDailyCache:
    """Daily S5P aerosol from features.csv; disk-cached under data/cache/s5p_csv/."""

    def __init__(self, reference_grid_da):
        self.reference_grid_da = reference_grid_da
        self._by_date: dict[str, np.ndarray] = {}
        self._last: Optional[np.ndarray] = None
        self._nan = np.full(
            (int(reference_grid_da.sizes["y"]), int(reference_grid_da.sizes["x"])),
            np.nan,
            dtype=np.float32,
        )

    def _disk_path(self, date_str: str):
        return config.CACHE_S5P_DIR / f"{date_str}.npy"

    def get_daily(self, date_str: str) -> dict[str, np.ndarray]:
        if date_str in self._by_date:
            return {"aerosol_index": self._by_date[date_str]}

        disk = load_array(self._disk_path(date_str)) if config.USE_DISK_CACHE else None
        if disk is not None:
            self._by_date[date_str] = disk
            self._last = disk
            return {"aerosol_index": disk}

        path = find_s5p_csv_for_date(date_str)
        if path is not None:
            try:
                grids = rasterize_feature_columns(
                    path,
                    columns=["s5p_aai_mean"],
                    reference_grid_da=self.reference_grid_da,
                )
                arr = grids["s5p_aai_mean"]
                if config.USE_DISK_CACHE:
                    save_array(self._disk_path(date_str), arr)
                self._by_date[date_str] = arr
                self._last = arr
                return {"aerosol_index": arr}
            except Exception as exc:
                print(f"  S5P CSV fail {date_str}: {exc}")

        if self._last is not None:
            return {"aerosol_index": self._last}
        return {"aerosol_index": self._nan}


class FIRMSLabelCache:
    """FIRMS binary labels; disk-cached under data/cache/firms/{date}.npy."""

    def __init__(self, confidence_threshold: int = 30, bounds: Optional[dict] = None):
        self.confidence_threshold = confidence_threshold
        self.bounds = bounds
        self._cache: dict[str, np.ndarray] = {}

    def _disk_path(self, date_str: str):
        return config.CACHE_FIRMS_DIR / f"{date_str}.npy"

    def get(self, date_str: str) -> np.ndarray:
        if date_str in self._cache:
            return self._cache[date_str]

        disk = load_array(self._disk_path(date_str)) if config.USE_DISK_CACHE else None
        if disk is not None:
            self._cache[date_str] = disk
            return disk

        from src.data.loaders import clip_da_to_bounds

        try:
            da = load_firms_raster(date_str)
            if self.bounds is not None:
                da = clip_da_to_bounds(da, self.bounds)
            label = firms_to_binary_label(da, self.confidence_threshold)
        except Exception as exc:
            print(f"  FIRMS label fail {date_str}: {exc}", flush=True)
            raise
        if config.USE_DISK_CACHE:
            save_array(self._disk_path(date_str), label)
        self._cache[date_str] = label
        return label
