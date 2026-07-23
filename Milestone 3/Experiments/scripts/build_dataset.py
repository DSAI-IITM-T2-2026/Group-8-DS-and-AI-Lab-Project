#!/usr/bin/env python3
"""
Build fire-centered candidate dataset from GCS (CSV S2/S5P + FIRMS/ERA5/DEM).

Usage:
  python scripts/build_dataset.py --year 2025
  python scripts/build_dataset.py --year 2025 --start 2025-07-01 --end 2025-08-15 --max-days 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GS_NO_SIGN_REQUEST"] = "YES"

from src import config
from src.data.aoi import load_locked_aoi
from src.data.caches import (
    ERA5DailyCache,
    FIRMSLabelCache,
    S5PCSVDailyCache,
    build_s2_csv_forward_fill_cache,
)
from src.data.dataset import normalize_channels, save_splits
from src.data.loaders import get_firms_reference, list_firms_dates, load_and_regrid_dem_features
from src.data.sampling import build_fire_centered_dataset, temporal_split_by_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=config.CANDIDATE_YEAR)
    parser.add_argument("--max-days", type=int, default=None, help="Limit number of target dates")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--no-disk-cache",
        action="store_true",
        help="Stream from GCS only; do not read/write data/cache/ (except existing DEM files)",
    )
    args = parser.parse_args()
    if args.no_disk_cache:
        config.USE_DISK_CACHE = False
        print("Disk cache OFF — streaming from GCS (smoke-style)")

    aoi = load_locked_aoi()
    print(f"Using locked AOI: {aoi}")

    # Prefer a mid-year FIRMS day in the candidate year for the reference grid
    ref_date = f"{args.year}-08-15"
    print(f"Loading FIRMS reference ({ref_date}, clipped)...")
    firms_ref = get_firms_reference(ref_date, bounds=aoi)
    print(f"  reference grid: {firms_ref.sizes}")

    print("Regridding DEM (local download + retry)...")
    t0 = time.time()
    dem = load_and_regrid_dem_features(firms_ref)
    print(f"  DEM done in {time.time()-t0:.1f}s")

    start = args.start or f"{args.year}-01-01"
    # S2 Dec 2025 may still be missing — default end to Nov 30 for 2025
    default_end = f"{args.year}-11-30" if args.year == 2025 else f"{args.year}-12-31"
    end = args.end or default_end

    months = sorted(
        set(
            pd.date_range(start, end, freq="D").month.tolist()
            + pd.date_range(
                (pd.Timestamp(start) - pd.Timedelta(days=config.HISTORY_DAYS)).strftime("%Y-%m-%d"),
                end,
                freq="D",
            ).month.tolist()
        )
    )

    print(f"Building S2 CSV forward-fill cache for months {months}...")
    t0 = time.time()
    s2_cache = build_s2_csv_forward_fill_cache(firms_ref, year=args.year, months=months)
    print(f"  S2 CSV cache done in {time.time()-t0:.1f}s, scenes={s2_cache._sorted_dates}")

    era5_cache = ERA5DailyCache(firms_ref)
    s5p_cache = S5PCSVDailyCache(firms_ref)
    firms_label_cache = FIRMSLabelCache(
        confidence_threshold=config.FIRE_CONFIDENCE_THRESHOLD, bounds=aoi
    )

    all_firms = list_firms_dates()
    dates = [d for d in all_firms if start <= d <= end]
    hist_start = (pd.Timestamp(start) - pd.Timedelta(days=config.HISTORY_DAYS)).strftime("%Y-%m-%d")
    dates_with_hist = [d for d in all_firms if hist_start <= d <= end]
    if args.max_days is not None:
        targets = [d for d in dates_with_hist if d >= start][: args.max_days]
        if targets:
            hist_start2 = (
                pd.Timestamp(targets[0]) - pd.Timedelta(days=config.HISTORY_DAYS)
            ).strftime("%Y-%m-%d")
            dates_with_hist = [d for d in dates_with_hist if hist_start2 <= d <= targets[-1]]

    print(f"Building samples over {len(dates_with_hist)} calendar dates ({start} → {end})...")
    t0 = time.time()
    X, y, meta = build_fire_centered_dataset(
        dates_with_hist,
        s2_cache=s2_cache,
        era5_cache=era5_cache,
        s5p_cache=s5p_cache,
        firms_label_cache=firms_label_cache,
        dem_features_on_firms_grid=dem,
    )
    print(f"Built X={X.shape}, y={y.shape} in {time.time()-t0:.1f}s")
    print(f"Fire-positive samples: {sum(m['has_fire'] for m in meta)} / {len(meta)}")

    splits_raw = temporal_split_by_date(X, y, meta)
    Xtr, Xva, Xte, stats = normalize_channels(
        splits_raw["train"]["X"], splits_raw["val"]["X"], splits_raw["test"]["X"]
    )
    splits = {
        "train": {**splits_raw["train"], "X": Xtr},
        "val": {**splits_raw["val"], "X": Xva},
        "test": {**splits_raw["test"], "X": Xte},
    }
    save_splits(
        splits,
        stats,
        aoi,
        extra_meta={
            "year": args.year,
            "max_days": args.max_days,
            "start": start,
            "end": end,
            "s2_source": "csv_features_v3",
            "s5p_source": "csv_features_daily",
        },
    )
    print("Dataset build complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
