#!/usr/bin/env python3
"""Build MVP next-day wildfire training tables (ERA5 + DEM + FIRMS labels)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.assemble import assemble_samples, apply_split, feature_columns, write_dataset
from src.cells import load_dem_cells
from src.config import load_config
from src.era5_daily import build_era5_daily_range
from src.firms_labels import build_firms_cell_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--start", default=None, help="Override start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Override end date YYYY-MM-DD")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Short smoke run: 2024-08-01 → 2024-08-31 (good fire-season month)",
    )
    p.add_argument(
        "--fire-season",
        action="store_true",
        help="Restrict to May–November (California fire season)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.smoke:
        start = pd.Timestamp("2024-08-01")
        end = pd.Timestamp("2024-08-31")
        fire_season_months = None
    else:
        start = pd.Timestamp(args.start or cfg["temporal"]["start_date"])
        end = pd.Timestamp(args.end or cfg["temporal"]["end_date"])
        fire_season_months = list(range(5, 12)) if args.fire_season else None

    history = int(cfg["task"]["history_days"])
    lead = int(cfg["task"]["lead_days"])
    # Pull extra ERA5 history so rolling 7d windows are valid near start
    era5_start = start - pd.Timedelta(days=history)
    era5_end = end
    label_start = start
    label_end = end

    out_dir = Path(cfg["paths"]["output_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    raw_era5 = cache_dir / "era5_raw"
    daily_era5 = cache_dir / "era5_daily"
    firms_cache = cache_dir / "firms_cells"

    logger.info(
        "Building MVP dataset  %s → %s  fire_season=%s",
        start.date(),
        end.date(),
        bool(fire_season_months),
    )

    dem = load_dem_cells(Path(cfg["paths"]["dem_cells"]))
    logger.info("DEM cells: %d", len(dem))

    era5 = build_era5_daily_range(
        start=era5_start,
        end=era5_end,
        gcs_prefix=cfg["gcs"]["era5_prefix"],
        raw_cache=raw_era5,
        daily_cache=daily_era5,
    )
    logger.info("ERA5 daily rows: %d", len(era5))

    # Labels must cover feature_end + lead, so extend end by lead_days
    firms = build_firms_cell_labels(
        start=label_start,
        end=label_end + pd.Timedelta(days=lead),
        vsigs_prefix=cfg["gcs"]["firms_vsigs_prefix"],
        confidence_min=float(cfg["task"]["firms_confidence_min"]),
        cache_dir=firms_cache,
        resolution=float(cfg["era5"]["resolution_deg"]),
        months=fire_season_months,
    )
    logger.info("FIRMS fire-cell rows: %d", len(firms))

    samples = assemble_samples(
        dem=dem,
        era5_daily=era5,
        firms_cells=firms,
        history_days=history,
        lead_days=lead,
    )

    # Keep samples whose label_date is inside the requested window
    samples = samples.loc[
        (samples["label_date"] >= start) & (samples["label_date"] <= end)
    ].reset_index(drop=True)

    if fire_season_months is not None:
        samples = samples.loc[
            samples["label_date"].dt.month.isin(fire_season_months)
        ].reset_index(drop=True)
        logger.info("After fire-season filter: %d samples", len(samples))

    feat_cols = feature_columns(samples)
    splits = apply_split(
        samples,
        train_end=cfg["split"]["train_end"],
        val_end=cfg["split"]["val_end"],
    )

    meta = {
        "task": "next_day_fire_occurrence",
        "lead_days": lead,
        "history_days": history,
        "firms_confidence_min": cfg["task"]["firms_confidence_min"],
        "sources": ["ERA5", "CopernicusDEM", "FIRMS"],
        "excluded_sources": ["Landsat8", "Sentinel2", "Sentinel5P"],
        "date_range_requested": [str(start.date()), str(end.date())],
        "n_samples": len(samples),
        "n_positives": int(samples["y_fire"].sum()),
        "feature_columns": feat_cols,
        "split": cfg["split"],
        "unit": "era5_0.25deg_cell_x_day",
        "output": {
            "train": "train.parquet",
            "val": "val.parquet",
            "test": "test.parquet",
            "target": "y_fire",
            "region_col": "region",
            "confidence": "model predict_proba → calibrated %",
        },
    }
    write_dataset(splits, feat_cols, out_dir, meta)
    logger.info("Done. Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
