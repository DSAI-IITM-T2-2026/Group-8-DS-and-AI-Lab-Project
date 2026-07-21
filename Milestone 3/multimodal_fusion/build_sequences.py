#!/usr/bin/env python3
"""Build ERA5+DEM [7, F] sequences for each manifest sample."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.sequences import build_sequences, load_dem_features, load_era5_daily_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_sequences")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    manifest_path = Path(cfg["paths"]["manifest"])
    sequences_dir = Path(cfg["paths"]["sequences_dir"])
    mvp_dir = Path(cfg["paths"]["mvp_output_dir"])
    history_days = int(cfg["task"]["history_days"])

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest {manifest_path}. Run build_dataset.py first.")

    manifest = pd.read_parquet(manifest_path)
    manifest["feature_end_date"] = pd.to_datetime(manifest["feature_end_date"]).dt.normalize()
    manifest["label_date"] = pd.to_datetime(manifest["label_date"]).dt.normalize()

    start = manifest["feature_end_date"].min() - pd.Timedelta(days=history_days)
    end = manifest["feature_end_date"].max()
    cache_dir = mvp_dir / "cache" / "era5_daily"
    logger.info("Loading ERA5 daily cache from %s", cache_dir)
    era5 = load_era5_daily_cache(cache_dir, start, end)
    dem = load_dem_features(mvp_dir)

    updated = build_sequences(
        manifest,
        era5,
        dem,
        sequences_dir,
        history_days=history_days,
        skip_existing=not args.no_skip_existing,
    )
    if updated.empty:
        logger.error("No sequences built")
        return 1

    updated.to_parquet(manifest_path, index=False)
    meta = {
        "n_with_sequences": len(updated),
        "history_days": history_days,
        "seq_feature_dim": updated.attrs.get("seq_feature_dim"),
        "era5_cols": updated.attrs.get("era5_cols"),
        "dem_cols": updated.attrs.get("dem_cols"),
        "split_counts": updated["split"].value_counts().to_dict(),
    }
    with (out_dir / "sequence_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Updated manifest with sequence_path → %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
