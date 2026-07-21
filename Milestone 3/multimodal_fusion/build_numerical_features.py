#!/usr/bin/env python3
"""Cache S2 (5-day) + S5P numerical feature tables and attach to manifest."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.numerical_features import (
    aggregate_features_to_cells,
    attach_forward_filled,
    build_era5_feature_grid_map,
    cache_csv_to_parquet,
    list_feature_files,
)
from src.sample import load_mvp_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_numerical")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--years", nargs="*", type=int, default=None, help="Override years to cache")
    p.add_argument("--limit-windows", type=int, default=None, help="Smoke: max windows per source")
    return p.parse_args()


def _fire_season_windows(index: pd.DataFrame, months: set[int]) -> pd.DataFrame:
    return index.loc[index["month"].isin(months)].copy()


def _ensure_grid_map(cfg: dict, sample_parquet: Path) -> pd.DataFrame:
    grid_map_path = Path(cfg["paths"]["grid_map"])
    if grid_map_path.exists():
        return pd.read_parquet(grid_map_path)

    mvp = load_mvp_frames(Path(cfg["paths"]["mvp_output_dir"]))
    dem = mvp[["cell_id", "latitude", "longitude"]].drop_duplicates("cell_id")
    sample = pd.read_parquet(sample_parquet, columns=["grid_id", "latitude", "longitude"])
    grid_map = build_era5_feature_grid_map(dem, sample)
    grid_map_path.parent.mkdir(parents=True, exist_ok=True)
    grid_map.to_parquet(grid_map_path, index=False)
    logger.info("Wrote grid map %s (%d links)", grid_map_path, len(grid_map))
    return grid_map


def _cache_and_aggregate(
    index: pd.DataFrame,
    bucket: str,
    cache_root: Path,
    grid_map: pd.DataFrame,
    value_cols: list[str],
    limit: int | None,
) -> list[pd.DataFrame]:
    tables = []
    rows = index.itertuples(index=False)
    if limit:
        rows = list(rows)[:limit]
    else:
        rows = list(rows)

    for row in tqdm(rows, desc=f"cache {bucket}"):
        dest = cache_root / f"y{row.year}_m{row.month:02d}_w{row.window:03d}.parquet"
        cache_csv_to_parquet(bucket, row.blob_name, dest, columns=value_cols)
        feat = pd.read_parquet(dest)
        # Build grid map from first file if caller needs sample — already have map
        cell_tab = aggregate_features_to_cells(feat, grid_map, value_cols)
        tables.append(cell_tab)
    return tables


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    manifest_path = Path(cfg["paths"]["manifest"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    months = set(cfg["temporal"]["fire_season_months"])
    years = args.years or list(range(
        pd.Timestamp(cfg["temporal"]["start_date"]).year,
        pd.Timestamp(cfg["temporal"]["end_date"]).year + 1,
    ))

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest {manifest_path}")

    manifest = pd.read_parquet(manifest_path)
    manifest["feature_end_date"] = pd.to_datetime(manifest["feature_end_date"]).dt.normalize()

    s2_cfg = cfg["sources"]["s2_features"]
    s5_cfg = cfg["sources"]["s5p_features"]

    # --- S2 numerical ---
    if s2_cfg.get("enabled", True):
        s2_index = list_feature_files(s2_cfg["gcs_bucket"], s2_cfg["gcs_prefix"], years)
        s2_index = _fire_season_windows(s2_index, months)
        s2_index.to_parquet(out_dir / "s2_feature_window_index.parquet", index=False)
        if s2_index.empty:
            logger.error("No S2 feature windows indexed")
            return 1

        # First window → grid map
        first = s2_index.iloc[0]
        first_pq = cache_dir / "s2" / f"y{first.year}_m{first.month:02d}_w{first.window:03d}.parquet"
        cache_csv_to_parquet(
            s2_cfg["gcs_bucket"], first.blob_name, first_pq, columns=s2_cfg["columns"]
        )
        grid_map = _ensure_grid_map(cfg, first_pq)

        s2_tables = _cache_and_aggregate(
            s2_index,
            s2_cfg["gcs_bucket"],
            cache_dir / "s2",
            grid_map,
            s2_cfg["columns"],
            args.limit_windows,
        )
        prefix = "s2n_"
        manifest = attach_forward_filled(
            manifest, s2_tables, s2_cfg["columns"], prefix=prefix
        )
        # fill missing
        for c in s2_cfg["columns"]:
            col = prefix + c
            if col in manifest.columns:
                med = float(np.nanmedian(manifest[col].to_numpy(dtype=float)))
                manifest[col] = manifest[col].fillna(med)
        logger.info("S2 numerical attached; available=%s", int(manifest[prefix + "available"].sum()))

    # --- S5P numerical ---
    if s5_cfg.get("enabled", True):
        s5_index = list_feature_files(s5_cfg["gcs_bucket"], s5_cfg["gcs_prefix"], years)
        s5_index = _fire_season_windows(s5_index, months)
        s5_index.to_parquet(out_dir / "s5p_feature_window_index.parquet", index=False)
        if s5_index.empty:
            logger.warning("No S5P numerical windows for years=%s — filling zeros", years)
            for c in s5_cfg["columns"]:
                manifest["s5n_" + c] = 0.0
            manifest["s5n_available"] = 0
            manifest["s5n_lag_days"] = np.nan
        else:
            grid_map_path = Path(cfg["paths"]["grid_map"])
            if not grid_map_path.exists():
                raise SystemExit("grid map missing — enable s2_features first or build map")
            grid_map = pd.read_parquet(grid_map_path)
            s5_tables = _cache_and_aggregate(
                s5_index,
                s5_cfg["gcs_bucket"],
                cache_dir / "s5p",
                grid_map,
                s5_cfg["columns"],
                args.limit_windows,
            )
            prefix = "s5n_"
            manifest = attach_forward_filled(
                manifest,
                s5_tables,
                s5_cfg["columns"],
                prefix=prefix,
                max_lag_days=int(s5_cfg.get("forward_fill_max_days", 7)),
            )
            for c in s5_cfg["columns"]:
                col = prefix + c
                if col in manifest.columns:
                    manifest[col] = manifest[col].fillna(0.0)
            logger.info(
                "S5P numerical attached; available=%s",
                int(manifest[prefix + "available"].sum()),
            )

    manifest.to_parquet(manifest_path, index=False)
    meta = {
        "years": years,
        "s2_enabled": bool(s2_cfg.get("enabled", True)),
        "s5p_enabled": bool(s5_cfg.get("enabled", True)),
        "n_rows": len(manifest),
    }
    with (out_dir / "numerical_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Updated manifest → %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
