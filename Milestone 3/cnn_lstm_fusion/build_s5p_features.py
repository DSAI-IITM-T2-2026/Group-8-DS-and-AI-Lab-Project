#!/usr/bin/env python3
"""Build cell-level Sentinel-5P aerosol features (optimized: group-by-month, tile reuse)."""

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
from src.patches import TileHandleCache, sample_point_from_dataset
from src.s5p_index import (
    available_year_months,
    list_s5p_tiles,
    pick_tile,
    resolve_month_for_date,
    resolve_read_path,
)
from src.tile_download import ensure_local_tile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_s5p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--download-tiles",
        action="store_true",
        help="Download each monthly S5P mosaic locally before sampling",
    )
    p.add_argument("--keep-tiles", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-sample even if s5p_aerosol already present",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    s5p_cfg = cfg.get("sources", {}).get("sentinel5p", {})
    if not s5p_cfg.get("enabled", False) and not args.force:
        logger.warning(
            "sources.sentinel5p.enabled is false. Proceeding anyway (explicit build script)."
        )

    out_dir = Path(cfg["paths"]["output_dir"])
    manifest_path = Path(cfg["paths"]["manifest"])
    tiles_dir = Path(cfg["paths"].get("s5p_tiles_dir", out_dir / "s5p_tiles"))
    tiles_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest {manifest_path}. Run build_dataset.py first.")

    manifest = pd.read_parquet(manifest_path)
    manifest["feature_end_date"] = pd.to_datetime(manifest["feature_end_date"]).dt.normalize()

    if "s5p_aerosol" in manifest.columns and not args.force:
        already = manifest["s5p_aerosol"].notna().sum()
        if already == len(manifest):
            logger.info("All %d samples already have s5p_aerosol — nothing to do", already)
            return 0

    gcs_prefix = s5p_cfg.get(
        "gcs_prefix", "gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p"
    )
    logger.info("Indexing S5P tiles under %s", gcs_prefix)
    tile_index = list_s5p_tiles(gcs_prefix)
    tile_index.to_parquet(out_dir / "s5p_tile_index.parquet", index=False)
    avail = available_year_months(tile_index)

    # Assign month + tile metadata
    assign_rows = []
    n_fail = 0
    for _, row in manifest.iterrows():
        ym = resolve_month_for_date(pd.Timestamp(row["feature_end_date"]), avail)
        if ym is None:
            n_fail += 1
            continue
        year, month = ym
        tile = pick_tile(tile_index, year, month)
        if tile is None:
            n_fail += 1
            continue
        rec = row.to_dict()
        rec["s5p_year"] = year
        rec["s5p_month"] = month
        rec["s5p_filename"] = tile["filename"]
        rec["s5p_gs_uri"] = tile["gs_uri"]
        rec["s5p_vsigs"] = tile["vsigs_path"]
        assign_rows.append(rec)

    if not assign_rows:
        logger.error("No samples could be assigned to S5P months")
        return 1

    assigned = pd.DataFrame(assign_rows)
    logger.info(
        "Assigned %d samples to %d unique months (assign_fail=%d)",
        len(assigned),
        assigned["s5p_filename"].nunique(),
        n_fail,
    )

    values: dict[str, float] = {}
    sample_fail = 0
    skipped = 0
    grouped = assigned.groupby("s5p_filename", sort=True)

    with TileHandleCache() as cache:
        for filename, group in tqdm(grouped, total=grouped.ngroups, desc="S5P months"):
            gs_uri = group["s5p_gs_uri"].iloc[0]
            vsigs = group["s5p_vsigs"].iloc[0]
            local_path = tiles_dir / filename

            if args.download_tiles:
                try:
                    read_path = str(ensure_local_tile(gs_uri, local_path))
                except Exception as exc:
                    logger.warning(
                        "S5P download failed %s: %s — falling back to /vsigs/", filename, exc
                    )
                    read_path = vsigs
            else:
                read_path = resolve_read_path(
                    {"filename": filename, "vsigs_path": vsigs},
                    tiles_dir if tiles_dir.exists() else None,
                )

            try:
                src = cache.get(read_path)
            except Exception as exc:
                logger.warning("Cannot open S5P %s: %s", read_path, exc)
                sample_fail += len(group)
                continue

            for _, row in group.iterrows():
                sid = str(row["sample_id"])
                if (
                    not args.force
                    and "s5p_aerosol" in row
                    and pd.notna(row.get("s5p_aerosol"))
                ):
                    values[sid] = float(row["s5p_aerosol"])
                    skipped += 1
                    continue
                val = sample_point_from_dataset(
                    src, float(row["longitude"]), float(row["latitude"]), band=1
                )
                if val is None:
                    sample_fail += 1
                    continue
                values[sid] = float(val)

            if args.download_tiles and not args.keep_tiles and local_path.exists():
                cache.close()
                try:
                    local_path.unlink()
                except OSError:
                    pass

    # Merge back onto full manifest; fill missing with train median later in train.py
    manifest = manifest.copy()
    manifest["s5p_aerosol"] = manifest["sample_id"].map(values).astype("float32")
    n_ok = int(manifest["s5p_aerosol"].notna().sum())
    if n_ok == 0:
        logger.error("No S5P values sampled")
        return 1

    # Fill remaining NaNs with overall median so training can proceed
    med = float(np.nanmedian(manifest["s5p_aerosol"].to_numpy()))
    n_filled = int(manifest["s5p_aerosol"].isna().sum())
    if n_filled:
        manifest["s5p_aerosol"] = manifest["s5p_aerosol"].fillna(med)
        logger.warning("Filled %d missing S5P values with median=%.4f", n_filled, med)

    drop_cols = [c for c in ("s5p_gs_uri", "s5p_vsigs") if c in manifest.columns]
    # Keep s5p_year/month/filename for debugging; drop bulky URIs if present from merge
    for c in drop_cols:
        if c in assigned.columns:
            pass
    manifest.to_parquet(manifest_path, index=False)

    meta = {
        "n_with_s5p": n_ok,
        "n_assign_failed": n_fail,
        "n_sample_failed": sample_fail,
        "n_skipped_existing": skipped,
        "n_filled_median": n_filled,
        "median_fill": med,
        "download_tiles": bool(args.download_tiles),
    }
    with (out_dir / "s5p_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(
        "Done. s5p ok=%d  assign_fail=%d  sample_fail=%d  skipped=%d",
        n_ok,
        n_fail,
        sample_fail,
        skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
