#!/usr/bin/env python3
"""Extract Sentinel-5P 2×64×64 patches from monthly mosaics (optimized)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.patches import TileHandleCache, extract_patch_from_dataset, save_patch
from src.s5p_index import (
    available_year_months,
    list_s5p_mosaics,
    pick_mosaic,
    resolve_month_for_date,
    resolve_read_path,
)
from src.tile_download import ensure_local_tile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_s5p_patches")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--download-tiles", action="store_true")
    p.add_argument("--keep-tiles", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if not cfg.get("sources", {}).get("sentinel5p_patches", True):
        logger.info("sentinel5p_patches disabled — skip")
        return 0

    manifest_path = Path(cfg["paths"]["manifest"])
    patches_dir = Path(cfg["paths"]["s5p_patches_dir"])
    tiles_dir = Path(cfg["paths"]["s5p_tiles_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    patches_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest {manifest_path}. Run build_dataset.py first.")

    manifest = pd.read_parquet(manifest_path)
    manifest["feature_end_date"] = pd.to_datetime(manifest["feature_end_date"]).dt.normalize()

    pcfg = cfg["patch_s5p"]
    tile_index = list_s5p_mosaics(pcfg["gcs_prefix"])
    tile_index.to_parquet(out_dir / "s5p_mosaic_index.parquet", index=False)
    avail = available_year_months(tile_index)
    size = int(pcfg["size"])
    bands = int(pcfg["bands"])

    assign_rows = []
    n_fail = 0
    for _, row in manifest.iterrows():
        ym = resolve_month_for_date(pd.Timestamp(row["feature_end_date"]), avail)
        if ym is None:
            n_fail += 1
            continue
        year, month = ym
        tile = pick_mosaic(tile_index, year, month)
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
        logger.error("No samples assigned to S5P mosaics")
        return 1

    assigned = pd.DataFrame(assign_rows)
    logger.info(
        "Assigned %d samples to %d mosaics (fail=%d)",
        len(assigned),
        assigned["s5p_filename"].nunique(),
        n_fail,
    )

    rows_ok = []
    extract_fail = 0
    skipped = 0
    grouped = assigned.groupby("s5p_filename", sort=True)

    with TileHandleCache() as cache:
        for filename, group in tqdm(grouped, total=grouped.ngroups, desc="S5P mosaics"):
            gs_uri = group["s5p_gs_uri"].iloc[0]
            vsigs = group["s5p_vsigs"].iloc[0]
            local_path = tiles_dir / filename

            if args.download_tiles:
                try:
                    read_path = str(ensure_local_tile(gs_uri, local_path))
                except Exception as exc:
                    logger.warning("Download failed %s: %s — vsigs fallback", filename, exc)
                    read_path = vsigs
            else:
                read_path = resolve_read_path(
                    {"filename": filename, "vsigs_path": vsigs},
                    tiles_dir if tiles_dir.exists() else None,
                )

            try:
                src = cache.get(read_path)
            except Exception as exc:
                logger.warning("Cannot open %s: %s", read_path, exc)
                extract_fail += len(group)
                continue

            for _, row in group.iterrows():
                sid = str(row["sample_id"])
                patch_path = patches_dir / f"{sid}.npy"
                if patch_path.exists() and patch_path.stat().st_size > 0:
                    skipped += 1
                    rec = row.to_dict()
                    rec["s5p_patch_path"] = str(patch_path)
                    rows_ok.append(rec)
                    continue
                patch = extract_patch_from_dataset(
                    src,
                    float(row["longitude"]),
                    float(row["latitude"]),
                    size=size,
                    bands=bands,
                )
                if patch is None:
                    extract_fail += 1
                    continue
                save_patch(patch_path, patch)
                rec = row.to_dict()
                rec["s5p_patch_path"] = str(patch_path)
                rows_ok.append(rec)

            if args.download_tiles and not args.keep_tiles and local_path.exists():
                cache.close()
                try:
                    local_path.unlink()
                except OSError:
                    pass

    if not rows_ok:
        logger.error("No S5P patches extracted")
        return 1

    out = pd.DataFrame(rows_ok)
    drop = [c for c in ("s5p_gs_uri", "s5p_vsigs") if c in out.columns]
    out = out.drop(columns=drop)
    out.to_parquet(manifest_path, index=False)

    meta = {
        "n_with_s5p_patches": len(out),
        "n_assign_failed": n_fail,
        "n_extract_failed": extract_fail,
        "n_skipped_existing": skipped,
        "patch_shape": [bands, size, size],
    }
    with (out_dir / "s5p_patch_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Done. s5p patches=%d skipped=%d fail=%d", len(out), skipped, extract_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
