#!/usr/bin/env python3
"""Build balanced CNN dataset: MVP tabular rows + Sentinel-2 patches."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.patches import TileHandleCache, extract_patch_from_dataset, save_patch
from src.s2_index import (
    available_year_months,
    list_s2_tiles,
    pick_tile_for_point,
    resolve_month_for_date,
    resolve_read_path,
)
from src.sample import (
    TABULAR_FEATURES,
    assign_split,
    balanced_sample,
    load_mvp_frames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cnn_build")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on samples for a quick dry-run after sampling",
    )
    p.add_argument(
        "--skip-patches",
        action="store_true",
        help="Only write sampled manifest without extracting patches",
    )
    p.add_argument(
        "--download-tiles",
        action="store_true",
        help="Download each S2 tile locally before extracting (faster windowed reads)",
    )
    p.add_argument(
        "--keep-tiles",
        action="store_true",
        help="Keep downloaded tiles on disk (default: delete after each tile is processed)",
    )
    return p.parse_args()


def _ensure_local_tile(gs_uri: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    logger.info("Downloading %s → %s", gs_uri, dest.name)
    subprocess.check_call(["gsutil", "-q", "cp", gs_uri, str(dest)])
    return dest


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    mvp_dir = Path(cfg["paths"]["mvp_output_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    patches_dir = Path(cfg["paths"]["patches_dir"])
    manifest_path = Path(cfg["paths"]["manifest"])
    tiles_dir = Path(cfg["paths"].get("tiles_dir", out_dir / "s2_tiles"))
    out_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MVP frames from %s", mvp_dir)
    df = load_mvp_frames(mvp_dir)

    months = set(cfg["temporal"]["fire_season_months"])
    df = df.loc[df["label_date"].dt.month.isin(months)].copy()
    start = pd.Timestamp(cfg["temporal"]["start_date"])
    end = pd.Timestamp(cfg["temporal"]["end_date"])
    df = df.loc[(df["label_date"] >= start) & (df["label_date"] <= end)].copy()

    df["split"] = assign_split(df["label_date"], cfg["split"]["train_end"], cfg["split"]["val_end"])
    logger.info("Pre-sample rows=%d pos=%d", len(df), int(df["y_fire"].sum()))
    if len(df) == 0:
        raise SystemExit(
            "No MVP rows left after date/fire-season filters.\n"
            "Finish mvp_era5_dem build for the configured years first."
        )

    samp = cfg["sampling"]
    sampled = balanced_sample(
        df,
        neg_pos_ratio=float(samp["neg_pos_ratio"]),
        random_seed=int(samp["random_seed"]),
        max_train=int(samp["max_train"]),
        max_val=int(samp["max_val"]),
        max_test=int(samp["max_test"]),
        hard_negative=bool(samp.get("hard_negative", True)),
    )
    if args.limit:
        sampled = sampled.sample(n=min(args.limit, len(sampled)), random_state=42).reset_index(
            drop=True
        )
        logger.info("Limited to %d samples for dry-run", len(sampled))

    feat_cols = [c for c in TABULAR_FEATURES if c in sampled.columns]
    missing = set(TABULAR_FEATURES) - set(feat_cols)
    if missing:
        logger.warning("Missing tabular features: %s", sorted(missing))

    logger.info("Indexing Sentinel-2 tiles…")
    tile_index = list_s2_tiles(cfg["patch"]["gcs_prefix"])
    tile_index.to_parquet(out_dir / "s2_tile_index.parquet", index=False)
    avail = available_year_months(tile_index)

    patch_size = int(cfg["patch"]["size"])
    n_bands = int(cfg["patch"]["bands"])

    if args.skip_patches:
        sampled["patch_path"] = ""
        sampled["s2_year"] = pd.NA
        sampled["s2_month"] = pd.NA
        sampled.to_parquet(manifest_path, index=False)
        logger.info("Wrote manifest without patches → %s", manifest_path)
        return 0

    # Assign tile metadata to each sample (no raster I/O)
    assign_rows = []
    n_fail = 0
    for _, row in sampled.iterrows():
        date = pd.Timestamp(row["feature_end_date"])
        ym = resolve_month_for_date(date, avail)
        if ym is None:
            n_fail += 1
            continue
        year, month = ym
        tile = pick_tile_for_point(
            tile_index, year, month, float(row["longitude"]), float(row["latitude"])
        )
        if tile is None:
            n_fail += 1
            continue
        rec = row.to_dict()
        rec["s2_year"] = year
        rec["s2_month"] = month
        rec["s2_filename"] = tile["filename"]
        rec["s2_gs_uri"] = tile["gs_uri"]
        rec["s2_vsigs"] = tile["vsigs_path"]
        assign_rows.append(rec)

    if not assign_rows:
        logger.error("No samples could be assigned to S2 tiles")
        return 1

    assigned = pd.DataFrame(assign_rows)
    logger.info(
        "Assigned %d samples to %d unique tiles (assign_fail=%d)",
        len(assigned),
        assigned["s2_filename"].nunique(),
        n_fail,
    )

    rows_ok = []
    extract_fail = 0
    skipped_existing = 0

    # Group by tile so each remote/local file is opened once
    grouped = assigned.groupby("s2_filename", sort=True)
    with TileHandleCache() as cache:
        for filename, group in tqdm(grouped, total=grouped.ngroups, desc="S2 tiles"):
            gs_uri = group["s2_gs_uri"].iloc[0]
            vsigs = group["s2_vsigs"].iloc[0]
            local_path = tiles_dir / filename

            if args.download_tiles:
                try:
                    read_path = str(_ensure_local_tile(gs_uri, local_path))
                except Exception as exc:
                    logger.warning("Tile download failed %s: %s — falling back to /vsigs/", filename, exc)
                    read_path = vsigs
            else:
                read_path = resolve_read_path(
                    {"filename": filename, "vsigs_path": vsigs},
                    tiles_dir if tiles_dir.exists() else None,
                )

            try:
                src = cache.get(read_path)
            except Exception as exc:
                logger.warning("Cannot open tile %s: %s", read_path, exc)
                extract_fail += len(group)
                continue

            for _, row in group.iterrows():
                sample_id = str(row["sample_id"])
                patch_path = patches_dir / f"{sample_id}.npy"
                if patch_path.exists() and patch_path.stat().st_size > 0:
                    skipped_existing += 1
                    rec = row.to_dict()
                    rec["patch_path"] = str(patch_path)
                    rows_ok.append(rec)
                    continue

                patch = extract_patch_from_dataset(
                    src,
                    float(row["longitude"]),
                    float(row["latitude"]),
                    size=patch_size,
                    bands=n_bands,
                )
                if patch is None:
                    extract_fail += 1
                    continue
                save_patch(patch_path, patch)
                rec = row.to_dict()
                rec["patch_path"] = str(patch_path)
                rows_ok.append(rec)

            # Free disk after each tile unless user wants to keep them
            if args.download_tiles and not args.keep_tiles and local_path.exists():
                cache.close()
                try:
                    local_path.unlink()
                except OSError:
                    pass

    if not rows_ok:
        logger.error("No patches extracted (extract_fail=%d)", extract_fail)
        return 1

    manifest = pd.DataFrame(rows_ok)
    # Drop bulky helper columns from manifest
    drop_cols = [c for c in ("s2_gs_uri", "s2_vsigs") if c in manifest.columns]
    manifest = manifest.drop(columns=drop_cols)
    manifest.to_parquet(manifest_path, index=False)

    meta = {
        "n_requested": len(sampled),
        "n_with_patches": len(manifest),
        "n_assign_failed": n_fail,
        "n_extract_failed": extract_fail,
        "n_skipped_existing": skipped_existing,
        "feature_columns": feat_cols,
        "patch_shape": [n_bands, patch_size, patch_size],
        "split_counts": manifest["split"].value_counts().to_dict(),
        "pos_counts": manifest.groupby("split")["y_fire"].sum().astype(int).to_dict(),
        "split": cfg["split"],
        "temporal": cfg["temporal"],
        "download_tiles": bool(args.download_tiles),
    }
    with (out_dir / "dataset_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(
        "Done. manifest=%s  ok=%d  assign_fail=%d  extract_fail=%d  skipped=%d",
        manifest_path,
        len(manifest),
        n_fail,
        extract_fail,
        skipped_existing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
