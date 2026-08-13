"""Multi-bucket S2 / S5P cell caches and Stage B/C joins."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from numerical_nextday.config import shared_cache
from numerical_nextday.data.claims import claim, is_done, mark_done
from numerical_nextday.data.gcs_features import cache_csv_robust, list_feature_files_robust
from numerical_nextday.data.joins import apply_train_median_fill, attach_causal_window_end
from numerical_nextday.data.m3_imports import load_mvp_modules, load_numerical_features

logger = logging.getLogger(__name__)


def _atomic_to_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    df.to_parquet(partial, index=False)
    partial.replace(dest)


def _bucket_for(cfg: dict, modality: str, year: int) -> tuple[str, str]:
    key = "s2_by_year" if modality == "s2" else "s5p_by_year"
    entry = cfg["gcs"][key].get(str(year)) or cfg["gcs"][key].get(year)
    if not entry:
        raise KeyError(f"No GCS mapping for {modality} year={year}")
    return entry["bucket"], entry["prefix"]


def ensure_grid_map(cfg: dict, sample_parquet: Path) -> pd.DataFrame:
    path = Path(cfg["paths"]["grid_map"])
    if path.exists():
        return pd.read_parquet(path)
    nf = load_numerical_features(cfg)
    mvp = load_mvp_modules(cfg)
    dem = mvp["cells"].load_dem_cells(Path(cfg["paths"]["dem_cells"]))
    sample = pd.read_parquet(sample_parquet, columns=["grid_id", "latitude", "longitude"])
    grid_map = nf.build_era5_feature_grid_map(dem, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_map.to_parquet(path, index=False)
    logger.info("Wrote grid map %s (%d)", path, len(grid_map))
    return grid_map


def build_eo_cell_cache(
    cfg: dict,
    modality: str,
    years: list[int],
    months: list[int] | None = None,
    worker: str = "local",
    force: bool = False,
    limit_windows: int | None = None,
) -> list[Path]:
    """Cache Hive feature CSVs → ERA5 cell means per window; write month parquet lists."""
    assert modality in ("s2", "s5p")
    nf = load_numerical_features(cfg)
    cache = shared_cache(cfg)
    months_set = set(months or list(range(1, 13)))
    cols = cfg["s2_features"]["columns"] if modality == "s2" else cfg["s5p_features"]["columns"]
    stage = f"{modality}_cell_cache"
    out_paths: list[Path] = []

    for year in years:
        if modality == "s5p" and year == 2021 and cfg.get("s5p_2021_mode", "placeholder") == "placeholder":
            # Skip download; Stage C will placeholder
            logger.warning("S5P 2021 placeholder mode — skipping cell cache for 2021")
            mark_done(cache, worker=worker, stage=stage, year=year, month=None)
            continue

        bucket, prefix = _bucket_for(cfg, modality, year)
        index = list_feature_files_robust(nf, bucket, prefix, years=[year])
        if index.empty:
            logger.warning("No %s files for year=%s under gs://%s/%s", modality, year, bucket, prefix)
            continue
        index = index.loc[index["month"].isin(months_set)].copy()

        # First window → grid map
        first = index.iloc[0]
        win_cache = cache / f"{modality}_windows" / f"year={year}"
        first_pq = win_cache / f"m{first.month:02d}_w{first.window:03d}.parquet"
        cache_csv_robust(nf, bucket, first.blob_name, first_pq, columns=cols)
        grid_map = ensure_grid_map(cfg, first_pq)

        month_tables: dict[int, list[pd.DataFrame]] = {m: [] for m in months_set}
        rows = list(index.itertuples(index=False))
        if limit_windows:
            rows = rows[:limit_windows]
        logger.info(
            "%s cell cache year=%s: %d window file(s) to load (months=%s)",
            modality,
            year,
            len(rows),
            sorted(months_set),
        )

        for row in tqdm(rows, desc=f"{modality} {year}"):
            dest = win_cache / f"m{row.month:02d}_w{row.window:03d}.parquet"
            cache_csv_robust(nf, bucket, row.blob_name, dest, columns=cols)
            feat = pd.read_parquet(dest)
            cell_tab = nf.aggregate_features_to_cells(feat, grid_map, cols)
            month_tables[int(row.month)].append(cell_tab)

        for m, tabs in month_tables.items():
            if not tabs and not force:
                month_path = cache / f"{modality}_cell" / f"year={year}" / f"month={m:02d}.parquet"
                if month_path.exists():
                    out_paths.append(month_path)
                    continue
            if not tabs:
                continue
            if not claim(
                cache,
                worker=worker,
                stage=stage,
                year=year,
                month=m,
                force=force,
            ):
                month_path = cache / f"{modality}_cell" / f"year={year}" / f"month={m:02d}.parquet"
                if month_path.exists():
                    out_paths.append(month_path)
                continue
            month_path = cache / f"{modality}_cell" / f"year={year}" / f"month={m:02d}.parquet"
            combined = pd.concat(tabs, ignore_index=True)
            _atomic_to_parquet(combined, month_path)
            mark_done(cache, worker=worker, stage=stage, year=year, month=m)
            out_paths.append(month_path)

    return out_paths


def _load_cell_windows(cache: Path, modality: str, years: list[int]) -> list[pd.DataFrame]:
    tables = []
    root = cache / f"{modality}_cell"
    for year in years:
        ydir = root / f"year={year}"
        if not ydir.exists():
            continue
        for p in sorted(ydir.glob("month=*.parquet")):
            tables.append(pd.read_parquet(p))
    return tables


def _lag_prefix(cfg: dict) -> Path:
    """Relative cache root for primary (lag-5) vs lag0 ablation tables."""
    lag = int(cfg["task"].get("era5_lag_days", 5))
    cache = shared_cache(cfg)
    if lag == 0:
        return cache / "lag0"
    return cache


def _infer_feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "cell_id",
        "latitude",
        "longitude",
        "feature_end_date",
        "label_date",
        "eo_asof_date",
        "y_fire",
        "firms_n_pixels",
        "firms_max_confidence",
        "region",
        "era5_lag_days",
        "s5p_2021_status",
        "s2n_lag_days",
        "s5n_lag_days",
    }
    return [c for c in df.columns if c not in skip]


def merge_stage_a(cfg: dict, years: list[int] | None = None) -> Path:
    cache = shared_cache(cfg)
    root = _lag_prefix(cfg)
    years = years or list(range(2019, 2026))
    frames = []
    for y in years:
        p = root / "stage_a" / f"year={y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            logger.warning("Missing Stage A %s", p)
    if not frames:
        raise FileNotFoundError("No Stage A year parquets found")
    all_df = pd.concat(frames, ignore_index=True)
    out = root / "stage_a" / "all.parquet"
    _atomic_to_parquet(all_df, out)
    return out


def write_splits(cfg: dict, samples: pd.DataFrame, stage_name: str) -> dict[str, Path]:
    root = _lag_prefix(cfg)
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    d = pd.to_datetime(samples["label_date"])
    splits = {
        "train": samples.loc[d <= train_end].copy(),
        "val": samples.loc[(d > train_end) & (d <= val_end)].copy(),
        "test": samples.loc[d > val_end].copy(),
    }
    out_dir = root / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in splits.items():
        p = out_dir / f"{name}.parquet"
        _atomic_to_parquet(df, p)
        paths[name] = p
        logger.info("%s/%s n=%d pos=%d", stage_name, name, len(df), int(df["y_fire"].sum()))

    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = _infer_feature_columns(samples)
    (meta_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    years = sorted({int(pd.Timestamp(x).year) for x in samples["label_date"].unique()})
    dataset_meta = {
        "stage": stage_name,
        "era5_lag_days": int(cfg["task"].get("era5_lag_days", 5)),
        "s5p_2021_mode": cfg.get("s5p_2021_mode"),
        "years": years,
        "n_rows": len(samples),
        "n_pos": int(samples["y_fire"].sum()) if "y_fire" in samples.columns else None,
        "split": cfg["split"],
        "feature_columns": feat_cols,
    }
    (meta_dir / "dataset_metadata.json").write_text(json.dumps(dataset_meta, indent=2, default=str))
    return paths


def attach_s2_stage_b(cfg: dict, years: list[int] | None = None) -> Path:
    cache = shared_cache(cfg)
    root = _lag_prefix(cfg)
    years = years or list(range(2019, 2026))
    stage_a = root / "stage_a" / "all.parquet"
    if not stage_a.exists():
        merge_stage_a(cfg, years)
    samples = pd.read_parquet(stage_a)
    if "eo_asof_date" not in samples.columns:
        samples["eo_asof_date"] = pd.to_datetime(samples["label_date"]) - pd.Timedelta(days=1)

    tables = _load_cell_windows(cache, "s2", years)
    cols = cfg["s2_features"]["columns"]
    attached = attach_causal_window_end(
        samples, tables, cols, date_col="eo_asof_date", prefix="s2n_"
    )
    # Split then train-median fill
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    d = pd.to_datetime(attached["label_date"])
    train = attached.loc[d <= train_end]
    val = attached.loc[(d > train_end) & (d <= val_end)]
    test = attached.loc[d > val_end]
    s2_cols = ["s2n_" + c for c in cols]
    train, val, test = apply_train_median_fill(train, val, test, cols=s2_cols)
    all_b = pd.concat([train, val, test], ignore_index=True)
    out = root / "stage_b" / "all.parquet"
    _atomic_to_parquet(all_b, out)
    write_splits(cfg, all_b, "stage_b")
    return out


def attach_s5p_stage_c(cfg: dict, years: list[int] | None = None) -> Path:
    cache = shared_cache(cfg)
    root = _lag_prefix(cfg)
    years = years or list(range(2019, 2026))
    stage_b = root / "stage_b" / "all.parquet"
    if not stage_b.exists():
        raise FileNotFoundError("Run Stage B first")
    samples = pd.read_parquet(stage_b)
    cols = cfg["s5p_features"]["columns"]
    max_lag = int(cfg["s5p_features"].get("forward_fill_max_days", 7))

    ready_years = [y for y in years if not (y == 2021 and cfg.get("s5p_2021_mode") == "placeholder")]
    tables = _load_cell_windows(cache, "s5p", ready_years)
    attached = attach_causal_window_end(
        samples,
        tables,
        cols,
        date_col="eo_asof_date",
        max_lag_days=max_lag,
        prefix="s5n_",
    )
    # Placeholder / zero fill
    for c in cols:
        col = "s5n_" + c
        if col in attached.columns:
            attached[col] = attached[col].fillna(0.0)
        else:
            attached[col] = 0.0
    if "s5n_available" not in attached.columns:
        attached["s5n_available"] = 0
    attached["s5n_available"] = attached["s5n_available"].fillna(0).astype(int)

    # Force 2021 unavailable when placeholder mode
    if cfg.get("s5p_2021_mode") == "placeholder":
        mask = pd.to_datetime(attached["label_date"]).dt.year == 2021
        for c in cols:
            attached.loc[mask, "s5n_" + c] = 0.0
        attached.loc[mask, "s5n_available"] = 0
        attached["s5p_2021_status"] = np.where(mask, "placeholder", "ready")

    out = root / "stage_c" / "all.parquet"
    _atomic_to_parquet(attached, out)
    write_splits(cfg, attached, "stage_c")
    meta = {
        "s5p_2021_mode": cfg.get("s5p_2021_mode"),
        "era5_lag_days": int(cfg["task"].get("era5_lag_days", 5)),
        "n_rows": len(attached),
        "s5n_available_sum": int(attached["s5n_available"].sum()),
    }
    (root / "stage_c" / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
