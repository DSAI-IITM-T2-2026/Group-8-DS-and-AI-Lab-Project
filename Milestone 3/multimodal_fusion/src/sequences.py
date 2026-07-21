from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .sample import DEM_FEATURES, ERA5_DAY_FEATURES

logger = logging.getLogger(__name__)


def load_era5_daily_cache(cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load monthly era5_daily_YYYY_MM.parquet files covering [start, end]."""
    cache_dir = Path(cache_dir)
    frames = []
    # Include one month before start for history warmup reads
    cursor = (start - pd.Timedelta(days=40)).to_period("M").to_timestamp()
    last = end.to_period("M").to_timestamp()
    while cursor <= last:
        path = cache_dir / f"era5_daily_{cursor.year}_{cursor.month:02d}.parquet"
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_parquet(path))
        cursor = (cursor + pd.offsets.MonthBegin(1))
    if not frames:
        raise FileNotFoundError(
            f"No ERA5 daily cache under {cache_dir} for {start.date()}–{end.date()}. "
            "Rebuild mvp_era5_dem first."
        )
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_dem_features(mvp_dir: Path) -> pd.DataFrame:
    """Prefer DEM columns from any MVP split parquet; fallback to dem cells file."""
    for name in ("train", "val", "test"):
        path = mvp_dir / f"{name}.parquet"
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_parquet(path)
            cols = ["cell_id"] + [c for c in DEM_FEATURES if c in df.columns]
            if len(cols) > 1:
                dem = df[cols].drop_duplicates("cell_id")
                return dem

    dem_path = mvp_dir.parent / "data" / "era5_grid_dem_features.parquet"
    if dem_path.exists():
        dem = pd.read_parquet(dem_path)
        if "aspect" in dem.columns and "aspect_sin" not in dem.columns:
            rad = np.deg2rad(dem["aspect"].astype(float))
            dem["aspect_sin"] = np.sin(rad)
            dem["aspect_cos"] = np.cos(rad)
        if "orographic_index" not in dem.columns and {"elevation", "slope"} <= set(dem.columns):
            dem["orographic_index"] = dem["elevation"] * dem["slope"]
        cols = ["cell_id"] + [c for c in DEM_FEATURES if c in dem.columns]
        return dem[cols].drop_duplicates("cell_id")
    raise FileNotFoundError("Could not load DEM features from MVP outputs or data/")


def build_sequence_for_sample(
    cell_id: str,
    feature_end_date: pd.Timestamp,
    era5_by_cell: dict[str, pd.DataFrame],
    dem_row: np.ndarray,
    history_days: int,
    era5_cols: list[str],
) -> np.ndarray | None:
    """Return [history_days, F] float32 or None if incomplete."""
    D = pd.Timestamp(feature_end_date).normalize()
    days = pd.date_range(D - pd.Timedelta(days=history_days - 1), D, freq="D")
    cell = era5_by_cell.get(cell_id)
    if cell is None or cell.empty:
        return None
    sub = cell.loc[cell["date"].isin(days)].set_index("date").reindex(days)
    if sub[era5_cols].isna().any().any():
        return None
    x = sub[era5_cols].to_numpy(dtype=np.float32)  # [T, 19]
    dem = np.broadcast_to(dem_row.astype(np.float32), (history_days, dem_row.shape[0]))
    return np.concatenate([x, dem], axis=1)


def build_sequences(
    manifest: pd.DataFrame,
    era5_daily: pd.DataFrame,
    dem: pd.DataFrame,
    sequences_dir: Path,
    history_days: int = 7,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Write sequences/{sample_id}.npy and return manifest with sequence_path."""
    sequences_dir = Path(sequences_dir)
    sequences_dir.mkdir(parents=True, exist_ok=True)

    era5_cols = [c for c in ERA5_DAY_FEATURES if c in era5_daily.columns]
    dem_cols = [c for c in DEM_FEATURES if c in dem.columns]
    if len(era5_cols) < len(ERA5_DAY_FEATURES):
        missing = set(ERA5_DAY_FEATURES) - set(era5_cols)
        logger.warning("Missing ERA5 day features: %s", sorted(missing))
    if not dem_cols:
        raise RuntimeError("No DEM features available for sequences")

    dem_map = dem.set_index("cell_id")[dem_cols]
    era5 = era5_daily[["cell_id", "date"] + era5_cols].copy()
    era5_by_cell = {cid: g for cid, g in era5.groupby("cell_id", sort=False)}

    rows = []
    n_fail = 0
    n_skip = 0
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        out_path = sequences_dir / f"{sample_id}.npy"
        rec = row.to_dict()
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            rec["sequence_path"] = str(out_path)
            rows.append(rec)
            n_skip += 1
            continue
        cell_id = row["cell_id"]
        if cell_id not in dem_map.index:
            n_fail += 1
            continue
        dem_row = dem_map.loc[cell_id].to_numpy(dtype=np.float32)
        seq = build_sequence_for_sample(
            cell_id,
            row["feature_end_date"],
            era5_by_cell,
            dem_row,
            history_days=history_days,
            era5_cols=era5_cols,
        )
        if seq is None:
            n_fail += 1
            continue
        np.save(out_path, seq.astype(np.float32))
        rec["sequence_path"] = str(out_path)
        rows.append(rec)

    logger.info(
        "Sequences: ok=%d  skip_existing=%d  fail=%d  shape=[%d,%d]",
        len(rows),
        n_skip,
        n_fail,
        history_days,
        len(era5_cols) + len(dem_cols),
    )
    out = pd.DataFrame(rows)
    out.attrs["seq_feature_dim"] = len(era5_cols) + len(dem_cols)
    out.attrs["era5_cols"] = era5_cols
    out.attrs["dem_cols"] = dem_cols
    return out
