"""GCS numerical feature tables (S2 5-day windows, S5P daily windows)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _gcs_client(anonymous: bool = True):
    from google.cloud.storage import Client

    if anonymous:
        try:
            return Client.create_anonymous_client()
        except Exception:
            pass
    return Client()


def list_feature_files(bucket: str, prefix: str, years: list[int] | None = None) -> pd.DataFrame:
    """List year=/month=/window=/features.csv objects."""
    client = _gcs_client(anonymous=True)
    rows = []
    pat = re.compile(
        r"year=(\d+)/month=(\d+)/window=(\d+)/features\.csv$"
    )
    full_prefix = prefix.rstrip("/") + "/"
    for blob in client.list_blobs(bucket, prefix=full_prefix):
        m = pat.search(blob.name)
        if not m:
            continue
        year, month, window = map(int, m.groups())
        if years and year not in years:
            continue
        rows.append(
            {
                "year": year,
                "month": month,
                "window": window,
                "blob_name": blob.name,
                "gs_uri": f"gs://{bucket}/{blob.name}",
                "size": blob.size,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No feature CSVs under gs://%s/%s", bucket, prefix)
    else:
        logger.info("Indexed %d feature files under %s/%s", len(df), bucket, prefix)
    return df.sort_values(["year", "month", "window"]).reset_index(drop=True)


def cache_csv_to_parquet(
    bucket: str,
    blob_name: str,
    dest: Path,
    columns: list[str] | None = None,
) -> Path:
    """Download one features.csv and write parquet (skip if exists)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    client = _gcs_client(anonymous=True)
    blob = client.bucket(bucket).blob(blob_name)
    tmp_csv = dest.with_suffix(".csv.partial")
    logger.info("Downloading gs://%s/%s", bucket, blob_name)
    blob.download_to_filename(str(tmp_csv))
    usecols = None
    if columns:
        # Always keep join keys
        base = ["grid_id", "latitude", "longitude", "window_start", "window_end"]
        usecols = list(dict.fromkeys(base + columns))
    try:
        df = pd.read_csv(tmp_csv, usecols=lambda c: usecols is None or c in usecols)
    except ValueError:
        df = pd.read_csv(tmp_csv)
        keep = [c for c in (usecols or df.columns) if c in df.columns]
        df = df[keep]
    df["window_start"] = pd.to_datetime(df["window_start"]).dt.normalize()
    df["window_end"] = pd.to_datetime(df["window_end"]).dt.normalize()
    df.to_parquet(dest, index=False)
    try:
        tmp_csv.unlink()
    except OSError:
        pass
    return dest


def build_era5_feature_grid_map(
    dem_cells: pd.DataFrame,
    feature_sample: pd.DataFrame,
    resolution: float = 0.25,
) -> pd.DataFrame:
    """
    Map each ERA5 cell_id to feature grid_ids inside the 0.25° cell (or nearest).

    dem_cells needs: cell_id, latitude, longitude
    feature_sample needs: grid_id, latitude, longitude (one snapshot is enough)
    """
    feats = feature_sample[["grid_id", "latitude", "longitude"]].drop_duplicates("grid_id")
    half = resolution / 2.0
    rows = []
    f_lat = feats["latitude"].to_numpy()
    f_lon = feats["longitude"].to_numpy()
    f_ids = feats["grid_id"].to_numpy()

    for _, cell in dem_cells.iterrows():
        clat, clon = float(cell["latitude"]), float(cell["longitude"])
        cid = cell["cell_id"]
        mask = (
            (f_lat >= clat - half)
            & (f_lat < clat + half)
            & (f_lon >= clon - half)
            & (f_lon < clon + half)
        )
        if mask.any():
            for gid in f_ids[mask]:
                rows.append({"cell_id": cid, "grid_id": gid, "join": "in_cell"})
        else:
            d2 = (f_lat - clat) ** 2 + (f_lon - clon) ** 2
            gid = f_ids[int(np.argmin(d2))]
            rows.append({"cell_id": cid, "grid_id": gid, "join": "nearest"})
    return pd.DataFrame(rows)


def aggregate_features_to_cells(
    features: pd.DataFrame,
    grid_map: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """Mean feature values per ERA5 cell_id for one window table."""
    cols = [c for c in value_cols if c in features.columns]
    merged = features.merge(grid_map[["cell_id", "grid_id"]], on="grid_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["cell_id", "window_start", "window_end"] + cols)
    ag = (
        merged.groupby("cell_id", as_index=False)[cols]
        .mean(numeric_only=True)
    )
    ag["window_start"] = features["window_start"].iloc[0]
    ag["window_end"] = features["window_end"].iloc[0]
    return ag


def attach_forward_filled(
    samples: pd.DataFrame,
    window_tables: list[pd.DataFrame],
    value_cols: list[str],
    date_col: str = "feature_end_date",
    max_lag_days: int | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """
    For each sample day D, attach features from window covering D,
    else latest window_end <= D (optional max_lag).
    """
    if not window_tables:
        out = samples.copy()
        for c in value_cols:
            out[prefix + c] = np.nan
        out[prefix + "available"] = 0
        out[prefix + "lag_days"] = np.nan
        return out

    panels = []
    for w in window_tables:
        cols = [c for c in value_cols if c in w.columns]
        part = w[["cell_id", "window_start", "window_end"] + cols].copy()
        panels.append(part)
    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["cell_id", "window_end"])

    samples = samples.copy()
    samples[date_col] = pd.to_datetime(samples[date_col]).dt.normalize()
    attached_rows = []
    for cell_id, grp in samples.groupby("cell_id"):
        cell_panel = panel.loc[panel["cell_id"] == cell_id]
        if cell_panel.empty:
            for _, row in grp.iterrows():
                rec = row.to_dict()
                for c in value_cols:
                    rec[prefix + c] = np.nan
                rec[prefix + "available"] = 0
                rec[prefix + "lag_days"] = np.nan
                attached_rows.append(rec)
            continue
        ends = cell_panel["window_end"].to_numpy()
        starts = cell_panel["window_start"].to_numpy()
        for _, row in grp.iterrows():
            D = row[date_col]
            # covering window
            cover = (starts <= D) & (ends >= D)
            if cover.any():
                idx = int(np.where(cover)[0][-1])
            else:
                prior = ends <= D
                if not prior.any():
                    rec = row.to_dict()
                    for c in value_cols:
                        rec[prefix + c] = np.nan
                    rec[prefix + "available"] = 0
                    rec[prefix + "lag_days"] = np.nan
                    attached_rows.append(rec)
                    continue
                idx = int(np.where(prior)[0][-1])
            wrow = cell_panel.iloc[idx]
            lag = (D - wrow["window_end"]).days
            if max_lag_days is not None and lag > max_lag_days:
                rec = row.to_dict()
                for c in value_cols:
                    rec[prefix + c] = np.nan
                rec[prefix + "available"] = 0
                rec[prefix + "lag_days"] = lag
                attached_rows.append(rec)
                continue
            rec = row.to_dict()
            for c in value_cols:
                rec[prefix + c] = wrow[c] if c in wrow.index else np.nan
            rec[prefix + "available"] = 1
            rec[prefix + "lag_days"] = lag
            attached_rows.append(rec)
    return pd.DataFrame(attached_rows)
