"""Preprocess: Stage C + KNN for a label date."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger("preprocess.build_stage_c_day")


def _ensure_m4(m4_src: Path) -> None:
    src = str(m4_src)
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")


def _months_in_range(start: date, end: date) -> list[int]:
    months: set[int] = set()
    current = date(start.year, start.month, 1)
    while current <= end:
        months.add(current.month)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return sorted(months)


def _years_in_range(start: date, end: date) -> list[int]:
    return list(range(start.year, end.year + 1))


def _patch_flat_eo_listing(daily_cfg: dict) -> None:
    """Route M4 EO listing through flat parquet adapters when layout=flat_parquet."""
    from preprocess.adapters_gcs import list_flat_s2, list_flat_s5p
    import numerical_nextday.data.gcs_features as gf

    bucket = daily_cfg["gcs"]["bucket"]

    def flat_list(nf, bucket_name, prefix, years=None):  # noqa: ANN001, ARG001
        s2_prefix = daily_cfg["gcs"]["prefixes"]["sentinel2"].rstrip("/")
        if prefix.rstrip("/") == s2_prefix:
            df = list_flat_s2(bucket_name, prefix)
        else:
            df = list_flat_s5p(bucket_name, prefix)
        if years and not df.empty:
            df = df.loc[df["year"].isin(years)].copy()
        if df.empty:
            return df
        out = df.rename(columns={"gs_uri": "gs_uri"}).copy()
        out["size"] = None
        return out[["year", "month", "window", "blob_name", "gs_uri", "size"]]

    def flat_cache(nf, bucket_name, blob_name, dest, columns=None):  # noqa: ANN001
        from preprocess.adapters_gcs import load_flat_feature_table

        cache = Path(daily_cfg["paths"]["local_cache"]) / "eo_raw"
        uri = f"gs://{bucket_name}/{blob_name}"
        df = load_flat_feature_table(uri, cache, columns=columns)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        return dest

    gf.list_feature_files_robust = flat_list  # type: ignore[attr-defined]
    gf.cache_csv_robust = flat_cache  # type: ignore[attr-defined]


def run_stage_c_pipeline(
    label_date: date,
    daily_cfg: dict,
    m4_cfg: dict,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build Stage C KNN rows for lookback window ending at label_date."""
    from config_loader import setup_m4_imports, pipeline_today
    from paths import resolve_path

    m4_src = setup_m4_imports(daily_cfg)
    _ensure_m4(m4_src)
    _patch_flat_eo_listing(daily_cfg)

    from numerical_nextday.data.era5_firms import assemble_stage_a_year, build_era5_firms_month
    from numerical_nextday.data.s2_s5p import (
        attach_s2_stage_b,
        attach_s5p_stage_c,
        build_eo_cell_cache,
        merge_stage_a,
        write_splits,
    )

    lookback = daily_cfg["task"].get("lookback_days", 30)
    # Live prediction: panel through D is fine for row keys, but ERA5/FIRMS assemble
    # must not require calendar days after today / after available as-of.
    as_of = min(label_date, pipeline_today(daily_cfg))
    start = label_date - timedelta(days=lookback)
    years = _years_in_range(start, as_of)
    months = _months_in_range(start, as_of)
    end_clip = pd.Timestamp(as_of)
    lag = int(m4_cfg["task"]["era5_lag_days"])
    lead = int(m4_cfg["task"]["lead_days"])
    history = int(m4_cfg["task"]["history_days"])
    era5_start_clip = pd.Timestamp(start) - pd.Timedelta(days=lag + lead + history)
    era5_end_clip = pd.Timestamp(as_of) - pd.Timedelta(days=lag + lead)
    firms_start_clip = pd.Timestamp(start)

    logger.info(
        "Stage C build label_date=%s as_of=%s years=%s months=%s "
        "labels %s…%s era5_from=%s era5_to=%s",
        label_date,
        as_of,
        years,
        months,
        start,
        as_of,
        era5_start_clip.date(),
        era5_end_clip.date(),
    )

    for year in years:
        year_months = [
            m
            for m in months
            if (year > start.year or m >= start.month)
            and (year < as_of.year or m <= as_of.month)
        ]
        if not year_months:
            year_months = months
        for month in year_months:
            build_era5_firms_month(
                m4_cfg,
                year,
                month,
                worker="daily",
                force=force,
                start_clip=era5_start_clip,
                end_clip=end_clip,
                era5_end_clip=era5_end_clip,
                firms_start_clip=firms_start_clip,
            )
        assemble_stage_a_year(
            m4_cfg,
            year,
            months=year_months,
            worker="daily",
            era5_lag_days=m4_cfg["task"]["era5_lag_days"],
            force=force,
            start_clip=pd.Timestamp(start),
            end_clip=end_clip,
        )

    merge_stage_a(m4_cfg, years)
    cache = Path(m4_cfg["paths"]["shared_cache"])
    stage_a = pd.read_parquet(cache / "stage_a" / "all.parquet")
    write_splits(m4_cfg, stage_a, "stage_a")

    build_eo_cell_cache(m4_cfg, "s2", years, months=months, worker="daily", force=force)
    attach_s2_stage_b(m4_cfg, years)
    build_eo_cell_cache(m4_cfg, "s5p", years, months=months, worker="daily", force=force)
    attach_s5p_stage_c(m4_cfg, years)

    stage_c_path = cache / "stage_c" / "all.parquet"
    if not stage_c_path.exists():
        raise FileNotFoundError(f"Stage C not produced: {stage_c_path}")

    stage_c_knn = _apply_knn(stage_c_path, cache / "stage_c_knn", daily_cfg)
    panel = pd.read_parquet(stage_c_knn / "all.parquet")
    panel["label_date"] = pd.to_datetime(panel["label_date"]).dt.normalize()
    mask = (panel["label_date"] >= pd.Timestamp(start)) & (panel["label_date"] <= pd.Timestamp(label_date))
    history = panel.loc[mask].copy()

    history_dir = cache / "stage_c_knn" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history.to_parquet(history_dir / "panel.parquet", index=False)

    day_dir = cache / "stage_c_knn" / f"day={label_date.isoformat()}"
    day_dir.mkdir(parents=True, exist_ok=True)
    day_rows = panel.loc[panel["label_date"].eq(pd.Timestamp(label_date))].copy()
    day_rows.to_parquet(day_dir / "stage_c.parquet", index=False)

    return history


def _forward_fill_s2(frame: pd.DataFrame, targets: list[str], max_days: int) -> pd.DataFrame:
    """Per-cell causal ffill of S2 bands, at most max_days after last valid observation."""
    if not targets or max_days < 1 or frame.empty:
        return frame
    out = frame.copy()
    out["label_date"] = pd.to_datetime(out["label_date"]).dt.normalize()
    out = out.sort_values(["cell_id", "label_date"]).reset_index(drop=True)
    filled = 0
    chunks: list[pd.DataFrame] = []
    for _, grp in out.groupby("cell_id", sort=False):
        part = grp.copy()
        missing = part["s2n_available"].fillna(0).astype(int).eq(0)
        part[targets] = part[targets].ffill(limit=max_days)
        now_ok = part[targets].notna().any(axis=1)
        newly = missing & now_ok
        n_new = int(newly.sum())
        if n_new:
            part.loc[newly, "s2n_available"] = 1
            if "s2n_lag_days" in part.columns:
                last_obs = part["label_date"].where(~missing).ffill()
                extra = (part["label_date"] - last_obs).dt.days.fillna(0)
                prev = pd.to_numeric(part["s2n_lag_days"], errors="coerce").fillna(0)
                part.loc[newly, "s2n_lag_days"] = prev.loc[newly] + extra.loc[newly]
            filled += n_new
        chunks.append(part)
    out = pd.concat(chunks, ignore_index=True)
    logger.info(
        "S2 forward-fill (limit=%d days): filled %d previously missing rows",
        max_days,
        filled,
    )
    return out


def _apply_knn(stage_c_path: Path, out_dir: Path, daily_cfg: dict) -> Path:
    """Impute missing S2 rows with a causal, distance-weighted donor pool.

    Training used observed rows from 2019–2022 as frozen donors. A daily live
    build intentionally contains only the configured lookback window, so those
    historical rows are normally absent. In that case, use observed S2 rows in
    the live window; every one of those observations is still available no
    later than D−1. This keeps the deployed stage finite without silently
    turning the model's passthrough preprocessor into a different estimator.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = pd.read_parquet(stage_c_path)
    S2_SKIP = {"s2n_available", "s2n_lag_days", "s2n_knn_imputed"}
    S5_SKIP = {"s5n_available", "s5n_lag_days"}
    ID_SKIP = {
        "cell_id", "latitude", "longitude", "feature_end_date", "label_date",
        "eo_asof_date", "y_fire", "firms_n_pixels", "firms_max_confidence",
        "region", "era5_lag_days", "s5p_2021_status",
    }

    def s2_targets(columns):
        return [c for c in columns if c.startswith("s2n_") and c not in S2_SKIP]

    def predictors(columns):
        weather_dem = [
            c for c in columns
            if c not in ID_SKIP and not c.startswith("s2n_") and not c.startswith("s5n_") and c != "s2n_knn_imputed"
        ]
        s5 = [c for c in columns if c.startswith("s5n_") and c not in S5_SKIP]
        return weather_dem + s5

    targets = s2_targets(df.columns.tolist())
    preds = predictors(df.columns.tolist())
    years = pd.to_datetime(df["label_date"]).dt.year
    miss = df["s2n_available"].fillna(0).astype(int).to_numpy() == 0
    observed = df["s2n_available"].fillna(0).astype(int).to_numpy() == 1
    target_values = df[targets].apply(pd.to_numeric, errors="coerce")
    observed &= np.isfinite(target_values.to_numpy(dtype=float)).all(axis=1)
    historical_donor = observed & (years.to_numpy() <= 2022)
    donor = historical_donor if historical_donor.any() else observed

    out = df.copy()
    out["s2n_knn_imputed"] = 0
    n_donors = int(donor.sum())
    n_miss = int(miss.sum())
    if n_miss and n_donors == 0:
        raise ValueError(
            f"Cannot KNN-impute {n_miss} missing S2 rows: the live window has no "
            "fully observed Sentinel-2 donor rows"
        )
    if n_miss and n_donors and targets:
        if not historical_donor.any():
            logger.info(
                "No 2019–2022 donors in the live lookback; using %d causal observed "
                "S2 rows from the current window",
                n_donors,
            )
        X = out[preds].apply(pd.to_numeric, errors="coerce")
        donor_medians = X.loc[donor].median(numeric_only=True)
        usable_predictors = donor_medians.loc[donor_medians.notna()].index.tolist()
        if not usable_predictors:
            raise ValueError("Cannot KNN-impute S2: no finite donor predictors are available")
        X_d = X.loc[donor, usable_predictors].fillna(donor_medians[usable_predictors])
        X_m = X.loc[miss, usable_predictors].fillna(donor_medians[usable_predictors])
        scaler = StandardScaler()
        X_d_s = scaler.fit_transform(X_d)
        X_m_s = scaler.transform(X_m)
        k = daily_cfg["preprocess"].get("knn_neighbors", 5)
        nn = NearestNeighbors(n_neighbors=min(k, n_donors), metric="euclidean")
        nn.fit(X_d_s)
        dist, idx = nn.kneighbors(X_m_s)
        weights = 1.0 / np.maximum(dist, 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        donor_vals = target_values.loc[donor, targets].to_numpy(dtype=float)
        for i, row_idx in enumerate(np.flatnonzero(miss)):
            out.loc[row_idx, targets] = (donor_vals[idx[i]] * weights[i][:, None]).sum(axis=0)
            out.loc[row_idx, "s2n_knn_imputed"] = 1
        remaining = ~np.isfinite(
            out.loc[miss, targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        )
        if remaining.any():
            raise ValueError(
                f"KNN imputation left {int(remaining.sum())} non-finite Sentinel-2 values"
            )
        logger.info("KNN imputed %d S2 rows from %d observed donors", n_miss, n_donors)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "all.parquet", index=False)
    return out_dir
