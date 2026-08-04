#!/usr/bin/env python3
"""
Build a new Stage C variant with KNN-imputed S2 (K=5 by default).

Does NOT overwrite stage_c/. Writes to outputs/m4_shared_cache/stage_c_knn/.

Rules:
  - Detect missing S2 via s2n_available == 0 (values may already be train-median).
  - Donor pool = train years with s2n_available == 1 (no val/test leakage).
  - Predictors = weather + DEM + S5P measurements (not labels).
  - Keep s2n_available=0; set s2n_knn_imputed=1 on filled rows.

Example:
  cd "Milestone 4/numerical_nextday"
  export PYTHONPATH=src
  python scripts/build_stage_c_knn.py --n-neighbors 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
STAGE_C = ROOT / "outputs" / "m4_shared_cache" / "stage_c"
OUT_DEFAULT = ROOT / "outputs" / "m4_shared_cache" / "stage_c_knn"

S2_SKIP = {"s2n_available", "s2n_lag_days", "s2n_knn_imputed"}
S5_SKIP = {"s5n_available", "s5n_lag_days"}
ID_SKIP = {
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
}


def _s2_targets(columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c.startswith("s2n_") and c not in S2_SKIP
    ]


def _predictors(columns: list[str]) -> list[str]:
    weather_dem = [
        c
        for c in columns
        if c not in ID_SKIP
        and not c.startswith("s2n_")
        and not c.startswith("s5n_")
        and c
        not in {
            "s2n_knn_imputed",
        }
    ]
    s5 = [c for c in columns if c.startswith("s5n_") and c not in S5_SKIP]
    return weather_dem + s5


def _impute_block(
    df: pd.DataFrame,
    miss_mask: np.ndarray,
    donor_mask: np.ndarray,
    predictors: list[str],
    targets: list[str],
    n_neighbors: int,
) -> tuple[pd.DataFrame, dict]:
    """Distance-weighted KNN fill for target cols on miss_mask rows."""
    out = df.copy()
    meta: dict = {
        "n_missing": int(miss_mask.sum()),
        "n_donors": int(donor_mask.sum()),
        "n_neighbors": n_neighbors,
        "predictors": predictors,
        "targets": targets,
    }
    if miss_mask.sum() == 0:
        out["s2n_knn_imputed"] = 0
        meta["skipped"] = "no_missing"
        return out, meta

    X = out[predictors].apply(pd.to_numeric, errors="coerce")
    # Fill predictor NaNs with donor medians so NN can run
    donor_med = X.loc[donor_mask].median(numeric_only=True)
    X = X.fillna(donor_med)

    scaler = StandardScaler()
    X_donor = scaler.fit_transform(X.loc[donor_mask])
    X_miss = scaler.transform(X.loc[miss_mask])

    k = min(n_neighbors, int(donor_mask.sum()))
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X_donor)
    dists, inds = nn.kneighbors(X_miss)
    t_nn = time.perf_counter() - t0

    Y_donor = out.loc[donor_mask, targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    miss_idx = np.flatnonzero(miss_mask)
    filled = np.zeros((len(miss_idx), len(targets)), dtype=float)

    for i, (neigh, dist) in enumerate(zip(inds, dists)):
        w = 1.0 / np.maximum(dist, 1e-6)
        for j in range(len(targets)):
            vals = Y_donor[neigh, j]
            m = np.isfinite(vals)
            if not m.any():
                filled[i, j] = float(np.nanmedian(Y_donor[:, j]))
            else:
                filled[i, j] = float(np.average(vals[m], weights=w[m]))

    for j, col in enumerate(targets):
        out.iloc[miss_idx, out.columns.get_loc(col)] = filled[:, j]

    out["s2n_knn_imputed"] = 0
    out.loc[miss_mask, "s2n_knn_imputed"] = 1
    # Keep honesty flag: still unavailable as real observation
    # s2n_available stays 0

    meta["timings_sec"] = {"nearest_neighbors_impute": round(t_nn, 3)}
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-c", type=Path, default=STAGE_C)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--train-end-year", type=int, default=2022)
    args = ap.parse_args()

    src = args.stage_c / "all.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    t_load0 = time.perf_counter()
    df = pd.read_parquet(src)
    t_load = time.perf_counter() - t_load0
    print(f"loaded {len(df)} rows in {t_load:.2f}s from {src}")

    years = pd.to_datetime(df["label_date"]).dt.year
    predictors = _predictors(df.columns.tolist())
    targets = _s2_targets(df.columns.tolist())
    if not targets:
        raise SystemExit("No S2 target columns found")
    if not predictors:
        raise SystemExit("No predictor columns found")

    miss = df["s2n_available"].fillna(0).astype(int).to_numpy() == 0
    donor = (df["s2n_available"].fillna(0).astype(int).to_numpy() == 1) & (
        years.to_numpy() <= args.train_end_year
    )
    print(
        f"S2 missing={miss.sum()} donors(train avail)={donor.sum()} "
        f"K={args.n_neighbors} predictors={len(predictors)} targets={len(targets)}"
    )

    t0 = time.perf_counter()
    out, knn_meta = _impute_block(
        df, miss, donor, predictors, targets, args.n_neighbors
    )
    t_impute = time.perf_counter() - t0
    print(f"impute done in {t_impute:.2f}s → imputed {int(out['s2n_knn_imputed'].sum())} rows")

    # Splits (same year cut as Stage C)
    y = pd.to_datetime(out["label_date"]).dt.year
    train = out.loc[y <= 2022].copy()
    val = out.loc[(y >= 2023) & (y <= 2024)].copy()
    test = out.loc[y >= 2025].copy()

    args.out.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    t_w0 = time.perf_counter()
    out.to_parquet(args.out / "all.parquet", index=False)
    train.to_parquet(args.out / "train.parquet", index=False)
    val.to_parquet(args.out / "val.parquet", index=False)
    test.to_parquet(args.out / "test.parquet", index=False)
    t_write = time.perf_counter() - t_w0

    # Feature allowlist = original Stage C features (+ keep s2n_available)
    feat_src = args.stage_c / "metadata" / "feature_columns.json"
    if feat_src.exists():
        feat_cols = json.loads(feat_src.read_text())
    else:
        feat_cols = [c for c in predictors + targets if c in out.columns]
        if "s2n_available" in out.columns and "s2n_available" not in feat_cols:
            feat_cols.append("s2n_available")
        if "s5n_available" in out.columns and "s5n_available" not in feat_cols:
            feat_cols.append("s5n_available")

    dataset_meta = {
        "stage": "stage_c_knn",
        "source_stage": "stage_c",
        "era5_lag_days": 5,
        "s5p_2021_mode": "ready",
        "knn": {
            "n_neighbors": args.n_neighbors,
            "weights": "distance",
            "targets": "s2_measurements_where_s2n_available_eq_0",
            "donor_pool": f"train_years_<=_{args.train_end_year}_and_s2n_available_eq_1",
            **knn_meta,
        },
        "years": sorted(y.unique().tolist()),
        "n_rows": int(len(out)),
        "n_pos": int(out["y_fire"].sum()) if "y_fire" in out.columns else None,
        "n_s2_knn_imputed": int(out["s2n_knn_imputed"].sum()),
        "split": {"train_end": "2022-12-31", "val_end": "2024-12-31"},
        "feature_columns": feat_cols,
        "timings_sec": {
            "load": round(t_load, 3),
            "impute": round(t_impute, 3),
            "write": round(t_write, 3),
        },
        "note": "s2n_available remains 0 on imputed rows; s2n_knn_imputed marks KNN fills.",
    }
    (meta_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (meta_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset_meta, indent=2, default=str)
    )
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "stage": "stage_c_knn",
                "s5p_2021_mode": "ready",
                "n_neighbors": args.n_neighbors,
                "n_rows": int(len(out)),
                "n_s2_knn_imputed": int(out["s2n_knn_imputed"].sum()),
            },
            indent=2,
        )
    )

    print(json.dumps(dataset_meta, indent=2, default=str))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
