from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ERA5_DAY_FEATURES = [
    "t2m_mean",
    "t2m_max",
    "t2m_min",
    "d2m_mean",
    "rh_mean",
    "sp_mean",
    "wind_speed_mean",
    "wind_dir_sin",
    "wind_dir_cos",
    "i10fg_max",
    "tp_sum_mm",
    "swvl1_mean",
    "swvl2_mean",
    "soil_moisture_index",
    "cvh_mean",
    "cvl_mean",
    "lai_hv_mean",
    "lai_lv_mean",
    "blh_mean",
]

DEM_FEATURES = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tri",
    "tpi",
    "orographic_index",
    "hillshade",
]

# Kept for hard-negative sampling / optional diagnostics (not fed to LSTM)
WINDOW_FEATURES = [
    "t2m_max_7d",
    "tp_sum_7d",
    "wind_speed_max_7d",
    "rh_min_7d",
    "swvl1_mean_7d",
    "i10fg_max_7d",
]


def load_mvp_frames(mvp_dir) -> pd.DataFrame:
    frames = []
    for name in ("train", "val", "test"):
        path = mvp_dir / f"{name}.parquet"
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_parquet(path)
            if len(df):
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No non-empty MVP parquets in {mvp_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["label_date"] = pd.to_datetime(out["label_date"]).dt.normalize()
    out["feature_end_date"] = pd.to_datetime(out["feature_end_date"]).dt.normalize()
    return out


def assign_split(label_date: pd.Series, train_end: str, val_end: str) -> pd.Series:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    split = pd.Series(index=label_date.index, dtype="object")
    split.loc[label_date <= train_end_ts] = "train"
    split.loc[(label_date > train_end_ts) & (label_date <= val_end_ts)] = "val"
    split.loc[label_date > val_end_ts] = "test"
    return split


def assert_year_splits(df: pd.DataFrame, train_years: list[int], val_year: int, test_year: int) -> None:
    years = {
        split: set(pd.to_datetime(g["label_date"]).dt.year.unique().tolist())
        for split, g in df.groupby("split")
    }
    expect = {
        "train": set(train_years),
        "val": {val_year},
        "test": {test_year},
    }
    for name, want in expect.items():
        got = years.get(name, set())
        if got != want:
            raise ValueError(f"Split {name} years={sorted(got)} expected {sorted(want)}")


def balanced_sample(
    df: pd.DataFrame,
    neg_pos_ratio: float,
    random_seed: int,
    max_train: int,
    max_val: int,
    max_test: int,
    hard_negative: bool = True,
) -> pd.DataFrame:
    """Keep all positives; sample negatives per split with optional hard-negative bias."""
    rng = np.random.default_rng(random_seed)
    caps = {"train": max_train, "val": max_val, "test": max_test}
    parts = []

    for split_name, group in df.groupby("split"):
        pos = group.loc[group["y_fire"] == 1]
        neg = group.loc[group["y_fire"] == 0]
        n_pos = len(pos)
        n_neg_target = int(min(len(neg), max(n_pos * neg_pos_ratio, n_pos)))

        if len(neg) and n_neg_target > 0:
            if hard_negative and "t2m_max" in neg.columns and "rh_min_7d" in neg.columns:
                score = neg["t2m_max"].rank(pct=True).fillna(0.5) + (
                    1.0 - neg["rh_min_7d"].rank(pct=True).fillna(0.5)
                )
                weights = score.to_numpy(dtype=np.float64)
                weights = np.clip(weights, 1e-6, None)
                weights = weights / weights.sum()
                n_take = min(n_neg_target, len(neg))
                idx = rng.choice(neg.index.to_numpy(), size=n_take, replace=False, p=weights)
                neg_s = neg.loc[idx]
            else:
                neg_s = neg.sample(n=min(n_neg_target, len(neg)), random_state=random_seed)
        else:
            neg_s = neg.iloc[0:0]

        part = pd.concat([pos, neg_s], ignore_index=True)
        cap = caps.get(split_name, len(part))
        if len(part) > cap:
            if n_pos >= cap:
                part = pos.sample(n=cap, random_state=random_seed)
            else:
                n_neg_keep = cap - n_pos
                neg_keep = neg_s.sample(n=n_neg_keep, random_state=random_seed)
                part = pd.concat([pos, neg_keep], ignore_index=True)

        logger.info(
            "sample %-5s  n=%d  pos=%d  neg=%d",
            split_name,
            len(part),
            int(part["y_fire"].sum()),
            int((part["y_fire"] == 0).sum()),
        )
        parts.append(part)

    if not parts:
        raise ValueError(
            "balanced_sample received an empty frame. "
            "Rebuild mvp_era5_dem for 2022–2025 before running the fusion dataset builder."
        )
    out = pd.concat(parts, ignore_index=True)
    out["sample_id"] = [
        f"{cid}_{pd.Timestamp(d).strftime('%Y%m%d')}"
        for cid, d in zip(out["cell_id"], out["feature_end_date"])
    ]
    return out
