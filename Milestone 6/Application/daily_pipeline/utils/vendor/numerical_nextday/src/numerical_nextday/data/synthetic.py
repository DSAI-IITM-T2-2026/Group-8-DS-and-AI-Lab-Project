"""Build a small synthetic Stage A/B/C for pipeline / training dry-runs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from numerical_nextday.config import shared_cache
from numerical_nextday.data.s2_s5p import write_splits

logger = logging.getLogger(__name__)

ERA5_COLS = [
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
    "t2m_max_7d",
    "tp_sum_7d",
    "wind_speed_max_7d",
    "rh_min_7d",
    "swvl1_mean_7d",
    "i10fg_max_7d",
]
DEM_COLS = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tri",
    "tpi",
    "orographic_index",
    "hillshade",
]
S2_COLS = [
    "B2_mean",
    "B3_mean",
    "B4_mean",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "B2_std",
    "B3_std",
    "B4_std",
    "B8_std",
    "B11_std",
    "B12_std",
    "NDVI_mean",
    "NDMI_mean",
    "NBR_mean",
    "NDWI_mean",
    "EVI_mean",
    "cloud_percentage",
    "valid_fraction",
]
S5_COLS = [
    "s5p_aai_mean",
    "s5p_aai_max",
    "s5p_aai_std",
    "s5p_aai_valid_fraction",
    "s5p_co_mean",
    "s5p_co_max",
    "s5p_co_std",
    "s5p_co_valid_fraction",
    "s5p_data_available",
]


def build_synthetic_tables(cfg: dict, n_cells: int = 40, seed: int = 42) -> None:
    """
    Create stage_a/b/c all+splits with enough positives for LightGBM smoke.
    Years: 2020–2025 sparse months (Aug focus) so splits are non-empty.
    """
    rng = np.random.default_rng(seed)
    cache = shared_cache(cfg)
    rows = []
    cells = [f"c{i:03d}" for i in range(n_cells)]
    for year in range(2020, 2026):
        for month in [1, 2, 3, 5, 8, 11, 12]:
            for day in (5, 15, 25):
                try:
                    label = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                feat = label - pd.Timedelta(days=6)
                for cid in cells:
                    lat = 34.0 + (hash(cid) % 80) * 0.05
                    lon = -120.0 + (hash(cid[::-1]) % 80) * 0.05
                    rec = {
                        "cell_id": cid,
                        "latitude": lat,
                        "longitude": lon,
                        "feature_end_date": feat,
                        "label_date": label,
                        "eo_asof_date": label - pd.Timedelta(days=1),
                        "era5_lag_days": 5,
                        "region": f"cell:{cid}",
                        "firms_n_pixels": 0,
                        "firms_max_confidence": np.nan,
                    }
                    for c in ERA5_COLS + DEM_COLS:
                        rec[c] = float(rng.normal())
                    # Inject signal: hot + dry → fire
                    score = rec["t2m_max"] - rec["rh_mean"] + 0.3 * rec["wind_speed_mean"]
                    y = int(score > 1.2 and month in (5, 8, 11) and rng.random() < 0.35)
                    if month in (1, 2, 12):
                        y = int(rng.random() < 0.02)
                    rec["y_fire"] = y
                    if y:
                        rec["firms_n_pixels"] = int(rng.integers(1, 5))
                        rec["firms_max_confidence"] = float(rng.uniform(40, 100))
                    rows.append(rec)

    stage_a = pd.DataFrame(rows)
    for c in S2_COLS:
        stage_a["s2n_" + c] = stage_a["t2m_mean"] * 0.1 + rng.normal(size=len(stage_a))
    stage_a["s2n_available"] = 1
    stage_a["s2n_lag_days"] = 2

    stage_b = stage_a.copy()
    for c in S5_COLS:
        stage_b["s5n_" + c] = rng.normal(size=len(stage_b)) * 0.1
    stage_b["s5n_available"] = 1
    stage_b["s5n_lag_days"] = 1
    mask_2021 = stage_b["label_date"].dt.year == 2021
    for c in S5_COLS:
        stage_b.loc[mask_2021, "s5n_" + c] = 0.0
    stage_b.loc[mask_2021, "s5n_available"] = 0
    stage_b["s5p_2021_status"] = np.where(mask_2021, "placeholder", "ready")

    stage_c = stage_b.copy()

    for name, df in [("stage_a", stage_a.drop(columns=[c for c in stage_a.columns if c.startswith("s2n_") or c.startswith("s5n_") or c == "s5p_2021_status"], errors="ignore")),
                     ("stage_b", stage_b.drop(columns=[c for c in stage_b.columns if c.startswith("s5n_") or c == "s5p_2021_status"], errors="ignore")),
                     ("stage_c", stage_c)]:
        out = cache / name
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "all.parquet", index=False)
        write_splits(cfg, df, name)
        logger.info("Synthetic %s n=%d pos=%d", name, len(df), int(df["y_fire"].sum()))
