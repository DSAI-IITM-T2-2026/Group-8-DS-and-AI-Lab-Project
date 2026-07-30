from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import ERA5_DAY_FEATURES
from ..grid import make_cell_id
from ..io import atomic_parquet
from .eo import eo_cache_path
from .era5 import daily_cache_path
from .firms import firms_cache_path


def build_synthetic_inputs(cfg: dict, cells: int = 12, seed: int = 42) -> None:
    """Create a small deterministic 2019–2025 fixture that exercises every stage."""
    rng = np.random.default_rng(seed)
    latitudes = 38.9 + 0.25 * np.arange(cells // 3)
    longitudes = -121.9 + 0.25 * np.arange(3)
    coordinates = [(lat, lon) for lat in latitudes for lon in longitudes][:cells]
    dem = pd.DataFrame(
        {
            "cell_id": [make_cell_id(lat, lon) for lat, lon in coordinates],
            "latitude": [lat for lat, _ in coordinates],
            "longitude": [lon for _, lon in coordinates],
            "elevation": rng.uniform(100, 2200, cells),
            "slope": rng.uniform(0, 35, cells),
            "aspect": rng.uniform(0, 360, cells),
            "hillshade": rng.uniform(80, 220, cells),
            "tri": rng.uniform(0, 12, cells),
            "tpi": rng.normal(0, 5, cells),
        }
    )
    cfg["paths"]["dem_cells"].parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(dem, cfg["paths"]["dem_cells"])

    dates = pd.date_range("2018-12-15", "2025-12-31", freq="D")
    base_rows = pd.MultiIndex.from_product(
        [dates, dem["cell_id"]], names=["date", "cell_id"]
    ).to_frame(index=False)
    base_rows = base_rows.merge(dem[["cell_id", "latitude", "longitude"]], on="cell_id")
    day = base_rows["date"].dt.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (day - 100) / 365.25)
    noise = rng.normal(size=len(base_rows))
    base_rows["t2m_mean"] = 287 + 11 * season + noise
    base_rows["t2m_max"] = base_rows["t2m_mean"] + 6 + rng.normal(0, 1, len(base_rows))
    base_rows["t2m_min"] = base_rows["t2m_mean"] - 5 + rng.normal(0, 1, len(base_rows))
    base_rows["d2m_mean"] = base_rows["t2m_mean"] - rng.uniform(3, 14, len(base_rows))
    base_rows["rh_mean"] = np.clip(65 - 28 * season + rng.normal(0, 8, len(base_rows)), 8, 100)
    base_rows["sp_mean"] = 90000 + rng.normal(0, 900, len(base_rows))
    base_rows["wind_speed_mean"] = np.abs(rng.normal(4 + season, 2, len(base_rows)))
    angle = rng.uniform(-np.pi, np.pi, len(base_rows))
    base_rows["wind_dir_sin"] = np.sin(angle)
    base_rows["wind_dir_cos"] = np.cos(angle)
    base_rows["i10fg_max"] = base_rows["wind_speed_mean"] + rng.uniform(1, 10, len(base_rows))
    wet = np.clip(2.5 - 2 * season + rng.normal(0, 1.5, len(base_rows)), 0, None)
    base_rows["tp_sum_mm"] = wet
    base_rows["swvl1_mean"] = np.clip(
        0.25 - 0.1 * season + rng.normal(0, 0.03, len(base_rows)), 0.02, 0.5
    )
    base_rows["swvl2_mean"] = np.clip(base_rows["swvl1_mean"] + 0.04, 0.02, 0.6)
    base_rows["soil_moisture_index"] = (base_rows["swvl1_mean"] + base_rows["swvl2_mean"]) / 2
    base_rows["cvh_mean"] = rng.uniform(0.2, 0.8, len(base_rows))
    base_rows["cvl_mean"] = 1 - base_rows["cvh_mean"]
    base_rows["lai_hv_mean"] = rng.uniform(0.5, 5, len(base_rows))
    base_rows["lai_lv_mean"] = rng.uniform(0, 2, len(base_rows))
    base_rows["blh_mean"] = rng.uniform(100, 1600, len(base_rows))
    missing = set(ERA5_DAY_FEATURES) - set(base_rows)
    if missing:
        raise AssertionError(f"Synthetic ERA5 generator is incomplete: {missing}")

    for (year, month), frame in base_rows.groupby(
        [base_rows["date"].dt.year, base_rows["date"].dt.month]
    ):
        atomic_parquet(frame, daily_cache_path(cfg, int(year), int(month)))

    label_dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    label_grid = pd.MultiIndex.from_product(
        [label_dates, dem["cell_id"]], names=["date", "cell_id"]
    ).to_frame(index=False)
    source_weather = base_rows[["date", "cell_id", "t2m_max", "rh_mean", "wind_speed_mean"]]
    source_weather = source_weather.rename(columns={"date": "weather_date"})
    label_grid["weather_date"] = label_grid["date"] - pd.Timedelta(days=6)
    label_grid = label_grid.merge(source_weather, on=["weather_date", "cell_id"], how="left")
    logit = (
        -4.0
        + 0.10 * (label_grid["t2m_max"] - 295)
        - 0.025 * (label_grid["rh_mean"] - 35)
        + 0.08 * label_grid["wind_speed_mean"]
    )
    probability = 1 / (1 + np.exp(-logit))
    label_grid["y_fire"] = (rng.random(len(label_grid)) < probability).astype("int8")
    positives = label_grid.loc[label_grid["y_fire"] == 1].copy()
    positives["firms_n_pixels"] = rng.integers(1, 20, len(positives))
    positives["firms_max_confidence"] = rng.uniform(30, 100, len(positives))
    for year in range(2019, 2026):
        for month in range(1, 13):
            shard = positives.loc[
                (positives["date"].dt.year == year) & (positives["date"].dt.month == month),
                ["date", "cell_id", "firms_n_pixels", "firms_max_confidence", "y_fire"],
            ]
            atomic_parquet(shard, firms_cache_path(cfg, year, month))

    for year in range(2018, 2026):
        for month in range(1, 13):
            period = pd.Period(f"{year}-{month:02d}", freq="M")
            s2_dates = pd.date_range(
                period.start_time + pd.Timedelta(days=4), period.end_time, freq="5D"
            )
            s5p_dates = pd.date_range(period.start_time, period.end_time, freq="D")
            s2 = _synthetic_eo_rows(dem, s2_dates, "s2", rng)
            s5p = _synthetic_eo_rows(dem, s5p_dates, "s5p", rng)
            if year == 2021:
                s5p = s5p.iloc[0:0]
            atomic_parquet(s2, eo_cache_path(cfg, "s2", year, month))
            atomic_parquet(s5p, eo_cache_path(cfg, "s5p", year, month))


def _synthetic_eo_rows(
    dem: pd.DataFrame, dates: pd.DatetimeIndex, source: str, rng: np.random.Generator
) -> pd.DataFrame:
    if len(dates) == 0:
        return pd.DataFrame(columns=["cell_id", "window_end"])
    frame = pd.MultiIndex.from_product(
        [dates, dem["cell_id"]], names=["window_end", "cell_id"]
    ).to_frame(index=False)
    n = len(frame)
    if source == "s2":
        frame["s2_ndvi_mean"] = rng.uniform(0.05, 0.85, n)
        frame["s2_ndmi_mean"] = rng.uniform(-0.4, 0.6, n)
        frame["s2_nbr_mean"] = rng.uniform(-0.5, 0.8, n)
        frame["s2_evi_mean"] = rng.uniform(0, 0.8, n)
        frame["s2_valid_fraction"] = rng.uniform(0.5, 1, n)
        frame["s2_cloud_percentage"] = rng.uniform(0, 40, n)
        frame["s2_data_available"] = 1.0
    else:
        frame["s5p_aai_mean"] = rng.normal(0.5, 0.4, n)
        frame["s5p_co_mean"] = rng.normal(0.03, 0.005, n)
        frame["s5p_aai_valid_fraction"] = rng.uniform(0.6, 1, n)
        frame["s5p_co_valid_fraction"] = rng.uniform(0.6, 1, n)
        frame["s5p_data_available"] = 1.0
    return frame
