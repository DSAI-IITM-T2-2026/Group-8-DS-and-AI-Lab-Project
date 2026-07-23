"""Patch-level fusion — crop before stack (memory-safe), 30 channels, no Landsat."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src import config


def feature_lookup_fn_patch(
    date_str: str,
    top: int,
    left: int,
    patch_size: int,
    *,
    s2_cache,
    era5_cache,
    s5p_cache,
    firms_label_cache,
    dem_features_on_firms_grid: dict,
) -> np.ndarray:
    """
    Same fusion logic as M2 feature_lookup_fn_patch, without Landsat.
    Crops each source BEFORE stacking so full-frame tensors are never built.
    """
    s2 = s2_cache.get_most_recent(date_str)
    if s2 is None:
        raise ValueError(f"No Sentinel-2 scene available at or before {date_str}")
    era5_daily = era5_cache.get_daily(date_str)
    s5p_daily = s5p_cache.get_daily(date_str)

    prev_date = (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prev_fire = firms_label_cache.get(prev_date).astype(np.float32)

    def crop(arr):
        return arr[top : top + patch_size, left : left + patch_size]

    channel_dict = {
        "S2_NDVI": crop(s2["NDVI_S2"]),
        "S2_NBR": crop(s2["NBR_S2"]),
        "S2_NDWI": crop(s2["NDWI_S2"]),
        "S5P_aerosol_index": crop(s5p_daily["aerosol_index"]),
        "firms_prev_day_fire": crop(prev_fire),
    }
    for k, v in era5_daily.items():
        channel_dict[f"era5_{k}"] = crop(v)
    for k, v in dem_features_on_firms_grid.items():
        channel_dict[f"dem_{k}"] = crop(v)

    stacked = np.stack(
        [channel_dict[name] for name in config.FEATURE_CHANNEL_NAMES], axis=-1
    )
    return stacked.astype(np.float32)


def label_lookup_fn_patch(
    date_str: str,
    top: int,
    left: int,
    patch_size: int,
    *,
    firms_label_cache,
) -> np.ndarray:
    label = firms_label_cache.get(date_str)
    cropped = label[top : top + patch_size, left : left + patch_size]
    return cropped[..., np.newaxis].astype(np.float32)
