"""FIRMS(D) → binary mask on the 1 km reference grid."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray as rxr

from . import config
from .gcs_fetch import ensure_firms, firms_local_path
from .regrid import build_land_mask, reference_da

logger = logging.getLogger(__name__)


def firms_to_binary(date_str: str, confidence_min: int | None = None) -> np.ndarray:
    confidence_min = config.FIRMS_CONFIDENCE_MIN if confidence_min is None else confidence_min
    path = ensure_firms(date_str)
    da = rxr.open_rasterio(path, masked=True)
    n_band = int(da.sizes.get("band", da.shape[0]))
    if n_band < 3:
        raise ValueError(f"FIRMS {date_str}: expected >=3 bands, got {n_band}")
    if n_band > 3:
        da = da.isel(band=slice(0, 3))
    da = da.assign_coords(band=("band", ["confidence", "brightness_temp_K", "detection_flag"]))

    ref = reference_da()
    da = da.rio.write_crs("EPSG:4326")
    matched = da.rio.reproject_match(ref, resampling=1)  # nearest
    confidence = np.asarray(matched.sel(band="confidence").values, dtype=np.float32)
    detection = np.asarray(matched.sel(band="detection_flag").values, dtype=np.float32)
    binary = (~np.isnan(detection)).astype(np.uint8)
    low = confidence < confidence_min
    low = np.where(np.isnan(confidence), False, low)
    binary = np.where(low, 0, binary).astype(np.uint8)

    land = build_land_mask(ref)
    binary = np.where(land, binary, 0).astype(np.uint8)
    return binary


def label_cache_path(date_str: str) -> Path:
    return config.CACHE_FIRMS / "labels" / f"{date_str}.npy"


def load_or_build_label(date_str: str) -> np.ndarray:
    path = label_cache_path(date_str)
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lab = firms_to_binary(date_str)
    np.save(path, lab)
    return lab
