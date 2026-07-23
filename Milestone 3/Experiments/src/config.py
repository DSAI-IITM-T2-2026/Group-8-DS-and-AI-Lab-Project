"""Milestone 3 configuration — local Mac paths, GCS buckets, 30-channel feature set."""
from __future__ import annotations

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths (relative to Milestone 3/ root — never hardcode user home paths)
# ---------------------------------------------------------------------------
M3_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = M3_ROOT / "data"
TMP_DIR = DATA_DIR / "tmp"
CACHE_DIR = DATA_DIR / "cache"  # durable local mirror (skip-if-exists on rebuild)
OUTPUT_DIR = DATA_DIR / "processed"
METADATA_DIR = OUTPUT_DIR / "metadata"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# Per-source disk caches (teammate Milestone 3 pattern: download once, rebuild offline-friendly)
CACHE_S2_DIR = CACHE_DIR / "s2_csv"
CACHE_S5P_DIR = CACHE_DIR / "s5p_csv"
CACHE_ERA5_DIR = CACHE_DIR / "era5"
CACHE_FIRMS_DIR = CACHE_DIR / "firms"
CACHE_DEM_DIR = CACHE_DIR / "dem"

# When False: stream from GCS only (no new writes under data/cache/).
# Existing DEM files may still be read to avoid re-fetching huge TIFFs.
USE_DISK_CACHE = os.environ.get("M3_USE_DISK_CACHE", "1").strip() not in ("0", "false", "False", "no")

for _d in (
    TMP_DIR,
    CACHE_DIR,
    CACHE_S2_DIR,
    CACHE_S5P_DIR,
    CACHE_ERA5_DIR,
    CACHE_FIRMS_DIR,
    CACHE_DEM_DIR,
    OUTPUT_DIR,
    METADATA_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Must be set before any rasterio/GDAL import in callers
os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")

# ---------------------------------------------------------------------------
# Device — Apple Silicon MPS (never CUDA on this machine)
# ---------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MPS_FALLBACKS: list[str] = []

# ---------------------------------------------------------------------------
# GCS buckets (anonymous access)
# ---------------------------------------------------------------------------
# See config.py — S2/S5P now use separate feature-CSV buckets:
#   sentinel-2-data-2016-2025/sentinel2_features_v3/
#   sentinel-2-2016-2025/sentinel5p_features_daily/
# ERA5 + DEM remain on dsai-lab-project; FIRMS on wildfire-detection-first.
FIRMS_BUCKET = "wildfire-detection-first"
FIRMS_PREFIX = "firms_daily_geotiff"

# Legacy multi-source bucket (ERA5 + DEM still live here)
GCS_BUCKET = "dsai-lab-project"
GCS_PREFIX = "wildfire_satellite"
PREFIX_ERA5 = f"{GCS_PREFIX}/era5/raw"
PREFIX_DEM = f"{GCS_PREFIX}/dem/2021-2025/california"
PREFIX_DEM_TERRAIN = f"{PREFIX_DEM}/terrain"

# New tabular feature stores (Hive partitions: year=/month=/window=/features.csv)
# Sentinel-2 — ~5-day windows, precomputed indices per CA grid cell (~413k rows)
S2_PROJECT_ID = "iitm-bs-mlops-500106"
S2_BUCKET = "sentinel-2-data-2016-2025"
S2_PREFIX = "sentinel2_features_v3"  # years present here: 2021–2025
# Future: S2 2019–2020 also exists in a separate GCS bucket (path TBD) — not needed for M3 2025 subset

# Sentinel-5P — daily windows, AAI + CO features (same grid_id / lat-lon)
S5P_PROJECT_ID = "project-815eaf36-ef08-42e9-963"
S5P_BUCKET = "sentinel-2-2016-2025"  # bucket name is shared; prefix is S5P
S5P_PREFIX = "sentinel5p_features_daily"

# Back-compat aliases used by older GeoTIFF loaders (legacy dsai-lab-project paths)
PREFIX_S2 = f"{GCS_PREFIX}/raw/sentinel2"
PREFIX_S5P = f"{GCS_PREFIX}/raw/sentinel5p"

# Expected schema highlights for new CSV features
S2_CSV_KEY_COLUMNS = [
    "grid_id", "latitude", "longitude", "window_start", "window_end",
    "NDVI_mean", "NBR_mean", "NDWI_mean", "s2_data_available",
]
S5P_CSV_KEY_COLUMNS = [
    "grid_id", "latitude", "longitude", "window_start", "window_end",
    "s5p_aai_mean", "s5p_co_mean", "s5p_data_available",
]

DEM_TERRAIN_FILES = {
    "elevation_m": "elevation.tif",
    "slope_deg": "slope.tif",
    "aspect_deg": "aspect.tif",
    "hillshade": "hillshade.tif",
    "tpi": "tpi.tif",
    "tri": "tri.tif",
}

ERA5_EXPECTED_VARIABLES = [
    "t2m", "d2m", "sp", "u10", "v10", "i10fg", "tp",
    "swvl1", "swvl2", "cvh", "cvl", "lai_hv", "lai_lv", "blh",
]

# Provisional AOI (M2 intended); replaced by intersection in aoi_bounds.json
DEFAULT_AOI_BOUNDS = {
    "north": 42.01,
    "south": 32.53,
    "west": -124.41,
    "east": -114.13,
}

# ---------------------------------------------------------------------------
# Feature channels — Landsat dropped → 30 channels
# ---------------------------------------------------------------------------
FEATURE_CHANNEL_NAMES = [
    "S2_NDVI", "S2_NBR", "S2_NDWI",
    "S5P_aerosol_index",
    "era5_t2m_max_C", "era5_t2m_min_C", "era5_t2m_mean_C", "era5_d2m_mean_C",
    "era5_sp_mean", "era5_wind_speed_mean", "era5_wind_dir_sin", "era5_wind_dir_cos",
    "era5_wind_gust_max", "era5_precip_sum_mm", "era5_soil_water_l1_mean",
    "era5_soil_water_l2_mean", "era5_high_veg_cover", "era5_low_veg_cover",
    "era5_lai_high_veg", "era5_lai_low_veg", "era5_blh_mean", "era5_blh_max", "era5_vpd",
    "dem_elevation_m", "dem_slope_deg", "dem_aspect_deg",
    "dem_hillshade", "dem_tpi", "dem_tri",
    "firms_prev_day_fire",
]
assert len(FEATURE_CHANNEL_NAMES) == 30, len(FEATURE_CHANNEL_NAMES)

# ---------------------------------------------------------------------------
# Dataset / sampling hyperparameters (judgment calls documented in reports/)
# ---------------------------------------------------------------------------
HISTORY_DAYS = 7
PATCH_SIZE = 64
FIRE_CONFIDENCE_THRESHOLD = 30
FIRE_CLUSTER_DILATE_PX = 8
MAX_FIRE_CLUSTERS_PER_DAY = 3
BACKGROUND_MIN_DISTANCE_PX = 32
CANDIDATE_YEAR = 2025  # M3 subset year (was 2024; S5P CSV download still in progress)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42

# Training defaults
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 3e-4
DEFAULT_HIDDEN = 32
DEFAULT_EPOCHS = 30
EARLY_STOP_PATIENCE = 7
