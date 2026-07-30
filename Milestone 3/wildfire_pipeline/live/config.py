"""Paths, GCS maps, lag policy, and model hyperparameters for the live pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
LIVE_DIR = Path(__file__).resolve().parent
PIPE_ROOT = LIVE_DIR.parent
M3_ROOT = PIPE_ROOT.parent

DATA_DIR = PIPE_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = PIPE_ROOT / "artifacts"

CACHE_FIRMS = CACHE_DIR / "firms"
CACHE_ERA5 = CACHE_DIR / "era5"
CACHE_DEM = CACHE_DIR / "dem"
CACHE_S2 = CACHE_DIR / "s2"
CACHE_S5P = CACHE_DIR / "s5p"
CACHE_GRID = CACHE_DIR / "grid"

for _d in (
    DATA_DIR,
    CACHE_DIR,
    CACHE_FIRMS,
    CACHE_ERA5,
    CACHE_DEM,
    CACHE_S2,
    CACHE_S5P,
    CACHE_GRID,
    ARTIFACTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")

# Teammate caches (local-first reuse)
MVP_ERA5_RAW = M3_ROOT / "mvp_era5_dem" / "outputs" / "cache" / "era5_raw"
MVP_ERA5_DAILY = M3_ROOT / "mvp_era5_dem" / "outputs" / "cache" / "era5_daily"
MM_S2_CACHE = M3_ROOT / "multimodal_fusion" / "outputs" / "cache" / "s2"
MM_S5P_CACHE = M3_ROOT / "multimodal_fusion" / "outputs" / "cache" / "s5p"
CA_GEOJSON = M3_ROOT / "Experiments" / "data" / "static" / "california.geojson"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------------------
# AOI / grid
# ---------------------------------------------------------------------------
AOI_BOUNDS = {
    "north": 42.01,
    "south": 32.53,
    "west": -124.41,
    "east": -114.13,
}
FIRMS_REF_DATE = "2024-08-15"  # stable day for reference grid
FIRMS_CONFIDENCE_MIN = 30

# ---------------------------------------------------------------------------
# Lag policy (predict day D using features as of D-2)
# ---------------------------------------------------------------------------
FEATURE_LAG_DAYS = 2  # newest feature day = D - 2
HISTORY_DAYS = 7  # D-8 … D-2
S5P_LOCF_MAX_DAYS = 3
S2_MAX_LAG_DAYS = 14
ERA5_MAX_MISSING_DAYS = 1
S5P_MAX_BAD_DAYS = 3  # drop/abort if > this many unrecoverable days in window

# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------
FIRMS_BUCKET = "wildfire-detection-first"
FIRMS_PREFIX = "firms_daily_geotiff"

ERA5_BUCKET = "dsai-lab-project"
ERA5_PREFIX = "wildfire_satellite/era5/raw"

DEM_BUCKET = "dsai-lab-project"
DEM_PREFIX = "wildfire_satellite/dem/2021-2025/california/terrain"
DEM_FILES = {
    "elevation": "elevation.tif",
    "slope": "slope.tif",
    "aspect": "aspect.tif",
}

# S2 numerical — year → (bucket, prefix)
S2_BUCKET_BY_YEAR: dict[int, tuple[str, str]] = {
    **{y: ("sentinel-2-2016-2025", "sentinel2_features_v3") for y in (2018, 2019, 2020)},
    **{y: ("sentinel-2-data-2016-2025", "sentinel2_features_v3") for y in (2021, 2022, 2023, 2024, 2025)},
}
S2_COLUMNS = ["NDVI_mean", "NBR_mean", "NDWI_mean"]

# S5P numerical — year → bucket (prefix differs for 2020/2022)
S5P_BUCKET_BY_YEAR: dict[int, tuple[str, str]] = {
    2019: ("plated-mechanic-s5p-2016-2025", "sentinel5p_features_daily"),
    2020: ("sentinel-5p", "sentinel5p_features"),
    2021: ("plated-mechanic-s5p-2016-2025", "sentinel5p_features_daily"),
    2022: ("sentinel-5p", "sentinel5p_features"),
    2023: ("sentinel-2-2016-2025", "sentinel5p_features_daily"),
    2024: ("sentinel-2-2016-2025", "sentinel5p_features_daily"),
    2025: ("sentinel-2-2016-2025", "sentinel5p_features_daily"),
}
S5P_COLUMNS = ["s5p_aai_mean", "s5p_co_mean"]

ERA5_DAILY_VARS = [
    "t2m",   # 2m temp → mean after daily agg
    "d2m",
    "u10",
    "v10",
    "tp",
    "swvl1",
    "blh",
]

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
# 2022–2025: all sources available on GCS (local cache reused when present)
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-11-30"  # ERA5 local through Nov 2025; Dec pullable if needed

# Prefer May–Nov for training (matches multimodal caches + fire season)
FIRE_SEASON_MONTHS = (5, 6, 7, 8, 9, 10, 11)
# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
TILE_SIZE = 256
INFER_OVERLAP = 32
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
MIN_EPOCHS = 5  # do not early-stop before this many epochs
EARLY_STOP_PATIENCE = 5  # stop after this many epochs without meaningful val gain
# Absolute gain in val precision@0.5 required to count as improvement.
# 0.5 pp absorbs pixel-level noise; smaller and patience never trips.
EARLY_STOP_MIN_DELTA = 0.005
SEED = 42

ERA5_CHANNELS = 7
S5P_CHANNELS = 4  # AAI, CO, valid, age
DEM_CHANNELS = 3
S2_CHANNELS = 4  # NDVI, NBR, NDWI, lag_days

FOCAL_ALPHA = 0.75  # weight on positive class in class-balanced focal
FOCAL_GAMMA = 2.0
# Tversky: FN weight = α, FP weight = β. Precision-first → higher β.
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
GRAD_CLIP_NORM = 1.0

DEPLOY_PRECISION_TARGET = 0.4
ALERT_HIT_RADIUS_KM = 5.0
CHECKPOINT_NAME = "best.pt"
CALIBRATOR_NAME = "calibrator.joblib"
NORM_STATS_NAME = "norm_stats.npz"
THRESHOLD_NAME = "threshold.json"
