# Milestone 3 — Experiments (patch segmentation stack)

Local-first **patch-level** wildfire segmentation pipeline (FIRMS ~1 km, 30 fused channels, ConvLSTM+U-Net). Lives under `Milestone 3/Experiments/` as a sibling to `multimodal_fusion/` (cell-day / monthly image CNNs — different task encoding).

Ports the verified M2 design to Apple Silicon (**MPS**), drops Landsat (**30 channels**), and trains on a fire-centered **2025** candidate subset.

## Current status

| Item | Status |
|---|---|
| Location | `Milestone 3/Experiments/` |
| Candidate year | **2025** full-year candidate (`2025-01-01`→`11-30`, 1035 patches) |
| Local `.npy` | Ready for offline train (no GCS needed to retrain) |
| Models | HistGB baseline · ConvLSTM+U-Net · U-Net last-day · focal / BCE+Dice |

## Multi-model experiments (offline)

```bash
cd "Milestone 3/Experiments"
source .venv/bin/activate

# Full pack A–D → metadata/experiment_comparison.json + confusion matrices
python scripts/run_experiments.py --epochs 15

# Or single model:
python scripts/train_models.py --model baseline
python scripts/train_models.py --model convlstm --loss focal --epochs 15
python scripts/train_models.py --model unet_last_day --loss bce_dice --epochs 15
python scripts/train_models.py --model convlstm --tune --epochs 15 --trials 8
```

`--model` ∈ `{baseline, convlstm, unet_last_day, all}` · `--loss` ∈ `{bce_dice, focal}`

Artifacts: `data/processed/metadata/eval_*.json`, `figures/confusion/cm_*.png`, `experiment_comparison.json`.

## Layout

```
Milestone 3/Experiments/
  src/           # config, data, models, training
  scripts/       # CLI entrypoints (train_models, run_experiments, …)
  notebooks/
  data/          # tmp + processed (gitignored)
  reports/       # architecture diagram, coverage map
  docs/          # Report, Work Log, Presentation_Outline, guides
  README.md
  requirements.txt
```

## Setup (venv)

```bash
cd "Milestone 3/Experiments"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `rasterio` / GDAL fails on Apple Silicon, prefer conda-forge:

```bash
conda create -n wildfire-m3 python=3.11
conda activate wildfire-m3
conda install -c conda-forge rasterio gdal rioxarray
pip install -r requirements.txt
```

## Smoke test (MPS + GCS)

```bash
python scripts/smoke_test.py
```

Expects: `DEVICE = mps` (or `cpu`), anonymous GCS listing, and a FIRMS GeoTIFF open.

## Data sources (anonymous GCS)

| Source | Bucket / prefix | Format | Notes |
|---|---|---|---|
| FIRMS | `wildfire-detection-first/firms_daily_geotiff/` | daily GeoTIFF | Labels + `firms_prev_day_fire`; **~1 km/pixel** reference grid |
| Sentinel-2 | `sentinel-2-data-2016-2025/sentinel2_features_v3/` | `year=/month=/window=/features.csv` | ~5-day windows; ~413k CA cells; NDVI/NBR/NDWI/… |
| Sentinel-5P | `sentinel-2-2016-2025/sentinel5p_features_daily/` | same Hive CSV layout | Daily; AAI + CO; same `grid_id` as S2 |
| ERA5 | `dsai-lab-project/wildfire_satellite/era5/raw/{year}/` | monthly ZIP-as-`.nc` (hourly inside) | 2016–2025 CA bbox → daily agg → regrid to FIRMS |
| DEM | `dsai-lab-project/.../terrain/*.tif` | 6 static GeoTIFFs | elevation, slope, aspect, hillshade, tpi, tri |

S2/S5P CSVs are **coordinate tables** (`grid_id`, lat, lon). FIRMS/ERA5/DEM are **rasters**; everything is aligned to the FIRMS grid for 64×64 patches (~**64 km** on a side, not 1 km).

Locked raster AOI (ERA5-bounded): `W −124.25, S 32.75, E −114.25, N 42.0`.  
Architecture: [`reports/architecture_diagram.md`](reports/architecture_diagram.md).  
Full runbook (folders, outputs, report flow): [`docs/COMPLETE_PIPELINE_GUIDE.md`](docs/COMPLETE_PIPELINE_GUIDE.md).  
Team status: [`docs/M3_STATUS_AND_GAP.md`](docs/M3_STATUS_AND_GAP.md).

## Data verification + coverage map

```bash
python scripts/verify_gcs_data.py --year 2025
open reports/coverage_map/index.html
```

Team gap list: `data/processed/metadata/missing_for_team.md`.

## Pipeline

```bash
# 0. Preflight disk space (no download) — run BEFORE a long build
python scripts/estimate_disk.py --year 2025 --start 2025-06-01 --end 2025-11-30 --require-free

# 1. Lock AOI (raster intersection) — already produced aoi_bounds.json
python scripts/lock_aoi.py

# 2. Build fire-centered 2025 candidate dataset (CSV S2/S5P)
python scripts/build_dataset.py --year 2025 --start 2025-06-01 --end 2025-11-30

# Fast iteration (smoke slice):
python scripts/build_dataset.py --year 2025 --max-days 15 --start 2025-07-01 --end 2025-07-31

# 3. Baseline + DL models (+ optional tuning + pred maps)
python scripts/run_experiments.py --epochs 15   # A–D comparison
python scripts/train_models.py --model all --loss bce_dice
python scripts/train_models.py --model convlstm --tune --epochs 15 --trials 8
python scripts/map_predictions.py --n 6
# Statewide CA risk map (teammate-style Confidence % + FIRMS rings):
python scripts/map_state_risk.py
# → data/processed/figures/maps/risk_*.png
```

Training reads local `.npy` only (no internet). Rebuilds reuse `data/cache/` when present.

Or open `notebooks/m3_pipeline.ipynb` and run section-by-section.

## Auth / paths

- GCS: anonymous via `gcsfs` + `GS_NO_SIGN_REQUEST=YES` (set before rasterio)
- Outputs: `./data/processed/` (never `/kaggle/working`)
- Device: `torch.device("mps" if … else "cpu")` — never CUDA

## Feature channels (30)

Sentinel-2 indices (3) + S5P aerosol (1) + ERA5 daily (19) + DEM (6) + `firms_prev_day_fire` (1). Landsat dropped.

## Models

- **Baseline:** HistGradientBoosting on last-day per-pixel features (30 cols; XGBoost blocked by OpenMP crash on macOS)
- **Primary:** ConvLSTM encoder + U-Net decoder → 64×64 fire probability map from 7-day history
- **Ablation:** U-Net last-day only (`unet_last_day`) — no temporal ConvLSTM
- **Losses:** `bce_dice` (default) · `focal`
- **Metrics:** precision, recall, F1, Dice, AUC-PR + confusion matrix / classification report (not accuracy as headline)
