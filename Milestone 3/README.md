# Milestone 3 — Modeling

Next-day California wildfire risk models. Raw satellite/weather data stays on GCS; this folder is **code + small DEM grid features** only. Rebuild datasets locally.

## Projects

| Folder | Model | Split |
|--------|--------|-------|
| [`mvp_era5_dem/`](mvp_era5_dem/) | LightGBM on ERA5 + DEM tabular features | configurable (default 2022–2025) |
| [`cnn_s2_mvp/`](cnn_s2_mvp/) | Dual-branch CNN (S2) + MLP (ERA5/DEM) | was 2018–2021 in first run |
| [`cnn_lstm_fusion/`](cnn_lstm_fusion/) | CNN + LSTM + optional S5P scalar | train 2022–2023 / val 2024 / test 2025 |
| [`multimodal_fusion/`](multimodal_fusion/) | **Full hybrid:** S2 CNN + S5P CNN + LSTM + S2/S5P numerical MLPs | train 2022–2023 / val 2024 / test 2025 |

**Released weights:** [`cnn_lstm_fusion/artifacts/`](cnn_lstm_fusion/artifacts/) · [`multimodal_fusion/artifacts/`](multimodal_fusion/artifacts/)  
**Architecture / preprocessing:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (includes §11 multimodal).

## Reproduce full multimodal hybrid

```bash
export GS_NO_SIGN_REQUEST=YES
cd mvp_era5_dem && python build_dataset.py --start 2022-05-01 --end 2025-11-30 --fire-season
cd ../multimodal_fusion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_dataset.py --download-tiles
python build_s5p_patches.py --download-tiles
python build_sequences.py
python build_numerical_features.py   # may need gcloud ADC for numerical buckets
python train.py
python map_predictions.py
```

Or load published weights from `artifacts/multimodal_full_2022_2025/` (see that folder’s README).

## Reproduce CNN + LSTM (+ optional S5P)

Requires: Python **3.11 or 3.12** (not 3.14), `gsutil` optional, network access to public GCS buckets, ~several GB disk for caches/patches.

```bash
export GS_NO_SIGN_REQUEST=YES

# 1) Tabular backbone
cd mvp_era5_dem
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_dataset.py --start 2022-05-01 --end 2025-11-30 --fire-season

# 2) Fusion model
cd ../cnn_lstm_fusion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GS_NO_SIGN_REQUEST=YES

python build_dataset.py --download-tiles   # S2 patches
python build_sequences.py                  # ERA5/DEM [7, F]
# optional:
# python build_s5p_features.py --download-tiles

python train.py --use-sentinel5p           # or --no-sentinel5p
python map_predictions.py
```

GCS sources used:

- ERA5: `gs://dsai-lab-project/wildfire_satellite/era5/raw/`
- Sentinel-2: `gs://dsai-lab-project/wildfire_satellite/raw/sentinel2/`
- Sentinel-5P: `gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/`
- FIRMS: `gs://wildfire-detection-first/firms_daily_geotiff/`
- DEM cells: shipped in `mvp_era5_dem/data/era5_grid_dem_features.parquet`

**Not in git:** `outputs/` (patches, sequences, caches).  

**In git (for teammates):** `artifacts/cnn_lstm_s5p_2022_2025/` — `best.pt`, calibrator, norm stats, metrics (~620 KB).

See each project’s `README.md` and the full lifecycle diagrams in [`ARCHITECTURE.md`](ARCHITECTURE.md).
