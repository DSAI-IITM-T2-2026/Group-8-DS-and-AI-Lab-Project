# Milestone 3 — Dual-branch CNN (Sentinel-2 + ERA5/DEM)

Next-day wildfire risk per ERA5 0.25° cell:

- **Vision:** Sentinel-2 monthly mosaic patches `6×64×64`
- **Tabular:** ERA5 + DEM features from [`../mvp_era5_dem`](../mvp_era5_dem)
- **Split:** train 2018–2019 · val 2020 · test 2021 (fire season May–Nov, class-balanced)
- **Outputs:** calibrated confidence %, regional alerts, California risk maps

## Setup

```bash
cd "Milestone 3/cnn_s2_mvp"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GS_NO_SIGN_REQUEST=YES
```

## Pipeline

```bash
# 1) Tabular backbone (from mvp_era5_dem) — already done for 2018–2021
cd ../mvp_era5_dem
python build_dataset.py --start 2018-05-01 --end 2021-11-30 --fire-season

# 2) Sample + extract S2 patches (downloads each tile once, extracts all windows, frees tile)
cd ../cnn_s2_mvp
export GS_NO_SIGN_REQUEST=YES
python build_dataset.py --download-tiles
# resume-friendly: existing outputs/patches/*.npy are skipped

# 3) Train CNN
python train.py

# 4) LightGBM baseline on same rows
python train_lgbm_baseline.py

# 5) Map predictions
python map_predictions.py
```

## Outputs

```text
outputs/
  manifest.parquet
  patches/*.npy
  model/
    best.pt
    metrics.json
    test_predictions.parquet
    test_alerts_topk.csv
    lgbm_metrics.json
    comparison_metrics.json
  maps/
    risk_YYYY-MM-DD.png
    risk_YYYY-MM-DD.html
```
