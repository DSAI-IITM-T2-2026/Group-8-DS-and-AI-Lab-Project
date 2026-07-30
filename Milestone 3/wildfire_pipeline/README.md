# LagFireNet — Live-Ready CA Wildfire Alert Pipeline

**LagFireNet** is a lag-consistent, FIRMS **1 km** dense wildfire alert model for California.

Predict fire on day **D** using features as of **D−2** (ERA5 + S5P window D−8…D−2, latest S2 window ≤ D−2, static DEM).

## Layout

```text
wildfire_pipeline/          # code package
  live/
    config.py           # AOI, lag, GCS year→bucket maps
    gcs_fetch.py        # local-first cache + GCS download
    regrid.py           # FIRMS reference grid, ERA5/DEM regrid
    labels.py           # FIRMS binary labels
    features.py         # lagged stacks + S5P LOCF/age + S2 lag
    dataset.py          # 256×256 tile dataset
    model.py            # LagFireNet
    losses.py           # Focal + Tversky
    train.py            # training entry point
    evaluate_alerts.py  # cluster alert metrics
    infer_live.py       # daily cron entry point
  data/cache/           # downloaded / linked tensors
  artifacts/            # best.pt, calibrator, threshold.json
  REPORT.md             # MultimodalFusion → LagFireNet transition report
```

## Setup

```bash
cd "Milestone 3/wildfire_pipeline"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GS_NO_SIGN_REQUEST=YES
# If personal S5P buckets need auth:
# gcloud auth application-default login
```

## S5P buckets (by year)

| Years | Bucket |
|-------|--------|
| 2019, 2021 | `gs://plated-mechanic-s5p-2016-2025/sentinel5p_features_daily/` |
| 2020, 2022 | `gs://sentinel-5p/sentinel5p_features/` |
| 2023–2025 | `gs://sentinel-2-2016-2025/sentinel5p_features_daily/` |

Local teammate caches are reused when present (`mvp_era5_dem` ERA5, multimodal fusion S2/S5P parquets).

## Train

```bash
# Smoke wiring check (few days)
python -m live.train --smoke

# Full training (train 2022–2023, val 2024, fire season May–Nov)
python -m live.train --fire-season --epochs 40 --patience 5 --min-delta 0.005
```

Artifacts written to `artifacts/`:
- `best.pt` — best val **precision @ 0.5**
- `norm_stats.npz`
- `calibrator.joblib` — isotonic on val
- `threshold.json` — smallest thr with val precision ≥ 0.4

## Alert evaluation (2025 test)

```bash
python -m live.evaluate_alerts --split test
# or subset:
python -m live.evaluate_alerts --split test --max-days 30
```

## Live / backtest inference

```bash
# Backtest one day
python -m live.infer_live --date 2025-08-15

# Cron (06:00 UTC): uses today's UTC date as label day D
python -m live.infer_live
```

Outputs: `artifacts/alerts/alerts_YYYY-MM-DD.geojson` + `.csv`.

## Model — LagFireNet

**LagFireNet**: ERA5 ConvLSTM + S5P ConvLSTM (AAI/CO/valid/age) + DEM/S2 spatial encoders → fuse → U-Net decoder. Train on 256×256 tiles; infer with 32 px overlap averaging.

See `REPORT.md` for the full MultimodalFusion → LagFireNet transition write-up.
