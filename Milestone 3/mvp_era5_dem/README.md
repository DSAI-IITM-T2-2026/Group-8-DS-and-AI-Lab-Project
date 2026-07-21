# Milestone 3 — MVP next-day wildfire risk (ERA5 + DEM → LightGBM)

Copied from Milestone 2 data-pipeline work. Efficient tabular path for proactive wildfire prediction over California:

- **Unit:** ERA5 0.25° cell × day  
- **Task:** predict fire on day `D+1` from weather/terrain through day `D`  
- **Output:** calibrated **confidence %** + **region** (cell / lat-lon)  
- **Labels:** FIRMS pixels with confidence ≥ 30, aggregated to cells  

Skips Landsat / Sentinel for speed. Upgrade later by joining S2/S5P onto the same `cell_id`.

## Setup

```bash
cd "Milestone 3/mvp_era5_dem"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GS_NO_SIGN_REQUEST=YES
```

## Build dataset

Smoke (one fire-season month):

```bash
python build_dataset.py --smoke
```

Full MVP window (config default 2021–2025):

```bash
python build_dataset.py
# or
python build_dataset.py --start 2021-01-01 --end 2025-12-31
```

Writes:

```text
outputs/
  train.parquet
  val.parquet
  test.parquet
  metadata/feature_columns.json
  metadata/dataset_metadata.json
  cache/   # ERA5 monthly + daily, FIRMS cell labels (reusable)
```

## Train + alerts

```bash
python train_baseline.py
```

Writes:

```text
outputs/model/
  baseline.joblib          # booster + calibrator + feature list
  metrics.json             # PR-AUC / ROC-AUC
  test_alerts_topk.csv     # region + confidence_pct per day
```

Each alert row: `label_date`, `region`, `cell_id`, `latitude`, `longitude`, `confidence_pct`, `y_fire`.
