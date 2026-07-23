# Group-8-DS-and-AI-Lab-Project

**AI-Powered Wildfire Early Detection and Alerting System**  
DSAI Lab Project · IIT Madras · Group 8

---

## Milestones

### Milestone 1

| Document | Path |
|----------|------|
| Report | [`Milestone 1/Report.md`](Milestone%201/Report.md) |
| Work log | [`Milestone 1/Work Log.md`](Milestone%201/Work%20Log.md) |

---

### Milestone 2 — Dataset preparation

| Document | Path |
|----------|------|
| Report | [`Milestone 2/Report.md`](Milestone%202/Report.md) |
| Work log | [`Milestone 2/Work Log.md`](Milestone%202/Work%20Log.md) |
| Figures | [`Milestone 2/report-images/`](Milestone%202/report-images/) |

#### Data pipelines

- **Data Pipeline — Copernicus DEM GLO-30:** [`Milestone 2/data-pipeline/copernicus-dem-30m/`](Milestone%202/data-pipeline/copernicus-dem-30m/)
  - Code for acquiring and processing GLO-30 terrain data (California)
  - Processed data on GCS: `gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/`

- **EDA — Copernicus DEM (California):** [`Milestone 2/data-pipeline/copernicus-dem-eda/`](Milestone%202/data-pipeline/copernicus-dem-eda/)
  - California-polygon clip, numerical EDA, and ERA5-grid fusion notebooks/scripts
  - Clipped GeoTIFFs on GCS: `gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/eda/clipped_ca/`

- **Data Pipeline — ERA5:** [`Milestone 2/data-pipeline/ERA5/`](Milestone%202/data-pipeline/ERA5/)
  - Code for acquiring and processing ECMWF ERA5 reanalysis weather data (California)
  - Raw data on GCS: `gs://dsai-lab-project/wildfire_satellite/era5/raw/`

- **Data Pipeline — FIRMS:** [`Milestone 2/data-pipeline/FIRMS/`](Milestone%202/data-pipeline/FIRMS/)
  - Code for acquiring and processing FIRMS fire label data (California)
  - Raw data on GCS: `gs://wildfire-detection-first/firms_daily_geotiff/`

#### Milestone 2 — sample figures

From the [Dataset Preparation Report](Milestone%202/Report.md):

| FIRMS bands | Data coverage | DEM terrain layers |
|-------------|---------------|--------------------|
| ![FIRMS confidence / brightness / detection](Milestone%202/report-images/figure-01.png) | ![Confirmed year-by-year coverage](Milestone%202/report-images/figure-08.png) | ![Copernicus DEM terrain layers](Milestone%202/report-images/figure-07.png) |

| ERA5 correlations | NDVI fire vs no-fire | Fire density |
|-------------------|----------------------|--------------|
| ![ERA5 variable correlation matrix](Milestone%202/report-images/figure-06.png) | ![NDVI fire vs no-fire](Milestone%202/report-images/figure-10.png) | ![FIRMS spatial density](Milestone%202/report-images/figure-09.png) |

See all 14 figures in [`Milestone 2/report-images/`](Milestone%202/report-images/) and the full write-up in [`Report.md`](Milestone%202/Report.md).

---

### Milestone 3 — Modeling

Next-day California wildfire risk models. Rebuild datasets from GCS; large `outputs/` / `.npy` / checkpoints are gitignored. Small released weights live under `artifacts/`.

| Document | Path |
|----------|------|
| Overview / reproduce | [`Milestone 3/README.md`](Milestone%203/README.md) |
| Architecture (lifecycle diagrams) | [`Milestone 3/ARCHITECTURE.md`](Milestone%203/ARCHITECTURE.md) |
| Final report | [`Milestone 3/Report.md`](Milestone%203/Report.md) |
| Work log | [`Milestone 3/Work Log.md`](Milestone%203/Work%20Log.md) |

#### Model projects

| Project | Description | Notes |
|---------|-------------|-------|
| [`mvp_era5_dem/`](Milestone%203/mvp_era5_dem/) | LightGBM tabular baseline (ERA5 + DEM) | Configurable split (default 2022–2025) |
| [`cnn_s2_mvp/`](Milestone%203/cnn_s2_mvp/) | Dual-branch S2 CNN + weather MLP | Optical + tabular MVP |
| [`cnn_lstm_fusion/`](Milestone%203/cnn_lstm_fusion/) | S2 CNN + 7-day LSTM ± S5P scalar | Train 2022–23 / val 2024 / test 2025 · [`artifacts/`](Milestone%203/cnn_lstm_fusion/artifacts/) |
| [`multimodal_fusion/`](Milestone%203/multimodal_fusion/) | **Full hybrid:** S2/S5P CNNs + LSTM + numerical MLPs | Same split · [`artifacts/`](Milestone%203/multimodal_fusion/artifacts/) · [detail report](Milestone%203/multimodal_fusion/REPORT.md) |
| [`Experiments/`](Milestone%203/Experiments/) | Patch segmentation (ConvLSTM+U-Net, HistGB, ablations) on FIRMS ~1 km · local MPS | Full-year 2025 candidate · sibling task (not 1:1 with cell-day metrics) |

Anonymous GCS: `export GS_NO_SIGN_REQUEST=YES`.

---

## Team

| Member | Roll Number |
|--------|-------------|
| Ripunjay Kumar | 21F3002511 |
| Lakshay Garg | 21F3001076 |
| Roushan Kumar Singh | 23F1002240 |
| Lakshmi Sruthi K | 21F1005626 |
| R Aditya | 21F1004839 |
