# Group-8-DS-and-AI-Lab-Project

AI-Powered Wildfire Early Detection and Alerting System — DSAI Lab Project (IITM).

## Milestones

### Milestone 1

- Project report and work log: [`Milestone 1/`](Milestone%201/)

### Milestone 2

- Report: [`Milestone 2/Report.md`](Milestone%202/Report.md) (figures in [`report-images/`](Milestone%202/report-images/))
- Work log: [`Milestone 2/Work Log.md`](Milestone%202/Work%20Log.md)

### Milestone 3

Modeling code (datasets rebuild from GCS — large `outputs/` are gitignored):

- Overview + reproduce steps: [`Milestone 3/README.md`](Milestone%203/README.md)
- **Full ML lifecycle architecture (diagrams):** [`Milestone 3/ARCHITECTURE.md`](Milestone%203/ARCHITECTURE.md)
- **Full multimodal hybrid** (S2 CNN + S5P CNN + LSTM + numerical MLPs): [`multimodal_fusion/`](Milestone%203/multimodal_fusion/)  
  Split: train 2022–2023 / val 2024 / test 2025 · released weights: [`artifacts/`](Milestone%203/multimodal_fusion/artifacts/)
- **CNN + LSTM (+ optional S5P scalar):** [`cnn_lstm_fusion/`](Milestone%203/cnn_lstm_fusion/) · weights: [`artifacts/`](Milestone%203/cnn_lstm_fusion/artifacts/)
- Dual-branch CNN baseline: [`cnn_s2_mvp/`](Milestone%203/cnn_s2_mvp/)
- Tabular ERA5 + DEM backbone: [`mvp_era5_dem/`](Milestone%203/mvp_era5_dem/)

### Milestone 2 — data pipelines

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
