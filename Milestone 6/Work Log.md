# Work Logs - Milestone 6 (Cumulative Project Work Log)

This document aggregates all work logs across **Milestone 1 through Milestone 6** for Group 8.

---

## Ripunjay Kumar (Roll Number: 21F3002511)

### Milestone 1
- Conducted a comprehensive literature review on existing deep learning-based wildfire detection models, platforms, and public satellite datasets (*Sentinel-2, Landsat-8/9, MODIS, VIIRS*).
- Identified critical research gaps in current academic and commercial solutions, specifically the over-reliance on post-ignition detection, single-source imagery, and the lack of temporal context.
- Defined the project's technical evaluation methodology, establishing classical/deep learning baselines, industry benchmarks (*NASA FIRMS, Copernicus EMS*), and a primary metric strategy prioritizing Accuracy and Recall.
- Synthesized the project report into a structured academic presentation layout designed for stakeholder communication.
- Engineered the narrative flow and technical storytelling of the pitch deck, ensuring a clear progression from problem statement to the proposed multimodal AI framework.

### Milestone 2
- **Copernicus DEM GLO-30 Pipeline**: Designed and implemented an end-to-end Copernicus DEM GLO-30 pipeline for the California study area (N 42.01°, S 32.53°, W −124.41°, E −114.13°). Built tile identification, parallel download, validation, merge, and study-area clip stages to produce a seamless elevation mosaic from ~100 original 1°×1° tiles. Derived ML-ready terrain features: elevation, slope, aspect, hillshade, TRI, and TPI. Published processed DEM outputs to Google Cloud Storage (`gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/`).
- **ERA5 Reanalysis Pipeline**: Designed and implemented an ECMWF ERA5 pipeline for the California bounding box, pulling hourly weather variables for 2016–2025. Configured multi-account CDS downloads with work partitioning, checkpointing, retries, and queue-limit handling. Processed 2m temperature/dewpoint, surface pressure, 10m wind components/gusts, precipitation, soil water, vegetation cover/LAI, and boundary layer height. Published NetCDF outputs to GCS (`gs://dsai-lab-project/wildfire_satellite/era5/raw/`).

### Milestone 3
- Designed and documented the Milestone 3 ML lifecycle (`ARCHITECTURE.md`): next-day cell-day fire risk, ERA5 0.25° grid, FIRMS labels, train/val/test protocol.
- Built the tabular baseline pipeline `mvp_era5_dem/` (ERA5 + DEM → LightGBM).
- Implemented progressive fusion models: `cnn_s2_mvp/`, `cnn_lstm_fusion/` (CNN + LSTM ± S5P), and full hybrid `multimodal_fusion/` (S2/S5P CNNs + LSTM + numerical MLPs).
- Released checkpoints and metrics under `cnn_lstm_fusion/artifacts/` and `multimodal_fusion/artifacts/`; added map/prediction scripts and project READMEs.

### Milestone 4
- Designed and implemented **LagFireNet**, a live-ready California wildfire alert model under `Milestone 3/wildfire_pipeline/` (predicting day D from features as of D−2).
- Built the full lag-consistent data path: FIRMS reference grid + land mask, ERA5 daily regrid, S5P AAI/CO rasterize with LOCF/valid/age, S2 numerical indices + lag channel, static DEM; local-first GCS fetch with year-to-bucket maps.
- Implemented `LagFireNet` (ERA5/S5P ConvLSTM + DEM/S2 encoders → fuse → U-Net), Focal+Tversky loss, tile dataset (256×256, fire oversample), training with early stopping on validation precision@0.5, isotonic calibration, and deploy thresholding (precision ≥ 0.4).
- Added daily inference (`infer_live.py`) and cluster alert evaluation (`evaluate_alerts.py`); documented the transition from cell risk to 1 km lag-consistent alerts in `REPORT.md`.

### Milestone 5
- Executed Optuna hyperparameter fine-tuning for the champion LightGBM dual-head pipeline (classifier search on 2019–2022 → 2023 validation, blending PR-AUC and Recall@25).
- Performed SHAP / TreeSHAP feature importance analysis on the fitted classifier and ranked drivers for interpretability.
- Selected top ~80% features by SHAP contribution, pruned bottom ~20%, re-trained classifier + ranker on the pruned set, and re-tuned the daily alert blend on 2023.
- Extended training notebooks with post-training diagnostics (slice analysis, Top-25 confusion maps, FP/FN sample generation) for LightGBM and CatBoost evaluation runs.
- Contributed to the Milestone 5 report write-up and presentation slides.

### Milestone 6
- Co-authored the final **Technical Report** and presentation materials, consolidating all system architectures, quantitative benchmarks, and spatial-temporal slice evaluations.
- Verified final model export scripts (`champion_model.joblib`) and deployment configurations for the daily operational pipeline.

---

## Lakshay Garg (Roll Number: 21F3001076)

### Milestone 1
- Authored the Problem Statement and subtopics — core problem, scope, stakeholders, limitations, and objectives — framing the project around predicting wildfire risk *before* ignition (24–48h lead time).
- Defined project scope and limitations, identified the full stakeholder set, and anchored motivation in a real Indian case study (April 2021 Uttarakhand forest fire / NDMA incident report).
- Formulated core objectives and evaluation-metric framing, ensuring alignment with the problem statement.

### Milestone 2
- **Unified EDA & Preprocessing (Kaggle Notebook)**: Built the consolidated Milestone 2 notebook that loads and validates all six data sources (FIRMS, Landsat-8, Sentinel-2, Sentinel-5P, ERA5, Copernicus DEM) from GCS.
- Implemented the full EDA suite — class imbalance, fire-event lifecycle validation, coverage timeline, and spatial clustering of fire detections.
- Validated the NDVI/LST-vs-fire-label relationship on fused data, including a land-cover fairness correction.
- Designed the spatial/temporal fusion strategy (reference grid choice, forward-fill for slow-revisit sources, correct ERA5 daily aggregation).

### Milestone 3
- Built the local MPS patch-segmentation stack under `Milestone 3/Experiments/` (30 fused channels, fire-centered 64×64 patches).
- Completed full-year 2025 candidate build (1,035 patches) and trained HistGB + ConvLSTM + U-Net architectures.
- Added multi-model experiments (baseline, ConvLSTM BCE+Dice/Focal, U-Net last-day), confusion matrices, and `run_experiments.py`.

### Milestone 4
- Built the Milestone 4 numerical next-day pipeline under `Milestone 4/numerical_nextday/` (ERA5 lag-5 primary, causal S2/S5P attach, Stage A/B/C tables for 2019–2025).
- Assembled train/val/test splits (2019–2022 / 2023–2024 / 2025) and trained LightGBM fire-season + seasonal models, Stage C HP sweep, and MLP secondary models.
- Generated evaluation artifacts (`experiments_log.csv`, `eval_metrics.json`, figures, sample alerts) and populated report findings.

### Milestone 5
- Consolidated Stage C / Stage C KNN datasets on Kaggle (`California_Wildfire_KNN`, `California_Wildfire_Median`, `firms_test` / `fire_analysis2.csv`) for reproducible training and inference inputs.
- Fixed the end-to-end training notebook (`Wildfire_Training_v2.ipynb` / `Wildfire_Training_final.ipynb`): Kaggle path configuration, train/val splits, Optuna tuning, validation metrics, and artifact export (`champion_model.joblib`).
- Fixed the inference notebook (`Wildfire_Inference.ipynb`) to load training notebook outputs and score 2025 via `test.parquet` with the exact feature contract.
- Tested pipeline performance across fire-region cell subsets: high-fire cells, high + medium cells, and all cells.

### Milestone 6
- Structured and validated Kaggle dataset publishing and environment configurations for 100% reproducible execution of the champion model pipeline.
- Contributed to final non-technical and technical documentation verification.
- Built the end-to-end daily wildfire data pipeline (download → Stage C features → 86-feature parquet export → GCS), including causal next-day labeling, GCS reuse for 2019–2025.
- With Aditya, set up the Cloud Run cron job on GCP and deployed the Wildfire IQ application on a GCE VM with Docker Compose (backend + frontend), verified Generate/prediction for live dates, and confirmed cron EE exports against GCS.

---

## Roushan Kumar Singh (Roll Number: 23F1002240)

### Milestone 1
- Conducted a thorough review of past works in wildfire prediction, analyzing methodologies employed across remote sensing and machine learning literature.
- Documented critical research gaps and extracted actionable insights to guide team methodology.

### Milestone 2
- **Sentinel-2**: Built a data fetch pipeline for Sentinel-2 imagery over the California target AOI using Google Earth Engine and exported rasters to GCS.
- **Sentinel-5P**: Built a data fetch pipeline for Sentinel-5P (TROPOMI) atmospheric imagery over the AOI using GEE and exported to GCS.
- **Landsat-8**: Built a data fetch pipeline for Landsat-8/9 imagery over the AOI via GEE and exported to GCS.

### Milestone 3
- Developed a data pipeline to fetch and process numeric data from Sentinel-2 and Sentinel-5P datasets, enabling a streamlined workflow for downstream tabular analysis.
- Conducted experiments with multimodal models, focusing on hyperparameter tuning to evaluate parameter configurations on model performance.
- Performed exploratory analysis and iterative testing to validate the data pipeline and assess model behavior under various parameter combinations.

### Milestone 4
- Audited the prepared numerical archive for D+1 labels, ERA5 through D−5 features, chronological train/validation/test splits, and data leakage risks.
- Implemented and tuned the isolated V1–V5 LightGBM/MLP experiment track, adding causal fire-history and direction-aware features, classifier–ranker blending, and hard-negative reranking; retained V4 with 38.33% Recall@25.
- Added reproducible training/evaluation scripts, 30 unit tests, metric comparisons, California risk maps, and consolidated report documentation.

### Milestone 5
- Integrated train/test split scripts with existing training and inference notebooks.
- Created refined training and inference notebooks from Milestone 4 code for improved experimentation, validation, and model testing.

### Milestone 6
- Conducted final code auditing, ensuring all script paths, test suites, and model loading functions operate smoothly in production deployment environments.
- Supported final documentation and repository organization.

---

## Lakshmi Sruthi K (Roll Number: 21F1005626)

### Milestone 1
- Identified and researched relevant datasets available on Google Earth Engine for wildfire detection and early warning.
- Contributed to the design and preparation of the project presentation (PPT).
- Participated in team brainstorming sessions to refine the project concept and scope.

### Milestone 2
- Explored Google Earth Engine datasets and finalized dataset selection.
- Processed the FIRMS wildfire dataset: extracted daily FIRMS GeoTIFF label rasters, reviewed band structures (T21, confidence), and established daily binary fire/no-fire labeling logic.
- Visualized FIRMS data in GEE and created scripts to export 10 years of FIRMS data to the GCS bucket.

### Milestone 3
- Identified optimal FIRMS parameters and confidence thresholds for establishing fire ground truth labels based on historical fire event datasets.

### Milestone 4
- Analyzed dataset structure and pipeline requirements, extracted S5P data for downstream numerical processing, and resolved uncertainties around feature construction.
- Investigated alternative grid-level FIRMS thresholding to align detections with actual events, assessing computational scalability bounds.
- Set up GCP execution instances to run and extend pipeline models in a cloud environment.

### Milestone 5
- Conducted an extensive analysis of historical wildfire hotspots by aggregating FIRMS fire pixels across 672 land regions and classifying each region into Low, Medium, and High-risk categories (`fire_analysis2.csv`).
- Designed, developed, and experimented with a novel ensemble modeling approach combining LightGBM and histogram-based XGBoost (`XGBoost-LGBM-blend.ipynb`).
- Collated, organized, and analyzed experimental results across all machine learning models developed throughout the milestone.

### Milestone 6
- Led the team and formulated strategic plans to partition project deliverables across deployment infrastructure, web application development, automated data pipeline, and report synthesis.
- Authored the **Non-Technical Report**, articulating the system's real-world impact, operational utility, stakeholder benefits, and deployment guidelines.
- Contributed significantly to authoring the **Technical Report** and synthesizing the final project presentation deck.

---

## R Aditya (Roll Number: 21F1004839)

### Milestone 1
- Researched wildfire incidents and case studies to understand real-world ignition causes and early detection failure points.
- Studied environmental and climatic factors contributing to wildfire occurrences.
- Brainstormed and documented overall project scope, boundaries, stakeholders, metrics, and system capabilities.

### Milestone 2
- Brainstormed selection of datasets and extraction methodologies.
- Formulated final data structures relevant for ingestion into ML models.
- Extracted Sentinel-2 and Sentinel-5P data for the image processing arm of the model.
- Collated insights and data extractions for reporting.

### Milestone 3
- Supported data ingestion strategy and feature extraction across satellite modalities.

### Milestone 4
- Collaborated on developing automated workflows for extracting and organizing daily Sentinel-5P satellite observations.
- Investigated spatial and temporal gaps in Sentinel-5P datasets by performing missing data analysis and quality checks.
- Designed data collation strategies to integrate multi-source geospatial datasets while maintaining temporal consistency.
- Evaluated missing value imputation techniques for satellite observations.
- Assisted in preprocessing atmospheric variables and prepared lag-based derived variables.

### Milestone 5
- **Milestone Team Lead**: Coordinated technical discussions, drove research direction, and facilitated architectural decisions.
- Designed and developed a **Spatial-Temporal Transformer** sequential modeling approach (`wildfire-prediction-transformer-architecture-v2.ipynb`) to predict next-day wildfire occurrence using multi-day observation windows.
- Proposed the adoption of **KNN-based spatial imputation** as an improved strategy for handling missing Sentinel-2 data over median imputation.
- Initiated historical fire pixel spatial hotspot analysis for risk classification.

### Milestone 6
- Designed and implemented the FastAPI backend to support model inference and data pipeline execution for user-specified dates, providing an API layer for interacting with the wildfire prediction workflow.

- Containerized the application components using Docker and established the deployment architecture across GCP services, including Cloud Run, Compute Engine, Artifact Registry, and GCS.

- Wrote the Docker Compose manifest and Nginx configuration to orchestrate and serve the frontend and backend services together on a GCP Compute Engine VM.

- Designed the deployment workflow for the automated daily extraction and inference pipeline, with containerized jobs stored in Artifact Registry, scheduled execution, and model inputs and inference outputs persisted to GCS.

---

## Signatures

| Member | Roll Number | Signature Commit |
| :--- | :--- | :---: |
| Ripunjay Kumar | 21F3002511 | ✅ |
| Lakshay Garg | 21F3001076 | ✅ |
| Roushan Kumar Singh | 23F1002240 | ✅ |
| Lakshmi Sruthi K | 21F1005626 | ✅ |
| R Aditya | 21F1004839 | ✅ |
