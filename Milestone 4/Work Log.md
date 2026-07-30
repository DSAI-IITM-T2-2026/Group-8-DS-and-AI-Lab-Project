# Work Logs - Milestone 4

## Lakshay Garg (Roll Number: 21F3001076)

- Built the Milestone 4 numerical next-day pipeline under `Milestone 4/numerical_nextday/` (ERA5 lag-5 primary, causal S2/S5P attach, Stage A/B/C tables for 2019–2025).
- Assembled train/val/test splits (2019–2022 / 2023–2024 / 2025) and trained LightGBM fire-season + jan/feb/mar/dec models, Stage C HP sweep, MLP secondary,
- Wrote eval artifacts (`experiments_log.csv`, `eval_metrics.json`, figures, sample alerts) and filled `Report.md` findings.


## Ripunjay Kumar (Roll Number: 21F3002511)

- Designed and implemented **LagFireNet**, a live-ready California wildfire alert model under `Milestone 3/wildfire_pipeline/` (FIRMS ~1 km dense maps; predict day **D** from features as of **D−2**).
- Built the full lag-consistent data path: FIRMS reference grid + land mask, ERA5 daily regrid (D−8…D−2), S5P AAI/CO rasterize with LOCF/valid/age, S2 numerical indices + lag channel, static DEM; local-first GCS fetch with year→bucket maps.
- Implemented `LagFireNet` (ERA5/S5P ConvLSTM + DEM/S2 encoders → fuse → U-Net), Focal+Tversky (mean-reduced, precision-first), tile dataset (256×256, fire oversample), train with early stop on val precision@0.5, isotonic calibration, and deploy threshold (precision ≥ 0.4).
- Added daily inference (`infer_live.py`) and cluster alert evaluation (`evaluate_alerts.py`); wrote `REPORT.md` documenting the shift from MultimodalFusion (cell risk) to LagFireNet (1 km lag-consistent alerts).
- Debugged training blockers (empty land mask wiping all modalities; Tversky sum-scale loss stuck near 1.0; disk-blowing full-day caches) and started fire-season training (train 2022–2023 / val 2024).

## Roushan Kumar Singh (Roll Number: 23F1002240)

- Audited the prepared numerical archive for D+1 labels, ERA5 through D−5, chronological train/validation/test splits, and potential data leakage.
- Implemented and tuned the isolated V1–V5 LightGBM/MLP experiment track, adding causal fire-history and direction-aware features, classifier–ranker blending and hard-negative reranking; retained V4 with 38.33% Recall@25.
- Added reproducible training/evaluation scripts, 30 tests, metric comparisons, California risk maps and the consolidated Milestone 4 report while excluding large input data, caches, virtual environments and model weights from Git.

## R Aditya (Roll Number: 21F1004839)
- Collaborated on developing automated workflows for extracting and organizing daily Sentinel-5P satellite observations for downstream analysis.
- Investigated spatial and temporal gaps in Sentinel-5P datasets by performing comprehensive missing data analysis and quality checks.
- Designed data collation strategies to integrate multi-source geospatial datasets while maintaining temporal consistency.
Explored and evaluated missing value imputation techniques for satellite observations, considering temporal continuity and feature reliability.
- Assisted in preprocessing atmospheric variables by cleaning, validating, and standardizing large-scale satellite datasets.
Contributed to the development of reproducible data pipelines for efficient ingestion and preparation of Earth observation data.
- Performed exploratory data analysis to identify data quality issues, coverage inconsistencies, and seasonal trends in satellite measurements.
Supported feature engineering by preparing lag-based and derived variables for machine learning workflows.
- Worked on validating processed datasets through consistency checks, ensuring compatibility across different satellite products and time periods.
Collaborated with the team to optimize preprocessing workflows, improving data availability and reducing manual intervention in the pipeline.


## Signatures

| Member              | Roll Number | Signature Commit |
| ------------------- | ----------- | ---------------- |
| Ripunjay Kumar      | 21F3002511  | ✅               |
| Lakshay Garg        | 21F3001076  | ✅               |
| Roushan Kumar Singh | 23F1002240  |   ✅            |
| Lakshmi Sruthi K    | 21F1005626  |                |
| R Aditya            | 21F1004839  |                |
