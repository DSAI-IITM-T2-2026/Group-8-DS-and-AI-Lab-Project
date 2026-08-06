# Work Logs - Milestone 5

## Lakshay Garg (Roll Number: 21F3001076)

- Consolidated Stage C / Stage C KNN datasets on Kaggle (`California_Wildfire_KNN`, `California_Wildfire_Median`, `firms_test` / `fire_analysis2.csv`) for reproducible training and inference Inputs.
- Fixed the end-to-end training notebook (`Wildfire_Training_v2.ipynb`): Kaggle path/config, train/val splits, Optuna tuning, validation metrics, and artifact export (`wildfire_model.joblib`).
- Fixed the inference notebook (`Wildfire_Inference.ipynb`) to load training notebook outputs and score 2025 via `test.parquet` with the same feature contract.
- Tested the pipeline on fire-region cell subsets: high-fire cells, high + medium cells, and all cells.


## R Aditya (Roll Number: 21F1004839)
- Led the team for this milestone by coordinating technical discussions, driving research direction, and facilitating architectural decisions to ensure steady progress towards the project objectives.
- Ideated, designed, and developed a Transformer-based sequential modelling approach to predict next-day wildfire occurrence using observations from previous days, exploring temporal dependencies across the different selected datasets.(Available as `wildfire-prediction-transformer-architecture-v2.ipynb`)
- Proposed the adoption of KNN-based imputation as an improved strategy for handling missing Sentinel-2 data, with the objective of preserving local feature relationships and improving data quality over the previously used median imputation approach.
- Initiated the analysis of historical fire pixels to identify recurring wildfire hotspots, exploring their potential use for spatial risk assessment, feature engineering, and improving future modelling efforts.
- Conducted technical research and experimentation on temporal learning architectures, multimodal geospatial data integration, and preprocessing strategies to support the development of a more robust wildfire prediction pipeline.

## Lakshmi Sruthi K (Roll Number: 21F1005626)
- Conducted an extensive analysis of historical wildfire hotspots by aggregating FIRMS fire pixels across 672 land regions and classifying each region into Low, Medium, and High-risk categories. Explored the use of these spatial risk priors to guide model training towards fire-prone regions, with the objective of improving predictive performance and reducing class imbalance effects.
- Designed, developed, and experimented with a novel ensemble modelling approach combining LightGBM and histogram-based XGBoost, investigating how complementary gradient boosting algorithms could improve robustness, generalization, and overall wildfire prediction performance.
- Collated, organized, and analysed experimental results from the various machine learning models developed throughout the milestone.


## Signatures


| Member              | Roll Number | Signature Commit |
| ------------------- | ----------- | ---------------- |
| Ripunjay Kumar      | 21F3002511  |                  |
| Lakshay Garg        | 21F3001076  | ✅                |
| Roushan Kumar Singh | 23F1002240  |                  |
| Lakshmi Sruthi K    | 21F1005626  | ✅               |
| R Aditya            | 21F1004839  | ✅️               |


