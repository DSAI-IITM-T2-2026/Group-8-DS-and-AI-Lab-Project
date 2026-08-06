# Work Logs - Milestone 5

## Lakshay Garg (Roll Number: 21F3001076)

- Consolidated Stage C / Stage C KNN datasets on Kaggle (`California_Wildfire_KNN`, `California_Wildfire_Median`, `firms_test` / `fire_analysis2.csv`) for reproducible training and inference Inputs.
- Fixed the end-to-end training notebook (`Wildfire_Training_v2.ipynb`): Kaggle path/config, train/val splits, Optuna tuning, validation metrics, and artifact export (`wildfire_model.joblib`).
- Fixed the inference notebook (`Wildfire_Inference.ipynb`) to load training notebook outputs and score 2025 via `test.parquet` with the same feature contract.
- Tested the pipeline on fire-region cell subsets: high-fire cells, high + medium cells, and all cells.



## Signatures


| Member              | Roll Number | Signature Commit |
| ------------------- | ----------- | ---------------- |
| Ripunjay Kumar      | 21F3002511  |                  |
| Lakshay Garg        | 21F3001076  | ✅                |
| Roushan Kumar Singh | 23F1002240  |                  |
| Lakshmi Sruthi K    | 21F1005626  |                  |
| R Aditya            | 21F1004839  |                  |


