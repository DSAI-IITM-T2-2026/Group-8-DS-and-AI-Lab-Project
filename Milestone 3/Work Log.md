# Work Logs - Milestone 3

## Lakshay Garg (Roll Number: 21F3001076)

- Built the local MPS patch-segmentation stack under `Milestone 3/Experiments/` (30 fused channels, fire-centered 64×64 patches).
- Completed full-year 2025 candidate build (1035 patches) and trained HistGB + ConvLSTM+U-Net.
- Added multi-model experiments (baseline, ConvLSTM BCE+Dice/Focal, U-Net last-day), confusion matrices, and `run_experiments.py` (test F1 ≈ 0.029 / 0.110 / 0.086).
- Supported in documentation.


## Ripunjay Kumar (Roll Number: 21F3002511)

- ⁠Designed and documented the Milestone 3 ML lifecycle (⁠ ARCHITECTURE.md ⁠): next-day cell-day fire risk, ERA5 0.25° grid, FIRMS labels, train/val/test protocol.
- ⁠Built the tabular baseline pipeline ⁠ mvp_era5_dem/ ⁠ (ERA5 + DEM → LightGBM).
- ⁠Implemented progressive fusion models: ⁠ cnn_s2_mvp/ ⁠, ⁠ cnn_lstm_fusion/ ⁠ (CNN + LSTM ± S5P), and full hybrid ⁠ multimodal_fusion/ ⁠ (S2/S5P CNNs + LSTM + numerical MLPs).
- Released checkpoints and metrics under ⁠ cnn_lstm_fusion/artifacts/ ⁠ and ⁠ multimodal_fusion/artifacts/ ⁠; added map/prediction scripts and project READMEs.



## Roushan Kumar Singh (Roll Number: 23F1002240)

- Developed a data pipeline to fetch and process numeric data from Sentinel-2 and Sentinel-5P datasets, enabling a streamlined workflow for downstream analysis and model development.
- Conducted experiments with multimodal models, focusing on hyperparameter tuning to evaluate the impact of different configurations on model performance and identify optimal settings.
- Performed exploratory analysis and iterative testing to validate the data pipeline and assess model behavior under various parameter combinations.



## Lakshmi Sruthi K (Roll Number: 21F1005626)

- ⁠Identified FIRMS parameter and threshold to be used for identifying whether a day is a fire day based on actual fire events dataset.


## R Aditya (Roll Number: 21F1004839)

- 

---



## Signatures


| Member              | Roll Number | Signature Commit |
| ------------------- | ----------- | ---------------- |
| Ripunjay Kumar      | 21F3002511  |      ✅            |
| Lakshay Garg        | 21F3001076  |      ✅            |
| Roushan Kumar Singh | 23F1002240  |       ✅          |
| Lakshmi Sruthi K    | 21F1005626  |      ✅            |
| R Aditya            | 21F1004839  |                  |


