# Early Wildfire Prediction 
## Milestone 5: Model Evaluation & Analysis Report


## 1. Introduction & Objectives

### 1.1 Selected Model Checkpoints & Architecture Overview
This evaluation phase benchmarked four distinct model checkpoints developed during Milestone 4 and Milestone 5:

1. LightGBM Dual-Head Model:
   A dual-head pipeline combining a LightGBM Binary Classifier (`LGBMClassifier`) for absolute probability estimation and a LightGBM LambdaRank Ranker (`LGBMRanker`) for spatial relative risk ordering, calibrated using a post-hoc Logistic Regression model.
2. Heterogeneous GBDT Ensemble:
   A dual-engine 50/50 ensemble combining LightGBM and histogram-accelerated XGBoost (`XGBClassifier` + `XGBRanker`) via daily within-day percentile score fusion.
3. Spatial-Temporal Transformer:
   A multi-modal 4-layer Transformer Encoder with 5 dedicated feature projection sub-encoders (ERA5, Sentinel-2, Sentinel-5P, DEM, and Causal Fire History) trained with a joint `HybridFocalRankingLoss`.
4. Champion CatBoost Model (Stage C KNN):
   A CatBoost gradient boosting pipeline (`CatBoostClassifier` + `CatBoostRanker`) operating on precomputed KNN spatial-temporal imputations, trained on 86 pruned features over high & medium fire-prone cells, using a 100% classifier daily percentile score blend.

---

## 2. Evaluation Setup & Test Dataset

### 2.1 Strictly Held-Out Test Set Details
Data integrity and temporal non-leakage were strictly enforced across all chronological splits:

- **Full Dataset Scope**: 2,557 calendar days (Jan 1, 2019 – Dec 31, 2025) across 672 California land cells (total 1,718,304 cell-days).
- **Training Split (2019–2022 / 2023)**: Used exclusively for model fitting and Optuna hyperparameter search.
- **Calibration Split (2024)**: Used exclusively for fitting the Logistic Regression Platt probability calibrator.
- **Held-Out Test Set (2025)**: 365 calendar days (Jan 1, 2025 – Dec 31, 2025), totaling **245,280 cell-days** and **2,275 true positive wildfire events** (0.928% positive rate). Zero 2025 rows influenced model fitting or hyperparameter selection.

### 2.2 Feature Pipeline Contract & Pruning Strategy (107 Initial -> 86 Kept Features)
The feature engineering pipeline ingests 63 raw source columns and constructs **107 initial candidate features**. To reduce multicollinearity, lower memory overhead, and optimize downstream model generalization, a **20% feature pruning step** (`drop_fraction: 0.2`) was applied during preprocessing (Stage C KNN pipeline).

#### 1. Feature Pruning Overview
- **Initial Feature Count**: 107 features
- **Pruning Rate**: 20.0% (21 features dropped)
- **Final Model Feature Set**: **86 features**
- **Imputation Method**: Precomputed K-Nearest Neighbors (`precomputed KNN`)
- **Target Spatial Cell Subset**: High & Medium Fire Prone Cells (437 selected grid cells out of 672 archive cells)
- **Features Used**: present in `metrics_summary.json`


### 2.3 Definition of Baseline Models
The final models were benchmarked against four established baselines:
1. **Naive Constant Rate Baseline**: Predicts constant positive rate (0.0093). `PR-AUC = 0.0093`, `Recall@25 = 3.7%`.
2. **Persistence Baseline (Yesterday's Fire == Tomorrow's Fire)**: Predicts tomorrow's fire solely from D-1 fire status. `PR-AUC = 0.0412`, `Recall@25 = 14.8%`.
3. **Standard Logistic Regression Baseline**: Linear model on raw ERA5 and terrain features. `PR-AUC = 0.0782`, `Recall@25 = 19.4%`.
4. **Milestone 4 V1/V2 Weather-Only Baseline**: GBDT model excluding causal fire history. `PR-AUC = 0.0958`, `Recall@25 = 25.1%`.


### 2.4 Historical Fire Activity Analysis & Regional Splitting 
To analyze spatial fire density across California's 672 land cells and prevent zero-inflated bias from non-burnable desert/urban cells, historical fire activity was analyzed:

1. **Cumulative Pixel Aggregation**: For each valid cell ID, total historical FIRMS thermal fire pixels were aggregated over the dataset period:
   firms_n_pixels_sum = df.groupby('cell_id')['y_fire'].sum()
2. **Quantile Risk Categorization**: Quantiles Q1 (25th percentile) and Q3 (75th percentile) were computed across all grid cells to define three distinct fire prone risk tiers:
   - **Low Fire Region**: Total historical fire pixels <= Q1 (25th percentile).
   - **Medium Fire Region**: Total historical fire pixels between Q1 and Q3 (25th to 75th percentile).
   - **High Fire Region**: Total historical fire pixels >= Q3 (75th percentile).
   - Outliers were flagged using standard Interquartile Range rules (IQR = Q3 - Q1, Lower Bound = Q1 - 1.5 * IQR, Upper Bound = Q3 + 1.5 * IQR).

The following figure illustrates the spatial distribution of California's grid cells categorized into outlier, high-, medium-, and low-fire-risk regions based on historical cumulative fire activity.
![](https://raw.githubusercontent.com/DSAI-IITM-T2-2026/Group-8-DS-and-AI-Lab-Project/refs/heads/main/Milestone%205/images/fire_analysis2_category_grid_map.png)



---

## 3. Metric Selection & Justification

### 3.1 Quantitative Metrics Specification
The evaluation framework uses six exact quantitative metrics:

* **PR-AUC (Precision-Recall Area Under Curve / Average Precision)**: Primary metric for severe class imbalance (~1.3% positive rate).
* **Recall @ 25 (Top-25 Daily Fire Recall)**: Proportion of all true statewide wildfires captured within a daily operational alert budget of 25 grid cells (k = 25).

### 3.2 Metric Justification & Business Context
- **Why PR-AUC instead of Accuracy?** Over 98.7% of cell-days have no fire. A dummy model predicting "no fire" for every cell gets 98.7% Accuracy but is useless. PR-AUC directly penalizes false alarms on rare positive events.
- **Why Recall@25 & Precision@25?** CAL FIRE emergency command centers have a fixed daily dispatch capacity of ~25 priority reconnaissance sweeps per day. Top-25 metrics directly mirror real-world responder capacity.

### 3.3 Explicit Mapping of Error Trade-Offs
- **False Negative (FN) Error — Unflagged Wildfire Ignition**: Catastrophic loss of life and property damage (e.g. 2018 Camp Fire caused $16.5B in damages). **Extremely High Penalty**.
- **False Positive (FP) Error — False Alarm Alert**: Routine verification sweep where no fire occurs. **Moderate Penalty (Minor patrol cost)**.
- **Task Priority**: Minimizing False Negatives (**maximizing Recall@25**) while maintaining acceptable precision.

### 3.4 Metric Benchmark Table Across All 3 Notebooks

| Metric |LightGBM Dual-Head Model | Spatial-Temporal Transformer | LightGBM + XGBoost Dual-Engine Ensemble |
| :--- | :--- | :--- | :--- |
| **PR-AUC (Primary)** | **0.3524** (Full 672 Grid) | **0.1971** (Validation) / **0.1638** (Test) | **0.2353** (High-Medium Fire Subset) |
| **ROC-AUC** | **0.8981** | **0.7526** | **0.8425** |
| **Recall @ 25 / day** | 43.50% | 39.59% (at p = 0.50) / **97.48%** (at p = 0.30) | **47.21%** (Highest Operational Recall) |
| **Loss Function** | Binary LogLoss + LambdaRank Loss | Hybrid Focal Loss + Pairwise Margin Ranking Loss | Binary LogLoss + NDCG Ranking Loss |
| **Calibration Method** | Logistic Regression Platt Calibrator | Post-Hoc Sigmoid Logit Calibration | Logistic Regression Platt Calibrator |

---


## 9. Deployment Readiness Assessment

### 9.1 Trade-Off Analysis: Accuracy vs. Speed vs. Memory

| Model Architecture | PR-AUC | Recall @ 25 | GPU Inference Latency (672 cells) | CPU Inference Latency (672 cells) | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Champion** | **0.3524** | 43.50% | **4.2 ms** | **12.5 ms** | **18 MB** |
| **Dual GBDT Ensemble** | 0.2353 | **47.21%** | **8.6 ms** | **24.1 ms** | **42 MB** |
| **Transformer Model** | 0.1971 | 39.59% | **42.1 ms** | 185.0 ms | **312 MB** |

### 9.2 Model Compression & Production Export
* **Serialized Model Export**: Final models were packaged into a 18 MB standalone .joblib artifact containing tree weights, probability calibrators, and feature contracts.
* **ONNX Runtime Acceleration**: Exporting the LightGBM booster to ONNX runtime reduces CPU inference latency to **2.8 ms per daily statewide sweep**, enabling real-time edge execution on field laptops or low-power command center servers.

## Signatures

| Member              | Roll Number | Signature Commit |
| ------------------- | ----------- | ---------------- |
| Ripunjay Kumar      | 21F3002511  |                  |
| Lakshay Garg        | 21F3001076  |                  |
| Roushan Kumar Singh | 23F1002240  |                  |
| Lakshmi Sruthi K    | 21F1005626  |                  |
| R Aditya            | 21F1004839  |                  |
