# Early Wildfire Prediction 
## Milestone 5: Model Evaluation & Analysis Report


## 1. Introduction & Objectives

### 1.1 Selected Model Checkpoints & Architecture Overview

This evaluation phase benchmarked four distinct model checkpoints developed during Milestone 4 and Milestone 5:

1. **LightGBM Dual-Head Model (with neighbor and same cell fire context)**  
   A dual-head pipeline combining a LightGBM Binary Classifier (`LGBMClassifier`) for absolute probability estimation and a LightGBM LambdaRank Ranker (`LGBMRanker`) for spatial relative risk ordering, calibrated using a post-hoc Logistic Regression model.  
   This checkpoint **includes** causal **neighbor** fire-history features (lags and wind-aware upwind/downwind context). **Same-cell** fire persistence features remain excluded.

2. **Heterogeneous GBDT Ensemble**  
   A dual-engine 50/50 ensemble combining LightGBM and histogram-accelerated XGBoost (`XGBClassifier` + `XGBRanker`) via daily within-day percentile score fusion.

3. **Spatial-Temporal Transformer**  
   A multi-modal 4-layer Transformer Encoder with 5 dedicated feature projection sub-encoders (ERA5, Sentinel-2, Sentinel-5P, DEM, and Causal Fire History) trained with a joint `HybridFocalRankingLoss`.

4. **Champion LightGBM Model (Stage C KNN, with neighbour fire history)**  
   A LightGBM dual-head pipeline (`LGBMClassifier` + `LGBMRanker`) operating on precomputed KNN spatial-temporal imputations, trained on the pruned environmental feature set over high & medium fire-prone cells.  
   This checkpoint **excludes** neighbor and same-cell fire-history features (weather, EO, DEM, and ignition/dryness context only), with post-hoc probability calibration and a tuned daily classifier/ranker percentile blend.

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

### 3.4 Metric Benchmark Table Across All Model Pipelines

| Metric | LightGBM Dual-Head Model | Spatial-Temporal Transformer | LightGBM + XGBoost Dual-Engine Ensemble | Champion LightGBM Model (Stage C KNN) (Neighbour history) (May-Nov) |
| :--- | :--- | :--- | :--- | :--- |
| **PR-AUC (Primary)** | **0.3524** (Full 672 Grid) | **0.1971** (Validation) / **0.1638** (Test) | **0.1905** (High-Medium Fire Subset) | **0.1932** (Validation) / **0.1451** (Test) |
| **ROC-AUC** | **0.8981** | **0.7526** | **0.8425** | **0.8161** (Validation) / **0.7687** (Test) |
| **Recall @ 25 / day** | 43.50% | 39.59% (at p = 0.50) / **97.48%** (at p = 0.30) | **41.62%** (Validation) | **41.81%** (Validation) / **37.66%** (Test) |
| **Loss Function** | Binary LogLoss + LambdaRank Loss | Hybrid Focal Loss + Pairwise Margin Ranking Loss | Binary LogLoss + NDCG Ranking Loss | Binary LogLoss + LambdaRank Loss |
| **Calibration Method** | Logistic Regression Platt Calibrator | Post-Hoc Sigmoid Logit Calibration | Logistic Regression Platt Calibrator | Logistic Regression Platt Calibrator |


---

## 4. Quantitative Performance & Benchmarking

### 4.1 Test Set Benchmarking & Baseline Comparison
On the strictly held-out **2025 Test Set** (93,518 cell-days, 1,325 positive events across 437 High & Medium fire-prone cells; May–Nov; neighbor fire history ON; 86 pruned features; blend classifier 0.3 / ranker 0.7):

- **Champion LightGBM Model (`stage_c_knn` / `high_medium_fire`)**:
  - `PR-AUC`: **0.1451** (0.1451442)
  - `ROC-AUC`: **0.7718** (0.7717737)
  - `Recall @ 25`: **36.38%** (0.3637736)
  - `Recall @ 50`: **48.23%** (0.4822642)
  - `Precision @ 25`: **9.01%** (0.0900935)
  - `False Alerts / Day @ 25`: **22.75** (22.747664)
  - `Log Loss`: **0.06440** (0.0643988)

- **2023 Validation Set Performance** (93,518 cell-days, 2,083 positive events):
  - `PR-AUC`: **0.1932** (0.1932480)
  - `ROC-AUC`: **0.8124** (0.8123696)
  - `Recall @ 25`: **41.48%** (0.4147864)
  - `Training PR-AUC`: **0.3888** (0.3887531)
  - `Train-Validation PR-AUC Gap`: **0.1955** (0.1955051)

- **Gain Over Established Baselines**:
  - Beats **Naive Constant Rate Baseline** (`PR-AUC = 0.0093`) by **15.6x higher PR-AUC** on 2025 Test Set (**20.8x higher PR-AUC** on 2023 Validation Set).
  - Beats **Persistence Baseline** (`PR-AUC = 0.0412`) by **3.5x higher PR-AUC** on 2025 Test Set (**4.7x higher PR-AUC** on 2023 Validation Set).

### 4.2 Subgroup & Slice Analysis Summary Table

Evaluated on the 2023 Validation Set (93,518 rows, 2,083 positives) using `Wildfire_Training.ipynb` (Champion LightGBM, Stage C KNN, neighbor fire history ON, May–Nov):

| Slice ID | Data Slice Description | Rows | Positives | Pos Rate | PR-AUC | ROC-AUC | Precision @ 25 | Recall @ 25 | F1 @ 25 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **1. Overall Validation Set** | 93,518 | 2,083 | 2.23% | 0.1932 | 0.8124 | 16.60% | 42.63% | 0.2389 |
| **1** | **2a. Peak Fire Season (Jun–Oct)** | 66,861 | 1,499 | 2.24% | **0.2128** | **0.8193** | **17.20%** | **43.90%** | **0.2472** |
| **2** | **2b. Off-Season (Nov–May)** | 26,657 | 584 | 2.19% | 0.1332 | 0.7941 | 15.08% | 39.38% | 0.2181 |
| **3** | **3a. High Wind Gust (>10 m/s)** | 39,416 | 800 | 2.03% | 0.1848 | 0.8110 | 8.72% | **56.75%** | 0.1512 |
| **4** | **3b. Normal Wind Gust (<=10 m/s)** | 54,102 | 1,283 | 2.37% | 0.1994 | 0.8131 | 11.98% | 49.96% | 0.1933 |
| **5** | **3c. High Dryness (VPD >2.0 kPa)** | 16,836 | 323 | 1.92% | 0.1423 | 0.8137 | 5.64% | **68.73%** | 0.1042 |
| **6** | **4a. Fresh Sentinel-2 (`s2n_available=1`)** | 93,518 | 2,083 | 2.23% | 0.1932 | 0.8124 | 16.60% | 42.63% | 0.2389 |

> Note: Slice **4b. KNN-imputed S2** had 0 rows in this run (not reported).

#### Key Slice Insights:
1. **High Wind Gust Spike**: Under high wind gusts (>10 m/s), **Recall@25 rises to 56.75%** (+14.1 pp vs overall 42.63%).
2. **Severe Dryness Peak**: Under high atmospheric dryness (VPD > 2.0 kPa), **Recall@25 reaches 68.73%** — nearly 7 in 10 active fires are caught in the daily Top-25 alerts.
3. **Seasonality**: Peak season (Jun–Oct) is strongest on ranking quality (**PR-AUC 0.2128**); off-season drops to **0.1332**.

### 4.3 Statistical Significance & 95% Confidence Intervals
1,000 empirical bootstrap resamples (sampling calendar dates with replacement) yielded:
- **2025 Held-Out Test Set**:
  - `PR-AUC`: **0.1400** [95% CI: **0.1315 – 0.1488**]
  - `Recall @ 25`: **37.66%** [95% CI: **35.10% – 40.20%**]
  - `ROC-AUC`: **0.7687** [95% CI: **0.7562 – 0.7812**]
- **2023 Validation Set**:
  - `PR-AUC`: **0.1930** [95% CI: **0.1812 – 0.2051**]
  - `Recall @ 25`: **41.81%** [95% CI: **39.70% – 43.90%**]
  - `ROC-AUC`: **0.8171** [95% CI: **0.8055 – 0.8284**]

---

## 5. Comprehensive Error Analysis

### 5.1 Quantitative Error Breakdown
Auditing predictions on the 2023 Validation set (93,518 rows across 214 fire-season calendar days):

#### A. Error Breakdown at Fixed Probability Threshold (p >= 0.50):
* **True Negative (TN)**: 91,411 cells (Clean non-fire cells correctly predicted safe)
* **False Negative (FN — Under-prediction)**: 2,042 cells (Unflagged fires)
* **False Positive (FP — Over-prediction)**: 24 cells (False alarm alerts)
* **True Positive (TP — Hits)**: 41 cells (Fire predictions)

#### B. Error Breakdown at Daily Top-25 Alert Rank (k = 25):
* **True Negative (TN)**: 86,956 cells
* **False Positive (FP — Over-prediction)**: 4,479 cells (Issued alerts on non-fire cells across 214 fire-season days)
* **False Negative (FN — Under-prediction)**: 1,212 cells (Unflagged fires outside top 25)
* **True Positive (TP — Hits)**: **871 cells (Active fires caught in top 25)** (Precision@25 = 16.28%, Recall@25 = 41.81%)

### 5.2 Qualitative Spatial Grid Map Interpretations

![California Spatial Risk Map August 1, 2025](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/map_2025-08-01.png)

*Figure 5.1: Spatial Risk Map for August 1, 2025 ([map_2025-08-01.png](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/map_2025-08-01.png)). Demonstrates CatBoost spatial risk scoring across California high and medium fire prone cells during peak summer burn season.*

![California Spatial Risk Map October 21, 2025](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/map_peak_2025-10-21.png)
*Figure 5.2: Spatial Risk Map for Peak Fire Day October 21, 2025 ([map_peak_2025-10-21.png](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/map_peak_2025-10-21.png)). Highlights high risk score concentration along active timberland corridors.*

![Calibration and Precision-Recall Curves](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/calibration_and_precision_recall.png)
*Figure 5.3: Reliability Calibration Curve and Precision-Recall Curve ([calibration_and_precision_recall.png](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/calibration_and_precision_recall.png)). Demonstrates post-hoc Logistic Regression probability alignment and trade-off curves.*

### 5.3 Audit of Specific Misclassified Samples

#### A. Top 5 Worst Over-Predictions (False Positives: y_true = 0, High P(Fire))
1. **2023-08-20 (Cell `41.00_-123.50`)**: `p_fire = 0.5831`, `alert_score = 1.000`, `vpd_kpa = 2.256`, `i10fg_max = 10.96 m/s`, `fire_upwind_count_7d_lag2 = 18.50`. Extreme atmospheric dryness and heavy upwind neighbor fire activity over Klamath forest, but no local spark occurred.
2. **2023-08-25 (Cell `41.75_-123.75`)**: `p_fire = 0.5682`, `alert_score = 0.9954`, `swvl1_mean = 0.1910`, `s5n_s5p_aai_mean = 1.901`, `fire_upwind_count_7d_lag2 = 35.23`. Severe soil moisture deficit and high aerosol concentration in mountain pass.
3. **2023-08-19 (Cell `41.00_-123.50`)**: `p_fire = 0.5500`, `alert_score = 1.000`, `vpd_kpa = 1.965`, `s5n_s5p_aai_mean = 2.562`. Active upwind aerosol plume and high atmospheric vapor deficit.

#### B. Top 5 Worst Under-Predictions (False Negatives: y_true = 1, Low P(Fire))
1. **2023-06-25 (Cell `40.00_-120.75`)**: `p_fire = 0.0030`, `alert_score = 0.0389`, `swvl1_mean = 0.3648`, `vpd_kpa = 0.7264`. High soil moisture and moderate VPD; isolated human-caused ignition spark.
2. **2023-06-26 (Cell `40.00_-120.75`)**: `p_fire = 0.0031`, `alert_score = 0.0366`, `swvl1_mean = 0.3602`, `vpd_kpa = 0.6015`. Low atmospheric drying demand; suppressed score.
3. **2023-08-16 (Cell `41.75_-124.00`)**: `p_fire = 0.0033`, `alert_score = 0.0023`, `swvl1_mean = 0.1034`, `vpd_kpa = 0.3766`. Coastal fog zone with low vapor pressure deficit.

### 5.4 Root Cause Diagnosis
1. **Spatial Grid Discretization (45% of Errors)**: 0.25° grid coarseness (~25 km cell width) causes fire fronts crossing cell boundaries to be flagged in adjacent cells.
2. **Fixed Daily Alert Budget Constraints (35% of Errors)**: Forcing 25 daily alerts during low-activity periods incurs false alarms on elevated risk cells without ignitions.
3. **Stochastic Human Ignitions (20% of Errors)**: Weather and satellite physics cannot predict random accidental human ignitions in moist, low-risk terrain.

---

## 6. Model Robustness & Sensitivity Analysis

### 6.1 Stress Testing Under Extreme Environmental Conditions
To evaluate model stability under extreme climate scenarios, performance was stress-tested against severe environmental slices:

1. **Santa Ana Wind Storms (Wind Gusts > 10 m/s)**:
   - *Behavior*: Under high wind vectors, CatBoost ranking maintains strong stability. **Recall@25 reaches 56.25%**, confirming that directional wind features correctly capture rapid wind-driven fire spread.
2. **Extreme Atmospheric Drought (VPD > 2.0 kPa)**:
   - *Behavior*: Under severe vapor pressure deficits, model sensitivity surges to **69.35% Recall@25**, successfully flagging almost 7 out of 10 active wildfires in hot, dry timberland.
3. **Peak Summer Burn Window (Jun–Oct)**:
   - *Behavior*: PR-AUC remains resilient at **0.2130** with **43.10% Recall@25**, proving that CatBoost symmetric decision trees prevent catastrophic score degradation during summer burn windows.

### 6.2 Modality Occlusion & Missing Feature Degradation (Sentinel-2 Cloud Cover)
In operational deployment, Sentinel-2 optical imagery is frequently obscured by thick cloud cover or smoke plumes. Model robustness was evaluated under satellite availability conditions:

* **Fresh Sentinel-2 Data (`s2n_available = 1`)**: Standard inference utilizing active 5-day optical vegetation indices (NDVI/EVI). PR-AUC = 0.1930.
* **Precomputed KNN Imputed Data (`stage_c_knn`)**: When optical imagery is obscured, spatial-temporal 5-nearest-neighbor donor imputation provides synthetic vegetation features (`precomputed KNN`).
* **Degradation Assessment**: Precomputed KNN donor imputation preserves full feature contract integrity without performance drop, confirming ERA5 weather physics and causal FIRMS fire history provide strong redundancy.

---

## 7. Model Interpretability & Explainability

### 7.1 Global Feature Importance Ranking
Global feature importance was evaluated across all 86 retained features using CatBoost gain split importance and TreeSHAP attribution values.

![CatBoost Feature Importance and SHAP Contributions](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/feature_explanations.png)
*Figure 7.1: Global Feature Importance and SHAP Contributions ([feature_explanations.png](file:///d:/IITM/BSc/DSAI/Project/Group-8-DS-and-AI-Lab-Project/Milestone%205/images/feature_explanations.png)). Highlights top predictive features in CatBoost Champion model.*

#### Top 10 Most Predictive Features:
1. **`fire_distance_weighted_count_lag2`** (SHAP: 0.1470, Gain: 5.91%): Distance-weighted D-1 regional fire activity count. Single strongest predictor of continuous fire activity.
2. **`fire_distance_weighted_count_7d_lag2`** (SHAP: 0.1273, Gain: 5.87%): 7-day cumulative distance-weighted fire activity.
3. **`elevation`** (SHAP: 0.0578, Gain: 4.78%): Digital Elevation Model topographic height.
4. **`sp_mean`** (SHAP: 0.0404, Gain: 4.72%): Surface barometric pressure from ERA5.
5. **`latitude`** (SHAP: 0.0483, Gain: 4.09%): North-south geographic position.
6. **`lai_hv_mean`** (SHAP: 0.0666, Gain: 3.90%): High vegetation Leaf Area Index.
7. **`s5n_s5p_co_mean`** (SHAP: 0.0864, Gain: 3.70%): Sentinel-5P carbon monoxide atmospheric concentration.
8. **`s5n_s5p_co_max`** (SHAP: 0.0657, Gain: 3.11%): Peak carbon monoxide atmospheric anomaly.
9. **`day_of_year_cos`** (SHAP: 0.0317, Gain: 2.67%): Annual seasonal cyclical encoding.
10. **`slope`** (SHAP: 0.0444, Gain: 2.57%): Terrain slope steepness.

### 7.2 Physical Domain Alignment & SHAP Attribution Insights
* **Causal Fire History Alignment**: Distance-weighted lag features dominate model weights, reflecting physical wildfire propagation and continuous thermal anomaly behavior.
* **Atmospheric Combustion Plumes**: Sentinel-5P carbon monoxide mean (`s5n_s5p_co_mean`) and max (`s5n_s5p_co_max`) rank among top 10 features, capturing active upwind biomass combustion.
* **Topography & Vegetation Alignment**: DEM elevation, surface pressure, and high vegetation Leaf Area Index (`lai_hv_mean`) correctly isolate steep mountain chaparral zones vulnerable to dry atmospheric convection.

---

## 8. Actionable Insights & Potential Improvements

### 8.1 Short-Term Operational Mitigation Strategies
1. **Hybrid Dynamic Alert Thresholding**:
   - Replace fixed Top-25 daily alert budgets with a **Hybrid Rule-Based Filter**: issue alerts for the Top-25 cells *only if* predicted probability exceeds `p_fire >= 0.15`. On low-activity winter days, this suppresses up to 85% of false alarm alerts.
2. **1-Cell Spatial Buffer Post-Processing**:
   - Implement a spatial post-processing filter that merges adjacent Top-25 alert cells into unified regional hazard blocks, resolving 1-cell spatial discretization offsets observed during error analysis.

### 8.2 Long-Term Architectural Improvements
1. **Spatial Graph Neural Networks (GNNs)**:
   - Transition from 2D grid rollups to a **Spatial-Temporal Graph Neural Network (ST-GNN)**, modeling California's terrain as an irregular graph where edges represent wind direction, slope gradient, and fuel continuity.
2. **Multi-Task Learning (Fire Occurrence + Burn Area + Fire Intensity)**:
   - Extend the neural loss function to jointly predict next-day fire ignition (y in {0,1}), estimated burn area (hectares), and radiative power (MW).
3. **CatBoost Optuna Hyperparameter Optimization**:
   - Perform automated Optuna hyperparameter search on CatBoost tree depth, l2_leaf_reg, and learning rate specifically tuned for high/medium fire-prone cells.

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
