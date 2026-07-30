# Wildfire Forecasting Experiments: V1–V5

**Generated:** 2026-07-30T15:25:27.799987+00:00

**Dataset:** `california_numerical_nextday_wildfire_lag5`

**Audit status:** `conditionally_ready`

## Executive summary

Five isolated model versions were trained from the local archive without repeating GCS downloads. V2 produced the largest improvement by adding strictly lagged fire-history features. V3 added directional spread features and improved ranking further. V4 directly optimized Recall@25 with longer, regularized training and score blending. V5 tested hard-negative reranking but provided negligible additional gain.

The best practical model is **V4**. V5 improves 2025 descriptive Recall@25 by only one captured positive while reducing Recall@50. Strict Recall@25=50% was not reached; V4 reaches 50% recall at K=53.

## Data

**Unit:** california_era5_0.25_degree_land_cell_x_forecast_day

**Time contract:** y_fire is the FIRMS outcome on D+1; ERA5 predictors end at D-5.

**Coverage:** 2019-01-01 through 2025-12-31

**Grid:** 672 cells × 2557 days = 1,718,304 rows

**Total positives:** 21,615 (1.258%)

| Split | Label years | Rows | Positives | Positive rate |
|---|---|---:|---:|---:|
| train | 2019–2022 | 981,792 | 13,804 | 1.406% |
| val | 2023–2024 | 491,232 | 5,536 | 1.127% |
| test | 2025–2025 | 245,280 | 2,275 | 0.928% |

The row count overstates the independent sample size: cells on the same day are spatially correlated, consecutive days are temporally correlated, and one incident can create many positive cell-days. The effective evidence is the number of independent incidents and fire seasons.

### Sensor and quality limitations

- Sentinel-5P is entirely missing for 2021: 245,280 placeholder rows with zero measurements and zero availability.
- A no-S5P V2 ablation retained mean walk-forward PR-AUC 0.174741, showing that the main V2 gain does not depend on 2021 S5P.
- 7,392 late-2025 S2 rows exceeded the documented 15-day age limit; the cleaned training pipeline masks stale measurements.
- 6,720 training S5P rows in 2020 exceeded the two-day limit and were masked.
- Small negative soil values were clipped and two constant S5P standard-deviation features were removed.

## Selected architectures and parameters

| Version | Selected architecture | Features | Rounds | Calibration |
|---|---|---:|---|---|
| V1 | Global Stage C LightGBM; MLP tested and rejected | 61 | 84 | IsotonicRegression |
| V2 | Leakage-safe global LightGBM | 99 | 128 | IsotonicRegression |
| V3 | Global direction-aware LightGBM; mixture and LambdaRank tested | 113 | 73 | PlattCalibrator |
| V4 | Regularized classifier + LambdaRank; 50/50 daily rank blend | 113 | classifier=248; ranker=221 | PlattCalibrator |
| V5 | Frozen V4 retrieval + top-75 hard-negative LambdaRank | 119 | 82 | Inherited V4 Platt probabilities |

### Key parameters

- **V1:** learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85, lambda_l2=0.0, scale_pos_weight=70.1237
- **V2:** learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85, lambda_l2=0.0, scale_pos_weight=71.8968
- **V3:** learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85, lambda_l2=0.0, scale_pos_weight=1.0
- **V4:** learning_rate=0.025, num_leaves=31, min_data_in_leaf=75, feature_fraction=0.9, bagging_fraction=0.9, lambda_l1=0.2, lambda_l2=3.0, scale_pos_weight=1.0
- **V5:** learning_rate=0.04, num_leaves=31, min_data_in_leaf=30, lambda_l2=2.0

Additional architecture experiments:

- V1 MLP `[128, 64]`, 20 epochs: 2023 PR-AUC 0.068074; rejected.
- V2 seasonal router: mean PR-AUC 0.178234; rejected in favor of global.
- V3 continuation/ignition mixture: mean PR-AUC 0.192672; rejected.
- V3 daily LambdaRank: mean PR-AUC 0.180420; rejected.
- V4 best pure PR-AUC candidate (`extra_trees_63`): 0.200190; not selected because Recall@25 was lower.
- V5 top-75 reranker: mean Recall@25 42.97%; negligible gain over V4.

## Forward-validation results

| Version | Protocol | Mean PR-AUC | Minimum PR-AUC | Mean Recall@25 | Worst Recall@25 | Mean Recall@50 |
|---|---|---:|---:|---:|---:|---:|
| V1 | Single 2023 tune year | 0.097380 | 0.097380 | 29.88% | 29.88% | — |
| V2 | Walk-forward 2022–2024 | 0.182239 | 0.161495 | 41.58% | 39.13% | — |
| V3 | Walk-forward 2022–2024 | 0.193298 | 0.183773 | 42.03% | 39.57% | — |
| V4 | Recall@25 walk-forward 2022–2024 | 0.197967 | 0.181969 | 42.85% | 40.25% | 55.00% |
| V5 | Forward stacked reranking 2022–2024 | 0.168388 | 0.142395 | 42.97% | 40.69% | 54.58% |

V1 is not directly comparable to later walk-forward means because it selected on 2023 only. V4/V5 directly optimize alert-budget recall, whereas earlier versions selected primarily on PR-AUC.

## 2025 result comparison

Probability metrics are recomputed from each version's saved raw and calibrated classifier probabilities. Alert-budget metrics are recomputed with that version's selected alert score.

| Version | Raw PR-AUC | Calibrated PR-AUC | ROC-AUC | Brier | P@25 | Recall@25 | Recall@50 | K for 50% recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.068515 | 0.062542 | 0.790728 | 0.008945 | 6.3014% | 25.2747% | 36.5275% | 99 |
| V2 | 0.159282 | 0.150528 | 0.838086 | 0.008396 | 9.1726% | 36.7912% | 47.3846% | 59 |
| V3 | 0.170424 | 0.170424 | 0.837402 | 0.008330 | 9.4356% | 37.8462% | 48.3077% | 56 |
| V4 | 0.175139 | 0.175139 | 0.841983 | 0.008307 | 9.5562% | 38.3297% | 48.8791% | 53 |
| V5 | 0.175139 | 0.175139 | 0.841983 | 0.008307 | 9.5671% | 38.3736% | 48.3956% | 56 |

V4 captures 872 of 2,275 positives at K=25. A 50% target requires 1,138 captures, leaving a gap of 266 correct alerts. At the fixed 9,125-alert annual budget, Precision@25 would need to increase from 9.56% to approximately 12.47%.

## Version-by-version findings

### V1 — Baseline

- Stage C LightGBM beat Stage A, Stage B, and the MLP.
- It established useful but limited discrimination: calibrated PR-AUC 0.0625 and Recall@25 25.27%.

### V2 — Causal history

- Added weather histories, calendar/geography, causal fire recency, cell rate, neighbor and statewide histories.
- Produced the largest improvement: calibrated PR-AUC 0.1505 and Recall@25 36.79%.
- The gain is dominated by persistence/continuation rather than pure ignition prediction.

### V3 — Direction-aware spread

- Added distance-weighted and wind-aligned neighborhood fire features plus dry/windy interactions.
- Unweighted global LightGBM beat the proposed mixture and ranker.
- Improved calibrated PR-AUC to 0.1704 and Recall@25 to 37.85%.

### V4 — Recall-specific tuning

- Increased the candidate maximum to 1,200 rounds but selected rounds using Recall@25 early stopping.
- Tested seven classifier configurations, four rankers, and three predeclared blends.
- Selected a 248-round regularized classifier and a 50/50 daily classifier/ranker blend.
- Best practical balance: PR-AUC 0.1751, Recall@25 38.33%, Recall@50 48.88%.

### V5 — Hard-negative reranking

- Generated forward-only base predictions for 2021–2024 and trained top-75/100/150 candidate rerankers.
- Added only one 2025 captured positive at K=25 and reduced Recall@50; therefore it should not replace V4.

## Data-leakage assessment

| Risk | Assessment | Status / action |
|---|---|---|
| Direct target leakage | `y_fire`, FIRMS counts, dates, IDs and outcome fields are excluded from direct model inputs. | Controlled |
| Weather look-ahead | Target is T=D+1; ERA5 and weather rolling windows end D−5. | Controlled |
| Fire-history look-ahead | Histories end T−2=D−1; forecast-day D and target-day D+1 outcomes are excluded. | Controlled, conditional on D−1 FIRMS availability at serving time |
| Rolling-window boundaries | Full-grid and mutation tests confirm future labels cannot change earlier features. | Controlled |
| Cross-year history | Early-year rows legitimately carry prior-year causal history; the split does not erase information available in production. | Controlled |
| Sentinel-5P 2021 | Missingness can act as a year/domain indicator, but it is not target leakage. No-S5P ablation remains strong. | Distribution-shift risk, not leakage |
| Correlated rows | 672 cells/day and repeated incident cell-days reduce effective sample size. | Statistical limitation, not leakage |
| Test-set reuse | Only V1's initial 2025 evaluation was fully untouched. V2–V5 were designed after earlier 2025 results were visible. | Evaluation leakage for cross-version 2025 comparison; treat V2–V5 2025 as descriptive |
| Calibration | V1/V2 use isotonic calibration; V3/V4 use order-preserving Platt calibration on 2024. | Controlled; calibration-year metrics are in-sample |

There is **no detected feature/label leakage** under the stated serving contract. The material leakage concern is evaluation reuse: later versions cannot use 2025 as proof of unseen generalization.

## Conclusions and recommendation

1. Use **V4** as the retained model; keep V1–V5 artifacts for the experiment record.
2. Do not claim Recall@25=50%. With current inputs, the supported operating point is approximately Recall@53=50%.
3. Freeze all architecture choices before evaluating a new 2026 or external-geography holdout.
4. Further tuning of the same variables is unlikely to provide a large gain and risks overfitting.
5. The next high-value inputs are forecast-day/target-day weather forecasts, lightning, human-access/roads/powerlines/WUI, live fuel moisture and drought trajectories, and explicit ignition/onset labels.

## Reproducibility

- V1–V5 use separate artifact directories.
- V2/V3 derived tables were created from the archive without GCS.
- All saved model predictions were reproduced from fresh bundle loads.
- All artifact manifests and dataset hashes passed verification.
- The full project suite contains 30 passing tests.
