# Fair side-by-side model comparison

**Generated:** 2026-07-30T16:42:01.586492+00:00

## Metric contract

- Test period: 2025.
- Fire season: April through November.
- Identical population: 163,968 cell-days and 1,624 positives.
- PR-AUC: `sklearn.metrics.average_precision_score`.
- ROC-AUC: `sklearn.metrics.roc_auc_score`.
- Our score column: calibrated classifier probability `p_fire`.

## Fire-season model comparison

| Model | PR-AUC | ROC-AUC | PR vs colleague MLP |
|---|---:|---:|---:|
| Colleague C default LGBM | 0.070167 | 0.783829 | 0.81× |
| Colleague val-selected LGBM | 0.075698 | 0.792498 | 0.88× |
| Colleague highest-test LGBM | 0.079564 | 0.784619 | 0.92× |
| Colleague C default MLP | 0.086500 | 0.781300 | 1.00× |
| Our V1 classifier | 0.076443 | 0.792097 | 0.88× |
| Our V2 classifier | 0.184619 | 0.845922 | 2.13× |
| Our V3 classifier | 0.210718 | 0.845143 | 2.44× |
| Our V4 classifier | 0.216727 | 0.848589 | 2.51× |
| Our V5 base classifier (inherits V4) | 0.216727 | 0.848589 | 2.51× |

V4 improves PR-AUC over the colleague's best reported MLP by **0.1302 absolute**, or **2.51×** (**150.6%** relative improvement). ROC-AUC improves by **0.0673**.

## Serving-time comparison

Both pipelines enforce ERA5 through D−5. They are not identical input contracts because V2–V5 also use prior FIRMS observations.

| Source | Colleague | Our V1 | Our V2–V5 |
|---|---|---|---|
| ERA5 | D-5 | D-5 | D-5 |
| S2/S5P observation window | latest window_end <= D | latest window_end <= D in archive | latest window_end <= D; stale S2 >15d and S5P >2d masked |
| FIRMS as predictor | not used | not used | history through D-1 |
| FIRMS target | D+1 | D+1 | D+1 |

If D−1 FIRMS is available when forecasting on D, V4 is a real-time-capable, richer system and the 2.51× comparison is the appropriate best-system comparison.

If explicit fire-history predictors are disallowed, V1 is our closest existing reference. The colleague MLP scores 0.0865 versus V1 0.0764: a 0.0101 absolute or 13.2% relative advantage for the colleague model.

## Identical calendar buckets: colleague routed Stage-C LGBM vs our global V4

| Bucket | Rows | Positives | Colleague PR | Our V4 PR | PR ratio | Colleague ROC | Our V4 ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| fire_season | 163,968 | 1,624 | 0.070167 | 0.216727 | 3.09× | 0.783829 | 0.848589 |
| jan | 20,832 | 260 | 0.024580 | 0.114602 | 4.66× | 0.717187 | 0.864992 |
| feb | 18,816 | 115 | 0.013882 | 0.053682 | 3.87× | 0.731525 | 0.778683 |
| mar | 20,832 | 158 | 0.027895 | 0.051227 | 1.84× | 0.759196 | 0.828453 |
| dec | 20,832 | 118 | 0.008302 | 0.043773 | 5.27× | 0.638302 | 0.766105 |

## Monthly fire-season comparison

| Month | Positives | Colleague PR | Our V4 PR | PR ratio | Colleague ROC | Our V4 ROC |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 177 | 0.040444 | 0.083665 | 2.07× | 0.805456 | 0.845478 |
| 6 | 154 | 0.038575 | 0.048110 | 1.25× | 0.790808 | 0.813969 |
| 7 | 233 | 0.207567 | 0.439975 | 2.12× | 0.831544 | 0.863711 |
| 8 | 198 | 0.091714 | 0.327386 | 3.57× | 0.758019 | 0.867043 |
| 9 | 217 | 0.111315 | 0.387255 | 3.48× | 0.804373 | 0.877945 |
| 10 | 229 | 0.053373 | 0.074638 | 1.40× | 0.780133 | 0.839647 |
| 11 | 165 | 0.027302 | 0.056414 | 2.07× | 0.676221 | 0.781339 |

## Interpretation limits

- The colleague MLP metrics are available only to four decimal places.
- The colleague report does not provide full-period Recall@25, so daily alert-budget recall cannot be compared yet.
- Both teams have inspected 2025 repeatedly; these are descriptive experiment comparisons, not a fresh production holdout.
- V4's advantage depends on D−1 FIRMS history being available on decision day D.
