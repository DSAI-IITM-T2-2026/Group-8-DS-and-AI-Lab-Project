# Model Version Comparison

| Version | Architecture | Raw PR-AUC | Calibrated PR-AUC | Brier | P@25 | Recall@25 |
|---|---|---:|---:|---:|---:|---:|
| V1 | Stage C global LightGBM | 0.068515 | 0.062542 | 0.008945 | 6.3014% | 25.2747% |
| V2 | Leakage-safe global LightGBM | 0.159282 | 0.150528 | 0.008396 | 9.1726% | 36.7912% |
| V3 | global_directional_unweighted with Platt calibration; alert head=classifier_score | 0.170424 | 0.170424 | 0.008330 | 9.4356% | 37.8462% |
| V4 | leaves31_long_regularized + blend_classifier_0.50 Recall@25 head | 0.175139 | 0.175139 | 0.008307 | 9.5562% | 38.3297% |
| V5 | V4 retrieval + rank31_pool75 top-75-to-25 reranker | 0.175139 | 0.175139 | 0.008307 | 9.5671% | 38.3736% |

Only V1's initial 2025 evaluation was fully untouched across the experiment series. V2, V3, V4, and V5 were designed after earlier 2025 results were inspected, so their 2025 comparisons are descriptive.
