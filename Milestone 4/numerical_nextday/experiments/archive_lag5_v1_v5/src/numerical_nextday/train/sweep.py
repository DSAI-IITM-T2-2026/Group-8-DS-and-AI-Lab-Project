from __future__ import annotations

from typing import Any

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "lgbm_lr_03": {"learning_rate": 0.03},
    "lgbm_lr_10": {"learning_rate": 0.10},
    "lgbm_leaves_31": {"num_leaves": 31},
    "lgbm_leaves_127": {"num_leaves": 127},
    "lgbm_minleaf_20": {"min_data_in_leaf": 20},
    "lgbm_ff_07": {"feature_fraction": 0.70},
    "lgbm_bf_07": {"bagging_fraction": 0.70},
    "lgbm_l2_1": {"lambda_l2": 1.0},
    "lgbm_l2_5": {"lambda_l2": 5.0},
    "lgbm_no_spw": {"scale_pos_weight": 1.0},
    "lgbm_combo_regularized": {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        "lambda_l2": 5.0,
    },
}
