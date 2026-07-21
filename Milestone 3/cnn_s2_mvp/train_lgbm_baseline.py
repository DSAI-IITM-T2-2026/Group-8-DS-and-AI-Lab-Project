#!/usr/bin/env python3
"""LightGBM tabular baseline on the same CNN manifest splits (for comparison)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.sample import TABULAR_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lgbm_baseline")


def _metrics(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(cfg["paths"]["manifest"])
    feat_cols = [c for c in TABULAR_FEATURES if c in manifest.columns]

    train = manifest.loc[manifest["split"] == "train"]
    val = manifest.loc[manifest["split"] == "val"]
    test = manifest.loc[manifest["split"] == "test"]

    X_train, y_train = train[feat_cols], train["y_fire"].astype(int)
    X_val, y_val = val[feat_cols], val["y_fire"].astype(int)
    X_test, y_test = test[feat_cols], test["y_fire"].astype(int)

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    params = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "scale_pos_weight": n_neg / n_pos,
        "verbosity": -1,
        "seed": 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )

    val_raw = np.asarray(booster.predict(X_val), dtype=np.float64)
    test_raw = np.asarray(booster.predict(X_test), dtype=np.float64)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(val_raw, y_val.to_numpy())
    test_cal = cal.predict(test_raw)

    metrics = {
        "val": _metrics(y_val.to_numpy(), cal.predict(val_raw)),
        "test": _metrics(y_test.to_numpy(), test_cal),
        "best_iteration": booster.best_iteration,
    }
    with (model_dir / "lgbm_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(
        {"booster": booster, "calibrator": cal, "feature_columns": feat_cols},
        model_dir / "lgbm_baseline.joblib",
    )

    test_out = test.copy()
    test_out["p_fire"] = test_cal.astype("float32")
    test_out["confidence_pct"] = (test_cal * 100.0).astype("float32")
    top_k = int(cfg["alerts"]["top_k"])
    alerts = (
        test_out.sort_values(["label_date", "confidence_pct"], ascending=[True, False])
        .groupby("label_date", group_keys=False)
        .head(top_k)
    )
    cols = [
        "label_date",
        "region",
        "cell_id",
        "latitude",
        "longitude",
        "confidence_pct",
        "p_fire",
        "y_fire",
    ]
    if "region" not in alerts.columns:
        alerts["region"] = [
            f"cell:{cid} ({lat:.2f},{lon:.2f})"
            for cid, lat, lon in zip(alerts["cell_id"], alerts["latitude"], alerts["longitude"])
        ]
    alerts[cols].to_csv(model_dir / "lgbm_test_alerts_topk.csv", index=False)

    # Side-by-side comparison if CNN metrics exist
    cnn_metrics_path = model_dir / "metrics.json"
    comparison = {"lgbm": metrics}
    if cnn_metrics_path.exists():
        comparison["cnn"] = json.loads(cnn_metrics_path.read_text())
    with (model_dir / "comparison_metrics.json").open("w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison: %s", json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
