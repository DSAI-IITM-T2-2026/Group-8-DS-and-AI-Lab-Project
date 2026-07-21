#!/usr/bin/env python3
"""Train LightGBM baseline → calibrated confidence % + regional alerts."""

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

from src.assemble import feature_columns
from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--data-dir",
        default=None,
        help="Directory with train/val/test.parquet (default: config paths.output_dir)",
    )
    p.add_argument("--top-k", type=int, default=25, help="Top regions to write per day")
    return p.parse_args()


def _load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        out[name] = pd.read_parquet(path)
        logger.info("%s: %d rows, pos=%d", name, len(out[name]), int(out[name]["y_fire"].sum()))
    return out


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["paths"]["output_dir"])
    model_dir = data_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_splits(data_dir)
    # If train empty (short smoke after calendar cut), fall back to 70/15/15 by date
    if len(splits["train"]) == 0 or splits["train"]["y_fire"].sum() == 0:
        logger.warning("Train split empty/no positives — re-splitting by label_date quantiles")
        all_df = pd.concat(splits.values(), ignore_index=True)
        dates = np.sort(all_df["label_date"].unique())
        n = len(dates)
        if n < 5:
            raise RuntimeError("Not enough distinct label_dates to train. Widen --start/--end.")
        train_end = dates[int(0.70 * n) - 1]
        val_end = dates[int(0.85 * n) - 1]
        d = all_df["label_date"]
        splits = {
            "train": all_df.loc[d <= train_end].copy(),
            "val": all_df.loc[(d > train_end) & (d <= val_end)].copy(),
            "test": all_df.loc[d > val_end].copy(),
        }
        for k, v in splits.items():
            logger.info("re-split %-5s n=%d pos=%d", k, len(v), int(v["y_fire"].sum()))

    feat_path = data_dir / "metadata" / "feature_columns.json"
    if feat_path.exists():
        feature_cols = json.loads(feat_path.read_text())
    else:
        feature_cols = feature_columns(splits["train"])

    X_train = splits["train"][feature_cols]
    y_train = splits["train"]["y_fire"].astype(int)
    X_val = splits["val"][feature_cols]
    y_val = splits["val"]["y_fire"].astype(int)
    X_test = splits["test"][feature_cols]
    y_test = splits["test"]["y_fire"].astype(int)

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    scale_pos_weight = n_neg / n_pos

    mcfg = cfg["model"]
    params = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": mcfg["learning_rate"],
        "num_leaves": mcfg["num_leaves"],
        "min_data_in_leaf": mcfg["min_data_in_leaf"],
        "feature_fraction": mcfg["feature_fraction"],
        "bagging_fraction": mcfg["bagging_fraction"],
        "bagging_freq": mcfg["bagging_freq"],
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": mcfg["random_seed"],
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=int(mcfg["num_boost_round"]),
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(int(mcfg["early_stopping_rounds"])),
            lgb.log_evaluation(50),
        ],
    )

    # Isotonic calibration on val so confidence % tracks empirical fire rate
    val_raw = np.asarray(booster.predict(X_val), dtype="float64")
    test_raw = np.asarray(booster.predict(X_test), dtype="float64")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, y_val.to_numpy())
    val_proba = calibrator.predict(val_raw)
    test_proba = calibrator.predict(test_raw)
    metrics = {
        "val": _metrics(y_val.to_numpy(), val_proba),
        "test": _metrics(y_test.to_numpy(), test_proba),
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": booster.best_iteration,
    }
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    joblib.dump(
        {
            "booster": booster,
            "calibrator": calibrator,
            "feature_columns": feature_cols,
            "predict": "p = calibrator.predict(booster.predict(X[feature_columns])); confidence_pct = 100*p",
        },
        model_dir / "baseline.joblib",
    )
    with (model_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Regional alerts on test: top-k cells per label_date by calibrated confidence
    test = splits["test"].copy()
    test["confidence_pct"] = (test_proba * 100.0).astype("float32")
    test["p_fire"] = test_proba.astype("float32")

    alerts = (
        test.sort_values(["label_date", "confidence_pct"], ascending=[True, False])
        .groupby("label_date", group_keys=False)
        .head(args.top_k)
    )
    alert_cols = [
        "label_date",
        "region",
        "cell_id",
        "latitude",
        "longitude",
        "confidence_pct",
        "p_fire",
        "y_fire",
    ]
    alerts[alert_cols].to_parquet(model_dir / "test_alerts_topk.parquet", index=False)
    alerts[alert_cols].to_csv(model_dir / "test_alerts_topk.csv", index=False)

    # Convenience: single-day example print
    if len(alerts):
        example_day = alerts["label_date"].value_counts().idxmax()
        day = alerts.loc[alerts["label_date"] == example_day].head(5)
        logger.info("Example alerts for %s:\n%s", example_day.date(), day[alert_cols].to_string(index=False))

    logger.info("Saved model + alerts under %s", model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
