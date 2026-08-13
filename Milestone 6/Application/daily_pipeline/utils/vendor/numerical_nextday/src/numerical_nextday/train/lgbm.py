"""LightGBM training: fire_season schedule + month models."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from numerical_nextday.config import shared_cache
from numerical_nextday.data.m3_imports import load_mvp_modules
from numerical_nextday.train.router import bucket_for_month, filter_bucket

logger = logging.getLogger(__name__)

LGBM_TRIALS = [
    ("C_default", {}),
    ("lgbm_lr_03", {"learning_rate": 0.03}),
    ("lgbm_lr_10", {"learning_rate": 0.10}),
    ("lgbm_leaves_31", {"num_leaves": 31}),
    ("lgbm_leaves_127", {"num_leaves": 127}),
    ("lgbm_minleaf_20", {"min_data_in_leaf": 20}),
    ("lgbm_ff_07", {"feature_fraction": 0.7}),
    ("lgbm_bf_07", {"bagging_fraction": 0.7}),
    ("lgbm_l2_1", {"lambda_l2": 1.0}),
    ("lgbm_l2_5", {"lambda_l2": 5.0}),
    ("lgbm_no_spw", {"scale_pos_weight": 1.0}),
]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def _feature_cols(df: pd.DataFrame, stage: str, cfg: dict | None = None) -> list[str]:
    if cfg is None:
        # ERA5/DEM heuristic without M3 import
        base = [
            c
            for c in df.columns
            if c
            not in (
                "cell_id",
                "latitude",
                "longitude",
                "feature_end_date",
                "label_date",
                "eo_asof_date",
                "y_fire",
                "firms_n_pixels",
                "firms_max_confidence",
                "region",
                "era5_lag_days",
                "s5p_2021_status",
            )
            and not c.endswith("_lag_days")
        ]
    else:
        mvp = load_mvp_modules(cfg)
        base = list(mvp["assemble"].feature_columns(df))
        if stage in ("B", "stage_b", "C", "stage_c"):
            base += [
                c
                for c in df.columns
                if c.startswith("s2n_") and c not in ("s2n_available", "s2n_lag_days")
            ]
        if stage in ("C", "stage_c"):
            base += [c for c in df.columns if c.startswith("s5n_") and c != "s5n_lag_days"]
    return [c for c in dict.fromkeys(base) if c in df.columns]


def _load_stage_splits(cfg: dict, stage: str) -> dict[str, pd.DataFrame]:
    cache = shared_cache(cfg)
    lag = int(cfg["task"].get("era5_lag_days", 5))
    root = cache / "lag0" if lag == 0 else cache
    d = root / stage
    out = {}
    for name in ("train", "val", "test"):
        p = d / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        out[name] = pd.read_parquet(p)
    return out


def train_lgbm(
    cfg: dict,
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    overrides: dict,
    out_dir: Path,
    experiment_id: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mcfg = dict(cfg["model"])
    mcfg.update({k: v for k, v in overrides.items() if k != "scale_pos_weight"})

    X_train = splits["train"][feature_cols]
    y_train = splits["train"]["y_fire"].astype(int)
    X_val = splits["val"][feature_cols]
    y_val = splits["val"]["y_fire"].astype(int)
    X_test = splits["test"][feature_cols]
    y_test = splits["test"]["y_fire"].astype(int)

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    spw = overrides.get("scale_pos_weight", n_neg / n_pos)

    params = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": mcfg["learning_rate"],
        "num_leaves": mcfg["num_leaves"],
        "min_data_in_leaf": mcfg["min_data_in_leaf"],
        "feature_fraction": mcfg["feature_fraction"],
        "bagging_fraction": mcfg["bagging_fraction"],
        "bagging_freq": mcfg["bagging_freq"],
        "scale_pos_weight": spw,
        "verbosity": -1,
        "seed": mcfg["random_seed"],
    }
    if "lambda_l2" in overrides:
        params["lambda_l2"] = overrides["lambda_l2"]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=int(mcfg["num_boost_round"]),
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(int(mcfg["early_stopping_rounds"])),
            lgb.log_evaluation(50),
        ],
    )
    val_raw = model.predict(X_val, num_iteration=model.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, y_val)
    test_raw = model.predict(X_test, num_iteration=model.best_iteration)
    test_cal = iso.predict(test_raw)
    val_cal = iso.predict(val_raw)

    metrics = {
        "experiment_id": experiment_id,
        "val": _metrics(y_val.to_numpy(), val_cal),
        "test": _metrics(y_test.to_numpy(), test_cal),
        "best_iteration": int(model.best_iteration or 0),
        "n_train": len(y_train),
        "n_pos_train": int(y_train.sum()),
        "overrides": overrides,
    }
    joblib.dump({"model": model, "iso": iso, "feature_cols": feature_cols}, out_dir / f"{experiment_id}.joblib")
    (out_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("%s test PR-AUC=%.4f ROC=%.4f", experiment_id, metrics["test"]["pr_auc"], metrics["test"]["roc_auc"])
    return metrics


def run_fire_season_schedule(cfg: dict, model_bucket: str = "fire_season") -> int:
    art = Path(cfg["paths"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    log_path = art / "experiments_log.csv"
    rows = []

    for stage_label, stage_dir in [("A", "stage_a"), ("B", "stage_b"), ("C", "stage_c")]:
        try:
            splits = _load_stage_splits(cfg, stage_dir)
        except FileNotFoundError:
            logger.warning("Missing %s — skip", stage_dir)
            continue
        for k in splits:
            splits[k] = filter_bucket(splits[k], model_bucket, cfg)
        if len(splits["train"]) == 0 or splits["train"]["y_fire"].sum() == 0:
            logger.warning("Empty train for %s/%s", stage_label, model_bucket)
            continue
        feat = _feature_cols(splits["train"], stage_label, cfg)
        exp_id = f"{stage_label}_default"
        metrics = train_lgbm(
            cfg,
            splits,
            feat,
            {},
            art / "models" / model_bucket,
            exp_id,
        )
        rows.append(
            {
                "experiment_id": exp_id,
                "stage": stage_label,
                "bucket": model_bucket,
                "test_pr_auc": metrics["test"]["pr_auc"],
                "test_roc_auc": metrics["test"]["roc_auc"],
                "val_pr_auc": metrics["val"]["pr_auc"],
            }
        )

    # HP on Stage C if present
    try:
        splits = _load_stage_splits(cfg, "stage_c")
        for k in splits:
            splits[k] = filter_bucket(splits[k], model_bucket, cfg)
        feat = _feature_cols(splits["train"], "C", cfg)
        best_pr = -1.0
        best_overrides: dict = {}
        for exp_id, overrides in LGBM_TRIALS[1:]:  # skip duplicate default
            metrics = train_lgbm(
                cfg,
                splits,
                feat,
                overrides,
                art / "models" / model_bucket,
                exp_id,
            )
            rows.append(
                {
                    "experiment_id": exp_id,
                    "stage": "C",
                    "bucket": model_bucket,
                    "test_pr_auc": metrics["test"]["pr_auc"],
                    "test_roc_auc": metrics["test"]["roc_auc"],
                    "val_pr_auc": metrics["val"]["pr_auc"],
                }
            )
            if metrics["val"]["pr_auc"] == metrics["val"]["pr_auc"] and metrics["val"]["pr_auc"] > best_pr:
                best_pr = metrics["val"]["pr_auc"]
                best_overrides = overrides
        (art / "models" / model_bucket / "best_overrides.json").write_text(json.dumps(best_overrides, indent=2))
    except FileNotFoundError:
        logger.warning("Stage C missing — skip HP")

    if rows:
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("Wrote LGBM experiments log → %s", log_path)

    # MLP secondary on Stage C (capped; failures must not block month models)
    try:
        from numerical_nextday.train.mlp import run_mlp_schedule

        logger.info("Starting MLP schedule")
        run_mlp_schedule(cfg)
    except Exception as e:
        logger.warning("MLP schedule skipped: %s", e)

    logger.info("fire_season schedule done; log=%s", log_path)
    return 0


def run_month_models(cfg: dict) -> int:
    art = Path(cfg["paths"]["artifacts_dir"])
    best_path = art / "models" / "fire_season" / "best_overrides.json"
    overrides = json.loads(best_path.read_text()) if best_path.exists() else {}
    splits = _load_stage_splits(cfg, "stage_c")
    feat = _feature_cols(splits["train"], "C", cfg)
    log_path = art / "experiments_log.csv"
    rows = []
    if log_path.exists():
        rows = list(csv.DictReader(log_path.open()))

    for month in cfg["model_buckets"]["month_models"]:
        bucket = bucket_for_month(month, cfg)
        if bucket == "fire_season":
            continue
        b_splits = {k: filter_bucket(v, bucket, cfg) for k, v in splits.items()}
        if len(b_splits["train"]) == 0 or int(b_splits["train"]["y_fire"].sum()) < 5:
            logger.warning("Too few positives for %s — winter_fallback", bucket)
            # Point to fire_season weights
            fb = art / "models" / bucket
            fb.mkdir(parents=True, exist_ok=True)
            (fb / "FALLBACK_fire_season.txt").write_text("winter_fallback: fire_season\n")
            continue
        exp_id = f"C_{bucket}"
        metrics = train_lgbm(
            cfg,
            b_splits,
            feat,
            overrides,
            art / "models" / bucket,
            exp_id,
        )
        rows.append(
            {
                "experiment_id": exp_id,
                "stage": "C",
                "bucket": bucket,
                "test_pr_auc": metrics["test"]["pr_auc"],
                "test_roc_auc": metrics["test"]["roc_auc"],
                "val_pr_auc": metrics["val"]["pr_auc"],
            }
        )

    if rows:
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 0


def run_lag0_ablation(cfg: dict) -> int:
    """Train fire_season LGBM on lag-0 Stage C (oracle ablation)."""
    cfg = dict(cfg)
    cfg["task"] = dict(cfg["task"])
    cfg["task"]["era5_lag_days"] = 0
    art = Path(cfg["paths"]["artifacts_dir"])
    out_dir = art / "models" / "fire_season_lag0"
    try:
        splits = _load_stage_splits(cfg, "stage_c")
    except FileNotFoundError:
        logger.error(
            "lag0 Stage C missing. Build with: "
            "--stage build_lag0_data (or stage_a_year/merge/s2_attach/s5p_attach with --era5-lag-days 0)"
        )
        return 1
    for k in splits:
        splits[k] = filter_bucket(splits[k], "fire_season", cfg)
    if len(splits["train"]) == 0:
        logger.error("Empty fire_season train for lag0")
        return 1
    feat = _feature_cols(splits["train"], "C", cfg)
    metrics = train_lgbm(cfg, splits, feat, {}, out_dir, "lgbm_era5_lag0")
    log_path = art / "experiments_log.csv"
    row = {
        "experiment_id": "lgbm_era5_lag0",
        "stage": "C",
        "bucket": "fire_season_lag0",
        "test_pr_auc": metrics["test"]["pr_auc"],
        "test_roc_auc": metrics["test"]["roc_auc"],
        "val_pr_auc": metrics["val"]["pr_auc"],
    }
    rows = list(csv.DictReader(log_path.open())) if log_path.exists() else []
    rows.append(row)
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("lag0 ablation done → %s", out_dir)
    return 0
