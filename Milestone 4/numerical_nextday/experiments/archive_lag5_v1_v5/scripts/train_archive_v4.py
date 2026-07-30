from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wildfire-v4-matplotlib")
)

from numerical_nextday.config import load_config  # noqa: E402
from numerical_nextday.evaluation.explain import (  # noqa: E402
    write_lightgbm_explanations,
)
from numerical_nextday.evaluation.metrics import (  # noqa: E402
    binary_metrics,
    top_k_metrics,
)
from numerical_nextday.evaluation.plots import (  # noqa: E402
    plot_calibration_and_pr,
    plot_risk_map,
)
from numerical_nextday.io import atomic_json, atomic_parquet, sha256_file  # noqa: E402
from numerical_nextday.train.calibration import fit_platt_calibrator  # noqa: E402
from numerical_nextday.train.lightgbm_model import (  # noqa: E402
    LightGBMBundle,
    lightgbm_params,
)
from numerical_nextday.train.preprocessing import TabularPreprocessor  # noqa: E402
from numerical_nextday.train.v3_model import V3RankerBundle  # noqa: E402
from numerical_nextday.train.v4_model import (  # noqa: E402
    V4AlertBundle,
    within_day_percentile,
)


IDENTITY_COLUMNS = ["label_date", "cell_id", "latitude", "longitude", "y_fire"]
CV_YEARS = (2022, 2023, 2024)
TOP_K_VALUES = (5, 10, 25, 50)
TARGET_RECALL_25 = 0.50
MAX_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 100

CLASSIFIER_CONFIGS = {
    "v3_recall_baseline": {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "scale_pos_weight": 1.0,
    },
    "leaves31_long_regularized": {
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_data_in_leaf": 75,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "scale_pos_weight": 1.0,
    },
    "leaves63_regularized": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 75,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 5.0,
        "scale_pos_weight": 1.0,
    },
    "leaves127_regularized": {
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": 10,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": 1.0,
    },
    "soft_positive_weight_2": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 75,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 5.0,
        "scale_pos_weight": 2.0,
    },
    "extra_trees_63": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 75,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "extra_trees": True,
        "scale_pos_weight": 1.0,
    },
    "goss_63": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 75,
        "feature_fraction": 0.85,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l2": 5.0,
        "boosting_type": "goss",
        "top_rate": 0.20,
        "other_rate": 0.10,
        "scale_pos_weight": 1.0,
    },
}

RANKER_CONFIGS = {
    "rank31_trunc25": {
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "lambdarank_truncation_level": 25,
    },
    "rank63_trunc50": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 75,
        "lambda_l2": 5.0,
        "lambdarank_truncation_level": 50,
    },
    "rank63_trunc100": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "lambda_l2": 8.0,
        "lambdarank_truncation_level": 100,
    },
    "rank127_trunc100": {
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": 10,
        "min_data_in_leaf": 100,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "lambdarank_truncation_level": 100,
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development(data_dir: Path, features: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features]))
    frame = pd.concat(
        [
            pd.read_parquet(data_dir / f"{split}.parquet", columns=columns)
            for split in ("train", "val")
        ],
        ignore_index=True,
    )
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    frame["year"] = frame["label_date"].dt.year.astype("int16")
    return frame


def date_groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("label_date", sort=False).size().to_numpy(dtype="int32")


def fast_recall_at_k(
    target: np.ndarray, score: np.ndarray, groups: np.ndarray, k: int
) -> float:
    positives = int(np.asarray(target, dtype=int).sum())
    if positives == 0:
        return 0.0
    if len(np.unique(groups)) == 1:
        width = int(groups[0])
        target_matrix = np.asarray(target).reshape(-1, width)
        score_matrix = np.asarray(score).reshape(-1, width)
        take = min(k, width)
        indices = np.argpartition(score_matrix, -take, axis=1)[:, -take:]
        captured = np.take_along_axis(target_matrix, indices, axis=1).sum()
        return float(captured / positives)
    captured = 0
    start = 0
    for size in groups:
        stop = start + int(size)
        take = min(k, int(size))
        indices = np.argpartition(score[start:stop], -take)[-take:]
        captured += int(np.asarray(target)[start:stop][indices].sum())
        start = stop
    return float(captured / positives)


def recall25_feval(target: np.ndarray, groups: np.ndarray):
    y = np.asarray(target, dtype="int8")

    def evaluate(prediction: np.ndarray, _: lgb.Dataset):
        return "recall_at_25", fast_recall_at_k(y, prediction, groups, 25), True

    return evaluate


def score_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict:
    scored = frame[["label_date", "y_fire"]].copy()
    scored["p_fire"] = np.asarray(score, dtype=float)
    result = {
        "pr_auc": float(average_precision_score(scored["y_fire"], score)),
        "roc_auc": float(roc_auc_score(scored["y_fire"], score)),
        "top_k": {
            str(k): top_k_metrics(scored, k) for k in TOP_K_VALUES
        },
    }
    result["recall_25"] = result["top_k"]["25"]["positive_recall"]
    result["recall_50"] = result["top_k"]["50"]["positive_recall"]
    return result


def base_params(cfg: dict, overrides: dict) -> dict:
    params = lightgbm_params(cfg)
    params.update(overrides)
    params["metric"] = "None"
    return params


def fit_classifier_fold(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    validation_year: int,
    overrides: dict,
) -> tuple[pd.DataFrame, dict]:
    training = development.loc[development["year"].lt(validation_year)]
    validation = development.loc[
        development["year"].eq(validation_year)
    ].sort_values(["label_date", "cell_id"])
    x_train = training[features].to_numpy(dtype="float32", copy=True)
    y_train = training["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    y_validation = validation["y_fire"].to_numpy(dtype="int8", copy=True)
    groups = date_groups(validation)
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=features,
        free_raw_data=False,
    )
    validation_data = lgb.Dataset(
        x_validation,
        label=y_validation,
        reference=train_data,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        base_params(cfg, overrides),
        train_data,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[validation_data],
        valid_names=["validation"],
        feval=recall25_feval(y_validation, groups),
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iteration = int(booster.best_iteration)
    prediction = np.asarray(
        booster.predict(x_validation, num_iteration=iteration), dtype=float
    )
    metrics = {
        "validation_year": validation_year,
        "best_iteration": iteration,
        **score_metrics(validation, prediction),
    }
    oof = validation[["label_date", "cell_id", "y_fire"]].copy()
    oof["score"] = prediction.astype("float32")
    del (
        training,
        validation,
        x_train,
        y_train,
        x_validation,
        y_validation,
        groups,
        train_data,
        validation_data,
        booster,
        prediction,
    )
    gc.collect()
    return oof, metrics


def ranker_params(cfg: dict, overrides: dict) -> dict:
    params = lightgbm_params(cfg)
    params.update(
        {
            "objective": "lambdarank",
            "metric": "None",
            "label_gain": [0, 1],
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
        }
    )
    params.update(overrides)
    params.pop("scale_pos_weight", None)
    return params


def fit_ranker_fold(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    validation_year: int,
    overrides: dict,
) -> tuple[pd.DataFrame, dict]:
    training = development.loc[
        development["year"].lt(validation_year)
    ].sort_values(["label_date", "cell_id"])
    validation = development.loc[
        development["year"].eq(validation_year)
    ].sort_values(["label_date", "cell_id"])
    x_train = training[features].to_numpy(dtype="float32", copy=True)
    y_train = training["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    y_validation = validation["y_fire"].to_numpy(dtype="int8", copy=True)
    train_groups = date_groups(training)
    validation_groups = date_groups(validation)
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        group=train_groups,
        feature_name=features,
        free_raw_data=False,
    )
    validation_data = lgb.Dataset(
        x_validation,
        label=y_validation,
        group=validation_groups,
        reference=train_data,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        ranker_params(cfg, overrides),
        train_data,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[validation_data],
        valid_names=["validation"],
        feval=recall25_feval(y_validation, validation_groups),
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iteration = int(booster.best_iteration)
    prediction = np.asarray(
        booster.predict(x_validation, num_iteration=iteration), dtype=float
    )
    metrics = {
        "validation_year": validation_year,
        "best_iteration": iteration,
        **score_metrics(validation, prediction),
    }
    oof = validation[["label_date", "cell_id", "y_fire"]].copy()
    oof["score"] = prediction.astype("float32")
    del (
        training,
        validation,
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_groups,
        validation_groups,
        train_data,
        validation_data,
        booster,
        prediction,
    )
    gc.collect()
    return oof, metrics


def summarize(folds: list[dict]) -> dict:
    return {
        "folds": folds,
        "mean_pr_auc": float(np.mean([fold["pr_auc"] for fold in folds])),
        "min_pr_auc": float(np.min([fold["pr_auc"] for fold in folds])),
        "mean_recall_25": float(
            np.mean([fold["recall_25"] for fold in folds])
        ),
        "min_recall_25": float(
            np.min([fold["recall_25"] for fold in folds])
        ),
        "mean_recall_50": float(
            np.mean([fold["recall_50"] for fold in folds])
        ),
        "median_best_iteration": int(
            np.median([fold["best_iteration"] for fold in folds])
        ),
    }


def run_grid(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    configurations: dict[str, dict],
    fitter,
    prefix: str,
    output: Path,
) -> tuple[dict, pd.DataFrame]:
    results = {}
    oof = None
    for name, overrides in configurations.items():
        log(f"[v4:{prefix}] {name}")
        fold_frames = []
        folds = []
        for year in CV_YEARS:
            fold_oof, metrics = fitter(
                cfg, development, features, year, overrides
            )
            fold_frames.append(fold_oof)
            folds.append(metrics)
        candidate_oof = pd.concat(fold_frames, ignore_index=True)
        if oof is None:
            oof = candidate_oof[["label_date", "cell_id", "y_fire"]].copy()
        elif not oof[["label_date", "cell_id"]].equals(
            candidate_oof[["label_date", "cell_id"]]
        ):
            raise RuntimeError(f"{prefix} OOF rows are misaligned")
        oof[name] = candidate_oof["score"].to_numpy(dtype="float32")
        results[name] = {
            "overrides": overrides,
            **summarize(folds),
        }
        atomic_json(results, output / f"{prefix}_grid.partial.json")
        log(
            f"[v4:{prefix}] {name}: mean R@25="
            f"{results[name]['mean_recall_25']:.4%}, R@50="
            f"{results[name]['mean_recall_50']:.4%}, PR="
            f"{results[name]['mean_pr_auc']:.6f}"
        )
    if oof is None:
        raise RuntimeError(f"{prefix} grid produced no OOF predictions")
    return results, oof


def select_classifier(results: dict) -> str:
    baseline = results["v3_recall_baseline"]
    eligible = [
        name
        for name, result in results.items()
        if result["mean_pr_auc"] >= baseline["mean_pr_auc"] * 0.98
        and result["min_recall_25"] >= baseline["min_recall_25"] - 0.015
    ]
    if not eligible:
        eligible = list(results)
    return max(
        eligible,
        key=lambda name: (
            results[name]["mean_recall_25"],
            results[name]["min_recall_25"],
            results[name]["mean_pr_auc"],
        ),
    )


def blend_experiments(
    classifier_oof: pd.DataFrame,
    classifier_name: str,
    ranker_oof: pd.DataFrame,
    ranker_name: str,
) -> tuple[dict, str]:
    if not classifier_oof[["label_date", "cell_id"]].equals(
        ranker_oof[["label_date", "cell_id"]]
    ):
        raise RuntimeError("Classifier/ranker OOF rows are misaligned")
    frame = classifier_oof[["label_date", "cell_id", "y_fire"]].copy()
    classifier_score = classifier_oof[classifier_name].to_numpy(dtype=float)
    ranker_score = ranker_oof[ranker_name].to_numpy(dtype=float)
    classifier_rank = within_day_percentile(
        classifier_score, frame["label_date"]
    )
    ranker_rank = within_day_percentile(ranker_score, frame["label_date"])
    experiments = {}
    candidates = {
        "classifier": classifier_score,
        "ranker": ranker_score,
    }
    for weight in (0.25, 0.50, 0.75):
        candidates[f"blend_classifier_{weight:.2f}"] = (
            weight * classifier_rank + (1 - weight) * ranker_rank
        )
    for name, score in candidates.items():
        folds = []
        for year in CV_YEARS:
            mask = frame["label_date"].dt.year.eq(year)
            metrics = score_metrics(frame.loc[mask], score[mask.to_numpy()])
            folds.append({"validation_year": year, **metrics})
        experiments[name] = summarize(
            [
                {
                    **fold,
                    "best_iteration": 0,
                }
                for fold in folds
            ]
        )
    baseline = experiments["classifier"]
    eligible = [
        name
        for name, result in experiments.items()
        if result["min_recall_25"] >= baseline["min_recall_25"] - 0.01
    ]
    selected = max(
        eligible,
        key=lambda name: (
            experiments[name]["mean_recall_25"],
            experiments[name]["min_recall_25"],
            experiments[name]["mean_recall_50"],
        ),
    )
    return experiments, selected


def fit_final_classifier(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    overrides: dict,
    rounds: int,
) -> LightGBMBundle:
    training = development.loc[development["year"].le(2023)]
    calibration = development.loc[development["year"].eq(2024)]
    preprocessor = TabularPreprocessor(features).fit(training, scale=False)
    x_train = preprocessor.transform(training)
    y_train = training["y_fire"].to_numpy(dtype="int8")
    params = base_params(cfg, overrides)
    params["metric"] = ["average_precision", "auc"]
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    raw_calibration = booster.predict(
        preprocessor.transform(calibration), num_iteration=rounds
    )
    calibrator = fit_platt_calibrator(
        raw_calibration,
        calibration["y_fire"].to_numpy(dtype="int8"),
        int(cfg["model_buckets"]["min_calibration_positives"]),
    )
    bundle = LightGBMBundle(
        booster=booster,
        calibrator=calibrator,
        preprocessor=preprocessor,
        feature_columns=features,
        stage="V4",
        model_bucket="global_recall25",
        params=params,
        random_seed=int(cfg["project"]["random_seed"]),
        best_iteration=rounds,
        calibration_note=getattr(calibrator, "reason", "platt"),
    )
    del training, calibration, x_train, y_train, train_data, raw_calibration
    gc.collect()
    return bundle


def fit_final_ranker(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    overrides: dict,
    rounds: int,
) -> V3RankerBundle:
    training = development.loc[development["year"].le(2023)].sort_values(
        ["label_date", "cell_id"]
    )
    preprocessor = TabularPreprocessor(features).fit(training, scale=False)
    x_train = preprocessor.transform(training)
    train_data = lgb.Dataset(
        x_train,
        label=training["y_fire"].to_numpy(dtype="int8"),
        group=date_groups(training),
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        ranker_params(cfg, overrides),
        train_data,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    bundle = V3RankerBundle(
        booster=booster,
        preprocessor=preprocessor,
        feature_columns=features,
        best_iteration=rounds,
    )
    del training, x_train, train_data
    gc.collect()
    return bundle


def alert_head_details(name: str) -> tuple[str, float]:
    if name == "classifier":
        return "classifier", 1.0
    if name == "ranker":
        return "ranker", 0.0
    return "blend", float(name.rsplit("_", 1)[1])


def write_importance(
    booster: lgb.Booster, features: list[str], destination: Path
) -> None:
    table = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": booster.feature_importance(
                importance_type="gain"
            ),
            "split_importance": booster.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)


def report_markdown(payload: dict, versions: dict) -> str:
    selected_classifier = payload["selected_classifier"]
    selected_ranker = payload["selected_ranker"]
    alert = payload["selected_alert_head"]
    test = payload["test_2025_descriptive"]
    lines = [
        "# V4 Recall@25 Optimization Report",
        "",
        f"**Completed:** {payload['completed_at_utc']}  ",
        f"**Target:** Recall@25 close to {TARGET_RECALL_25:.0%}  ",
        f"**Selected classifier:** `{selected_classifier}`  ",
        f"**Selected ranker:** `{selected_ranker}`  ",
        f"**Selected alert head:** `{alert}`  ",
        "",
        "All V4 model, round-count, and blend choices used walk-forward "
        "2022–2024 only. Each candidate allowed up to 1,200 boosting rounds "
        "with 100-round early stopping on Recall@25.",
        "",
        "## Classifier tuning",
        "",
        "| Candidate | Mean R@25 | Worst R@25 | Mean R@50 | Mean PR-AUC | Median rounds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["classifier_grid"].items():
        lines.append(
            f"| {name} | {result['mean_recall_25']:.4%} | "
            f"{result['min_recall_25']:.4%} | "
            f"{result['mean_recall_50']:.4%} | "
            f"{result['mean_pr_auc']:.6f} | "
            f"{result['median_best_iteration']} |"
        )
    lines.extend(
        [
            "",
            "## Ranker tuning",
            "",
            "| Candidate | Mean R@25 | Worst R@25 | Mean R@50 | Mean PR-AUC | Median rounds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in payload["ranker_grid"].items():
        lines.append(
            f"| {name} | {result['mean_recall_25']:.4%} | "
            f"{result['min_recall_25']:.4%} | "
            f"{result['mean_recall_50']:.4%} | "
            f"{result['mean_pr_auc']:.6f} | "
            f"{result['median_best_iteration']} |"
        )
    lines.extend(
        [
            "",
            "## Alert-head/blend selection",
            "",
            "| Alert head | Mean R@25 | Worst R@25 | Mean R@50 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in payload["blend_experiments"].items():
        lines.append(
            f"| {name} | {result['mean_recall_25']:.4%} | "
            f"{result['min_recall_25']:.4%} | "
            f"{result['mean_recall_50']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## 2025 descriptive comparison",
            "",
            "2025 is descriptive, not a new untouched holdout.",
            "",
            "| Version | PR-AUC | Brier | Recall@25 | Recall@50 | P@25 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("V3",):
        value = versions[name]
        lines.append(
            f"| {name} | {value['pr_auc']:.6f} | {value['brier']:.6f} | "
            f"{value['recall_25']:.4%} | {value['recall_50']:.4%} | "
            f"{value['precision_25']:.4%} |"
        )
    lines.append(
        f"| V4 | {test['classifier_calibrated']['pr_auc']:.6f} | "
        f"{test['classifier_calibrated']['brier']:.6f} | "
        f"{test['selected_alert_head']['recall_25']:.4%} | "
        f"{test['selected_alert_head']['recall_50']:.4%} | "
        f"{test['selected_alert_head']['top_k']['25']['alert_precision']:.4%} |"
    )
    lines.extend(
        [
            "",
            "## Overfitting controls",
            "",
            "- Selection used three forward validation years, never random row splits.",
            "- Recall@25 was optimized directly; PR-AUC and worst-year recall "
            "were eligibility guards.",
            "- Boosting rounds were selected by early stopping and then fixed.",
            "- Only three predefined blend weights were tested.",
            "- 2025 was evaluated only after V4 was locked and is labeled descriptive.",
            "",
        ]
    )
    return "\n".join(lines)


def update_registry(
    directory: Path, payload: dict, artifact_directory: Path
) -> None:
    registry_path = directory / "model_versions.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    test = payload["test_2025_descriptive"]
    registry["updated_at_utc"] = datetime.now(UTC).isoformat()
    registry["versions"]["V4"] = {
        "architecture": (
            f"{payload['selected_classifier']} + "
            f"{payload['selected_alert_head']} Recall@25 head"
        ),
        "artifact_directory": str(artifact_directory.resolve()),
        "raw_pr_auc": test["classifier_raw"]["pr_auc"],
        "calibrated_pr_auc": test["classifier_calibrated"]["pr_auc"],
        "roc_auc": test["classifier_calibrated"]["roc_auc"],
        "brier": test["classifier_calibrated"]["brier"],
        "precision_25": test["selected_alert_head"]["top_k"]["25"][
            "alert_precision"
        ],
        "recall_25": test["selected_alert_head"]["recall_25"],
        "recall_50": test["selected_alert_head"]["recall_50"],
    }
    registry["holdout_note"] = (
        "Only V1's initial 2025 evaluation was fully untouched across the "
        "experiment series. V2, V3, and V4 were designed after earlier 2025 "
        "results were inspected, so their 2025 comparisons are descriptive."
    )
    atomic_json(registry, registry_path)
    lines = [
        "# Model Version Comparison",
        "",
        "| Version | Architecture | Raw PR-AUC | Calibrated PR-AUC | Brier | P@25 | Recall@25 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in registry["versions"].items():
        lines.append(
            f"| {name} | {result['architecture']} | "
            f"{result['raw_pr_auc']:.6f} | "
            f"{result['calibrated_pr_auc']:.6f} | "
            f"{result['brier']:.6f} | "
            f"{result['precision_25']:.4%} | "
            f"{result['recall_25']:.4%} |"
        )
    lines.extend(["", registry["holdout_note"], ""])
    (directory / "MODEL_VERSION_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--v3-metrics", required=True, type=Path)
    args = parser.parse_args()
    data_dir = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    feature_sets = json.loads(
        (data_dir / "feature_sets.json").read_text(encoding="utf-8")
    )
    features = feature_sets["v3_directional"]
    log(f"[v4:load] reusing V3 data with {len(features)} features")
    development = load_development(data_dir, features)

    classifier_grid, classifier_oof = run_grid(
        cfg,
        development,
        features,
        CLASSIFIER_CONFIGS,
        fit_classifier_fold,
        "classifier",
        output,
    )
    selected_classifier = select_classifier(classifier_grid)
    log(f"[v4:selection] classifier={selected_classifier}")

    ranker_grid, ranker_oof = run_grid(
        cfg,
        development,
        features,
        RANKER_CONFIGS,
        fit_ranker_fold,
        "ranker",
        output,
    )
    selected_ranker = max(
        ranker_grid,
        key=lambda name: (
            ranker_grid[name]["mean_recall_25"],
            ranker_grid[name]["min_recall_25"],
        ),
    )
    log(f"[v4:selection] ranker={selected_ranker}")
    blend_results, selected_alert = blend_experiments(
        classifier_oof,
        selected_classifier,
        ranker_oof,
        selected_ranker,
    )
    atomic_json(blend_results, output / "blend_experiments.json")
    log(f"[v4:selection] alert_head={selected_alert}")
    oof = classifier_oof.copy()
    for column in ranker_oof.columns:
        if column not in {"label_date", "cell_id", "y_fire"}:
            oof[f"ranker_{column}"] = ranker_oof[column]
    atomic_parquet(oof, output / "oof_predictions_2022_2024.parquet")

    classifier = fit_final_classifier(
        cfg,
        development,
        features,
        CLASSIFIER_CONFIGS[selected_classifier],
        classifier_grid[selected_classifier]["median_best_iteration"],
    )
    ranker = fit_final_ranker(
        cfg,
        development,
        features,
        RANKER_CONFIGS[selected_ranker],
        ranker_grid[selected_ranker]["median_best_iteration"],
    )
    alert_head, classifier_weight = alert_head_details(selected_alert)
    bundle = V4AlertBundle(
        classifier=classifier,
        ranker=ranker,
        alert_head=alert_head,
        classifier_weight=classifier_weight,
    )
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(models_dir / "classifier.joblib")
    ranker.save(models_dir / "ranker.joblib")
    bundle.save(models_dir / "v4_alert_bundle.joblib")
    del development, classifier_oof, ranker_oof, oof
    gc.collect()

    log("[v4:test] scoring descriptive 2025 after V4 selection is locked")
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features]))
    test = pd.read_parquet(data_dir / "test.parquet", columns=columns)
    test["label_date"] = pd.to_datetime(test["label_date"]).dt.normalize()
    predictions = bundle.predict(test)
    classifier_raw = score_metrics(test, predictions["p_fire_raw"])
    classifier_calibrated = {
        **binary_metrics(test["y_fire"], predictions["p_fire"]),
        "top_k": {
            str(k): top_k_metrics(
                test[["label_date", "y_fire"]].assign(
                    p_fire=predictions["p_fire"]
                ),
                k,
            )
            for k in TOP_K_VALUES
        },
    }
    classifier_calibrated["recall_25"] = classifier_calibrated["top_k"]["25"][
        "positive_recall"
    ]
    classifier_calibrated["recall_50"] = classifier_calibrated["top_k"]["50"][
        "positive_recall"
    ]
    selected_alert_metrics = score_metrics(test, predictions["alert_score"])
    test_metrics = {
        "classifier_raw": classifier_raw,
        "classifier_calibrated": classifier_calibrated,
        "selected_alert_head": {
            "name": selected_alert,
            **selected_alert_metrics,
        },
        "target_recall_25": TARGET_RECALL_25,
        "target_met": selected_alert_metrics["recall_25"] >= TARGET_RECALL_25,
    }
    scored = test[IDENTITY_COLUMNS].copy()
    for name, values in predictions.items():
        scored[name] = np.asarray(values, dtype="float32")
    scored["confidence_pct"] = (100 * scored["p_fire"]).astype("float32")
    atomic_parquet(scored, output / "test_predictions.parquet")
    plot_calibration_and_pr(
        scored,
        output / "plots" / "calibration_and_pr_2025.png",
        "V4 Recall@25 classifier — 2025 descriptive",
    )
    peak_day = scored.groupby("label_date")["p_fire"].max().idxmax()
    plot_risk_map(
        scored, output / "plots" / "risk_map_peak_day_2025.png", peak_day
    )
    write_importance(
        classifier.booster,
        classifier.feature_columns,
        output / "explainability" / "classifier_importance.csv",
    )
    write_importance(
        ranker.booster,
        ranker.feature_columns,
        output / "explainability" / "ranker_importance.csv",
    )
    write_lightgbm_explanations(
        classifier,
        test,
        output / "explainability" / "classifier",
        max_rows=int(cfg["training"]["max_explain_rows"]),
    )

    v3 = json.loads(args.v3_metrics.read_text(encoding="utf-8"))
    v3_test = v3["test_2025_descriptive"]
    v3_predictions = pd.read_parquet(
        args.v3_metrics.parent / "test_predictions.parquet",
        columns=["label_date", "y_fire", "alert_score"],
    )
    v3_score = score_metrics(v3_predictions, v3_predictions["alert_score"])
    versions = {
        "V3": {
            "pr_auc": v3_test["classifier_calibrated"]["pr_auc"],
            "brier": v3_test["classifier_calibrated"]["brier"],
            "precision_25": v3_score["top_k"]["25"]["alert_precision"],
            "recall_25": v3_score["recall_25"],
            "recall_50": v3_score["recall_50"],
        }
    }
    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "version": "V4",
        "objective": "Maximize walk-forward daily Recall@25; target 0.50.",
        "selection_protocol": (
            "Up to 1,200 rounds; early stopping on Recall@25; selection on "
            "2022–2024 with PR-AUC and worst-year safeguards. 2025 descriptive."
        ),
        "selected_classifier": selected_classifier,
        "selected_ranker": selected_ranker,
        "selected_alert_head": selected_alert,
        "classifier_grid": classifier_grid,
        "ranker_grid": ranker_grid,
        "blend_experiments": blend_results,
        "test_2025_descriptive": test_metrics,
        "v3_reference": versions["V3"],
    }
    atomic_json(payload, output / "metrics.json")
    (output / "TRAINING_REPORT.md").write_text(
        report_markdown(payload, versions), encoding="utf-8"
    )
    artifacts = {
        "metrics": output / "metrics.json",
        "report": output / "TRAINING_REPORT.md",
        "predictions": output / "test_predictions.parquet",
        "oof_predictions": output / "oof_predictions_2022_2024.parquet",
        "classifier": models_dir / "classifier.joblib",
        "ranker": models_dir / "ranker.joblib",
        "alert_bundle": models_dir / "v4_alert_bundle.joblib",
        "calibration_plot": output / "plots" / "calibration_and_pr_2025.png",
        "risk_map": output / "plots" / "risk_map_peak_day_2025.png",
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "version": "V4",
        "software": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "input": {
            "metadata": json.loads(
                (data_dir / "metadata.json").read_text(encoding="utf-8")
            ),
            "feature_sets_sha256": sha256_file(data_dir / "feature_sets.json"),
        },
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    atomic_json(manifest, output / "run_manifest.json")
    update_registry(output.parent, payload, output)
    log(f"[v4:done] report={output / 'TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
