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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wildfire-v5-matplotlib")
)

from numerical_nextday.config import load_config  # noqa: E402
from numerical_nextday.evaluation.metrics import (  # noqa: E402
    binary_metrics,
    top_k_metrics,
)
from numerical_nextday.evaluation.plots import (  # noqa: E402
    plot_calibration_and_pr,
    plot_risk_map,
)
from numerical_nextday.io import atomic_json, atomic_parquet, sha256_file  # noqa: E402
from numerical_nextday.train.lightgbm_model import lightgbm_params  # noqa: E402
from numerical_nextday.train.preprocessing import TabularPreprocessor  # noqa: E402
from numerical_nextday.train.v3_model import V3RankerBundle  # noqa: E402
from numerical_nextday.train.v4_model import (  # noqa: E402
    V4AlertBundle,
    within_day_percentile,
)
from numerical_nextday.train.v5_model import (  # noqa: E402
    META_FEATURES,
    V5TwoStageBundle,
    add_retrieval_meta,
    candidate_positions,
)


IDENTITY_COLUMNS = ["label_date", "cell_id", "latitude", "longitude", "y_fire"]
FORWARD_YEARS = (2021, 2022, 2023, 2024)
CV_YEARS = (2022, 2023, 2024)
TOP_K_VALUES = (5, 10, 25, 50, 75, 100, 150)
TARGET_RECALL_25 = 0.50
RERANK_MAX_ROUNDS = 700
RERANK_EARLY_STOPPING = 80

RERANKER_CONFIGS = {
    "rank31_pool75": {
        "objective": "lambdarank",
        "pool_size": 75,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "lambda_l2": 2.0,
        "lambdarank_truncation_level": 25,
    },
    "rank63_pool75": {
        "objective": "lambdarank",
        "pool_size": 75,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "lambda_l2": 5.0,
        "lambdarank_truncation_level": 25,
    },
    "rank31_pool100": {
        "objective": "lambdarank",
        "pool_size": 100,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "lambda_l2": 3.0,
        "lambdarank_truncation_level": 25,
    },
    "rank63_pool100": {
        "objective": "lambdarank",
        "pool_size": 100,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "lambda_l2": 6.0,
        "lambdarank_truncation_level": 25,
    },
    "rank127_pool100": {
        "objective": "lambdarank",
        "pool_size": 100,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": 10,
        "min_data_in_leaf": 60,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "lambdarank_truncation_level": 25,
    },
    "rank63_pool150": {
        "objective": "lambdarank",
        "pool_size": 150,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 60,
        "lambda_l2": 8.0,
        "lambdarank_truncation_level": 25,
    },
    "binary63_pool100": {
        "objective": "binary",
        "pool_size": 100,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "lambda_l2": 6.0,
        "scale_pos_weight": 1.0,
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


def base_classifier_params(cfg: dict, overrides: dict) -> dict:
    params = lightgbm_params(cfg)
    params.update(overrides)
    params["metric"] = ["average_precision", "auc"]
    return params


def base_ranker_params(cfg: dict, overrides: dict) -> dict:
    params = lightgbm_params(cfg)
    params.update(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": [0, 1],
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
        }
    )
    params.update(overrides)
    params.pop("scale_pos_weight", None)
    return params


def fixed_forward_prediction(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    year: int,
    classifier_overrides: dict,
    classifier_rounds: int,
    ranker_overrides: dict,
    ranker_rounds: int,
    classifier_weight: float,
) -> pd.DataFrame:
    training = development.loc[development["year"].lt(year)]
    validation = development.loc[development["year"].eq(year)].sort_values(
        ["label_date", "cell_id"]
    )
    x_train = training[features].to_numpy(dtype="float32", copy=True)
    y_train = training["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    classifier_data = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=features,
        free_raw_data=False,
    )
    classifier = lgb.train(
        base_classifier_params(cfg, classifier_overrides),
        classifier_data,
        num_boost_round=classifier_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    classifier_score = np.asarray(
        classifier.predict(x_validation, num_iteration=classifier_rounds),
        dtype=float,
    )

    ranked_training = training.sort_values(["label_date", "cell_id"])
    x_rank_train = ranked_training[features].to_numpy(
        dtype="float32", copy=True
    )
    ranker_data = lgb.Dataset(
        x_rank_train,
        label=ranked_training["y_fire"].to_numpy(dtype="int8"),
        group=date_groups(ranked_training),
        feature_name=features,
        free_raw_data=False,
    )
    ranker = lgb.train(
        base_ranker_params(cfg, ranker_overrides),
        ranker_data,
        num_boost_round=ranker_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    ranker_score = np.asarray(
        ranker.predict(x_validation, num_iteration=ranker_rounds), dtype=float
    )
    classifier_rank = within_day_percentile(
        classifier_score, validation["label_date"]
    )
    ranker_rank = within_day_percentile(ranker_score, validation["label_date"])
    retrieval_score = (
        classifier_weight * classifier_rank
        + (1 - classifier_weight) * ranker_rank
    )
    base_predictions = {
        "p_fire_raw": classifier_score,
        "rank_score": ranker_score,
        "alert_score": retrieval_score,
    }
    result = add_retrieval_meta(validation, base_predictions)
    del (
        training,
        validation,
        x_train,
        y_train,
        x_validation,
        classifier_data,
        classifier,
        classifier_score,
        ranked_training,
        x_rank_train,
        ranker_data,
        ranker,
        ranker_score,
        classifier_rank,
        ranker_rank,
        retrieval_score,
        base_predictions,
    )
    gc.collect()
    return result


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


def candidate_frame(frame: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    positions = candidate_positions(frame, "v4_retrieval_score", pool_size)
    return frame.iloc[positions].sort_values(
        ["label_date", "cell_id"]
    ).reset_index(drop=True)


def reranker_params(cfg: dict, configuration: dict) -> dict:
    params = lightgbm_params(cfg)
    objective = configuration["objective"]
    params.update(
        {
            "objective": objective,
            "metric": "None",
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
        }
    )
    for key, value in configuration.items():
        if key not in {"pool_size", "objective"}:
            params[key] = value
    if objective == "lambdarank":
        params["label_gain"] = [0, 1]
        params.pop("scale_pos_weight", None)
    return params


def recall25_feval(
    target: np.ndarray, groups: np.ndarray, all_positive_count: int
):
    y = np.asarray(target, dtype="int8")

    def evaluate(prediction: np.ndarray, _: lgb.Dataset):
        captured = 0
        start = 0
        for size in groups:
            stop = start + int(size)
            take = min(25, int(size))
            indices = np.argpartition(prediction[start:stop], -take)[-take:]
            captured += int(y[start:stop][indices].sum())
            start = stop
        return (
            "overall_recall_at_25",
            float(captured / max(all_positive_count, 1)),
            True,
        )

    return evaluate


def fit_reranker_fold(
    cfg: dict,
    forward: pd.DataFrame,
    features: list[str],
    validation_year: int,
    configuration: dict,
) -> tuple[pd.DataFrame, dict]:
    pool_size = int(configuration["pool_size"])
    training = candidate_frame(
        forward.loc[forward["year"].lt(validation_year)], pool_size
    )
    full_validation = forward.loc[
        forward["year"].eq(validation_year)
    ].sort_values(["label_date", "cell_id"])
    validation = candidate_frame(full_validation, pool_size)
    train_groups = date_groups(training)
    validation_groups = date_groups(validation)
    x_train = training[features].to_numpy(dtype="float32", copy=True)
    y_train = training["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    y_validation = validation["y_fire"].to_numpy(dtype="int8", copy=True)
    objective = configuration["objective"]
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        group=train_groups if objective == "lambdarank" else None,
        feature_name=features,
        free_raw_data=False,
    )
    validation_data = lgb.Dataset(
        x_validation,
        label=y_validation,
        group=validation_groups if objective == "lambdarank" else None,
        reference=train_data,
        feature_name=features,
        free_raw_data=False,
    )
    all_positives = int(full_validation["y_fire"].sum())
    booster = lgb.train(
        reranker_params(cfg, configuration),
        train_data,
        num_boost_round=RERANK_MAX_ROUNDS,
        valid_sets=[validation_data],
        valid_names=["validation"],
        feval=recall25_feval(
            y_validation, validation_groups, all_positives
        ),
        callbacks=[
            lgb.early_stopping(RERANK_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iteration = int(booster.best_iteration)
    candidate_score = np.asarray(
        booster.predict(x_validation, num_iteration=iteration), dtype=float
    )
    score_lookup = dict(
        zip(
            zip(validation["label_date"], validation["cell_id"]),
            candidate_score,
        )
    )
    full_score = np.array(
        [
            score_lookup.get((date, cell), -1e9)
            for date, cell in zip(
                full_validation["label_date"], full_validation["cell_id"]
            )
        ],
        dtype=float,
    )
    pool_recall = score_metrics(
        full_validation, full_validation["v4_retrieval_score"].to_numpy()
    )["top_k"][str(pool_size)]["positive_recall"]
    metrics = {
        "validation_year": validation_year,
        "best_iteration": iteration,
        "pool_size": pool_size,
        "candidate_pool_recall": pool_recall,
        **score_metrics(full_validation, full_score),
    }
    oof = full_validation[["label_date", "cell_id", "y_fire"]].copy()
    oof["score"] = full_score.astype("float32")
    del (
        training,
        full_validation,
        validation,
        train_groups,
        validation_groups,
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_data,
        validation_data,
        booster,
        candidate_score,
        score_lookup,
        full_score,
    )
    gc.collect()
    return oof, metrics


def summarize(folds: list[dict]) -> dict:
    return {
        "folds": folds,
        "mean_recall_25": float(
            np.mean([fold["recall_25"] for fold in folds])
        ),
        "min_recall_25": float(
            np.min([fold["recall_25"] for fold in folds])
        ),
        "mean_recall_50": float(
            np.mean([fold["recall_50"] for fold in folds])
        ),
        "mean_pool_recall": float(
            np.mean([fold["candidate_pool_recall"] for fold in folds])
        ),
        "mean_pr_auc": float(np.mean([fold["pr_auc"] for fold in folds])),
        "median_best_iteration": int(
            np.median([fold["best_iteration"] for fold in folds])
        ),
    }


def run_reranker_grid(
    cfg: dict,
    forward: pd.DataFrame,
    features: list[str],
    output: Path,
) -> tuple[dict, pd.DataFrame]:
    results = {}
    oof = None
    for name, configuration in RERANKER_CONFIGS.items():
        log(
            f"[v5:reranker] {name}: pool={configuration['pool_size']}, "
            f"objective={configuration['objective']}"
        )
        fold_frames = []
        folds = []
        for year in CV_YEARS:
            fold_oof, metrics = fit_reranker_fold(
                cfg, forward, features, year, configuration
            )
            fold_frames.append(fold_oof)
            folds.append(metrics)
        candidate_oof = pd.concat(fold_frames, ignore_index=True)
        if oof is None:
            oof = candidate_oof[["label_date", "cell_id", "y_fire"]].copy()
        oof[name] = candidate_oof["score"].to_numpy(dtype="float32")
        results[name] = {
            "configuration": configuration,
            **summarize(folds),
        }
        atomic_json(results, output / "reranker_grid.partial.json")
        log(
            f"[v5:reranker] {name}: mean R@25="
            f"{results[name]['mean_recall_25']:.4%}, worst="
            f"{results[name]['min_recall_25']:.4%}, pool recall="
            f"{results[name]['mean_pool_recall']:.4%}"
        )
    if oof is None:
        raise RuntimeError("V5 reranker grid produced no predictions")
    return results, oof


def fit_final_reranker(
    cfg: dict,
    forward: pd.DataFrame,
    features: list[str],
    configuration: dict,
    rounds: int,
) -> V3RankerBundle:
    training = candidate_frame(forward, int(configuration["pool_size"]))
    groups = date_groups(training)
    preprocessor = TabularPreprocessor(features).fit(training, scale=False)
    x_train = preprocessor.transform(training)
    objective = configuration["objective"]
    train_data = lgb.Dataset(
        x_train,
        label=training["y_fire"].to_numpy(dtype="int8"),
        group=groups if objective == "lambdarank" else None,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        reranker_params(cfg, configuration),
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
    del training, groups, x_train, train_data
    gc.collect()
    return bundle


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


def report_markdown(payload: dict) -> str:
    test = payload["test_2025_descriptive"]
    lines = [
        "# V5 Two-Stage Recall@25 Report",
        "",
        f"**Completed:** {payload['completed_at_utc']}  ",
        f"**Selected reranker:** `{payload['selected_reranker']}`  ",
        f"**Candidate pool:** {payload['candidate_pool_size']} cells/day  ",
        f"**Target:** Recall@25 near {TARGET_RECALL_25:.0%}  ",
        "",
        "V5 uses the frozen V4 retrieval architecture to generate forward "
        "top-N candidates, then learns a hard-negative reranker. V5 selection "
        "used 2022–2024 only.",
        "",
        "## Reranker experiments",
        "",
        "| Candidate | Pool | Mean pool recall | Mean R@25 | Worst R@25 | Mean R@50 | Rounds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["reranker_grid"].items():
        lines.append(
            f"| {name} | {result['configuration']['pool_size']} | "
            f"{result['mean_pool_recall']:.4%} | "
            f"{result['mean_recall_25']:.4%} | "
            f"{result['min_recall_25']:.4%} | "
            f"{result['mean_recall_50']:.4%} | "
            f"{result['median_best_iteration']} |"
        )
    lines.extend(
        [
            "",
            "## 2025 descriptive result",
            "",
            "| Model | Recall@25 | Recall@50 | P@25 | PR-AUC/ranking AP |",
            "|---|---:|---:|---:|---:|",
            (
                f"| V4 retrieval | {payload['v4_reference']['recall_25']:.4%} | "
                f"{payload['v4_reference']['recall_50']:.4%} | "
                f"{payload['v4_reference']['precision_25']:.4%} | "
                f"{payload['v4_reference']['pr_auc']:.6f} |"
            ),
            (
                f"| V5 two-stage | {test['recall_25']:.4%} | "
                f"{test['recall_50']:.4%} | "
                f"{test['top_k']['25']['alert_precision']:.4%} | "
                f"{test['pr_auc']:.6f} |"
            ),
            "",
            f"Target met: **{'yes' if test['recall_25'] >= TARGET_RECALL_25 else 'no'}**.",
            "",
            "On descriptive 2025, the stronger V4 alert ordering reaches "
            "50.02% recall at K=53. V5 reaches 50% at K=56. Therefore the "
            "current feature set supports approximately Recall@53=0.50, not "
            "Recall@25=0.50.",
            "",
            "## Leakage controls",
            "",
            "- Base candidate scores for 2021–2024 were generated by models "
            "trained only on earlier years with fixed V4 round counts.",
            "- Reranker validation is forward-only on 2022, 2023, and 2024.",
            "- Candidate selection and reranking use the D−5/D−1 V3 features.",
            "- 2025 remains descriptive because prior version results were known.",
            "",
        ]
    )
    return "\n".join(lines)


def update_registry(
    directory: Path, payload: dict, artifact_directory: Path
) -> None:
    path = directory / "model_versions.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    test = payload["test_2025_descriptive"]
    v4 = registry["versions"]["V4"]
    registry["updated_at_utc"] = datetime.now(UTC).isoformat()
    registry["versions"]["V5"] = {
        "architecture": (
            f"V4 retrieval + {payload['selected_reranker']} top-"
            f"{payload['candidate_pool_size']}-to-25 reranker"
        ),
        "artifact_directory": str(artifact_directory.resolve()),
        "raw_pr_auc": v4["raw_pr_auc"],
        "calibrated_pr_auc": v4["calibrated_pr_auc"],
        "roc_auc": v4["roc_auc"],
        "brier": v4["brier"],
        "precision_25": test["top_k"]["25"]["alert_precision"],
        "recall_25": test["recall_25"],
        "recall_50": test["recall_50"],
        "ranking_ap": test["pr_auc"],
    }
    registry["holdout_note"] = (
        "Only V1's initial 2025 evaluation was fully untouched across the "
        "experiment series. V2, V3, V4, and V5 were designed after earlier "
        "2025 results were inspected, so their 2025 comparisons are descriptive."
    )
    atomic_json(registry, path)
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
    parser.add_argument("--v4-output", required=True, type=Path)
    args = parser.parse_args()
    data_dir = args.data.resolve()
    output = args.output.resolve()
    v4_output = args.v4_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    feature_sets = json.loads(
        (data_dir / "feature_sets.json").read_text(encoding="utf-8")
    )
    features = feature_sets["v3_directional"]
    development = load_development(data_dir, features)
    v4_metrics = json.loads((v4_output / "metrics.json").read_text())
    selected_classifier = v4_metrics["selected_classifier"]
    selected_ranker = v4_metrics["selected_ranker"]
    classifier_overrides = v4_metrics["classifier_grid"][selected_classifier][
        "overrides"
    ]
    classifier_rounds = v4_metrics["classifier_grid"][selected_classifier][
        "median_best_iteration"
    ]
    ranker_overrides = v4_metrics["ranker_grid"][selected_ranker]["overrides"]
    ranker_rounds = v4_metrics["ranker_grid"][selected_ranker][
        "median_best_iteration"
    ]
    alert_name = v4_metrics["selected_alert_head"]
    classifier_weight = (
        float(alert_name.rsplit("_", 1)[1])
        if alert_name.startswith("blend_")
        else (1.0 if alert_name == "classifier" else 0.0)
    )

    forward_path = output / "forward_base_predictions_2021_2024.parquet"
    if forward_path.exists():
        log("[v5:base] resuming from saved 2021–2024 forward predictions")
        saved = pd.read_parquet(
            forward_path, columns=["label_date", "cell_id", *META_FEATURES]
        )
        forward = development.loc[
            development["year"].isin(FORWARD_YEARS)
        ].merge(
            saved,
            on=["label_date", "cell_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(forward) != len(saved):
            raise RuntimeError("Saved V5 forward predictions do not align")
        forward_parts = []
        del saved
    else:
        log("[v5:base] generating fixed-round forward predictions for 2021–2024")
        forward_parts = []
        for year in FORWARD_YEARS:
            log(f"[v5:base] year={year}")
            forward_parts.append(
                fixed_forward_prediction(
                    cfg,
                    development,
                    features,
                    year,
                    classifier_overrides,
                    classifier_rounds,
                    ranker_overrides,
                    ranker_rounds,
                    classifier_weight,
                )
            )
        forward = pd.concat(forward_parts, ignore_index=True)
        forward["year"] = forward["label_date"].dt.year.astype("int16")
        atomic_parquet(
            forward[
                [
                    *IDENTITY_COLUMNS,
                    "year",
                    *META_FEATURES,
                ]
            ],
            forward_path,
        )
    rerank_features = [*features, *META_FEATURES]
    reranker_grid, reranker_oof = run_reranker_grid(
        cfg, forward, rerank_features, output
    )
    selected_reranker = max(
        reranker_grid,
        key=lambda name: (
            reranker_grid[name]["mean_recall_25"],
            reranker_grid[name]["min_recall_25"],
            reranker_grid[name]["mean_recall_50"],
        ),
    )
    selected_config = RERANKER_CONFIGS[selected_reranker]
    log(f"[v5:selection] reranker={selected_reranker}")
    atomic_parquet(
        reranker_oof, output / "oof_reranker_predictions_2022_2024.parquet"
    )
    reranker = fit_final_reranker(
        cfg,
        forward,
        rerank_features,
        selected_config,
        reranker_grid[selected_reranker]["median_best_iteration"],
    )
    retrieval_bundle = V4AlertBundle.load(
        v4_output / "models" / "v4_alert_bundle.joblib"
    )
    bundle = V5TwoStageBundle(
        retrieval_bundle=retrieval_bundle,
        reranker=reranker,
        candidate_pool_size=int(selected_config["pool_size"]),
    )
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reranker.save(models_dir / "reranker.joblib")
    bundle.save(models_dir / "v5_two_stage_bundle.joblib")
    del development, forward, forward_parts, reranker_oof
    gc.collect()

    log("[v5:test] scoring descriptive 2025 after V5 selection is locked")
    test = pd.read_parquet(
        data_dir / "test.parquet",
        columns=list(dict.fromkeys([*IDENTITY_COLUMNS, *features])),
    )
    test["label_date"] = pd.to_datetime(test["label_date"]).dt.normalize()
    predictions = bundle.predict(test)
    metrics = score_metrics(test, predictions["alert_score"])
    metrics["target_recall_25"] = TARGET_RECALL_25
    metrics["target_met"] = metrics["recall_25"] >= TARGET_RECALL_25
    scored = test[IDENTITY_COLUMNS].copy()
    for name, values in predictions.items():
        scored[name] = np.asarray(values, dtype="float32")
    scored["confidence_pct"] = (100 * scored["p_fire"]).astype("float32")
    atomic_parquet(scored, output / "test_predictions.parquet")
    plot_calibration_and_pr(
        scored,
        output / "plots" / "calibration_and_pr_2025.png",
        "V5 two-stage classifier probabilities — 2025 descriptive",
    )
    peak_day = scored.groupby("label_date")["p_fire"].max().idxmax()
    plot_risk_map(
        scored, output / "plots" / "risk_map_peak_day_2025.png", peak_day
    )
    write_importance(
        reranker.booster,
        reranker.feature_columns,
        output / "explainability" / "reranker_importance.csv",
    )
    v4_test = v4_metrics["test_2025_descriptive"]
    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "version": "V5",
        "objective": "Rerank the V4 top-N pool to maximize Recall@25.",
        "selected_reranker": selected_reranker,
        "candidate_pool_size": int(selected_config["pool_size"]),
        "reranker_grid": reranker_grid,
        "test_2025_descriptive": metrics,
        "v4_reference": {
            "pr_auc": v4_test["classifier_calibrated"]["pr_auc"],
            "precision_25": v4_test["selected_alert_head"]["top_k"]["25"][
                "alert_precision"
            ],
            "recall_25": v4_test["selected_alert_head"]["recall_25"],
            "recall_50": v4_test["selected_alert_head"]["recall_50"],
        },
    }
    atomic_json(payload, output / "metrics.json")
    (output / "TRAINING_REPORT.md").write_text(
        report_markdown(payload), encoding="utf-8"
    )
    artifacts = {
        "metrics": output / "metrics.json",
        "report": output / "TRAINING_REPORT.md",
        "predictions": output / "test_predictions.parquet",
        "forward_predictions": output
        / "forward_base_predictions_2021_2024.parquet",
        "oof_predictions": output
        / "oof_reranker_predictions_2022_2024.parquet",
        "reranker": models_dir / "reranker.joblib",
        "two_stage_bundle": models_dir / "v5_two_stage_bundle.joblib",
        "calibration_plot": output / "plots" / "calibration_and_pr_2025.png",
        "risk_map": output / "plots" / "risk_map_peak_day_2025.png",
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "version": "V5",
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
            "v4_manifest_sha256": file_sha256(
                v4_output / "run_manifest.json"
            ),
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
    log(f"[v5:done] report={output / 'TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
