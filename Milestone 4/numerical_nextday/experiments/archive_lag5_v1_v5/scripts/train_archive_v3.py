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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wildfire-v3-matplotlib")
)

from numerical_nextday.config import load_config  # noqa: E402
from numerical_nextday.evaluation.explain import (  # noqa: E402
    write_lightgbm_explanations,
)
from numerical_nextday.evaluation.metrics import (  # noqa: E402
    binary_metrics,
    complete_metrics,
    top_k_metrics,
)
from numerical_nextday.evaluation.plots import (  # noqa: E402
    plot_calibration_and_pr,
    plot_risk_map,
)
from numerical_nextday.io import atomic_json, atomic_parquet, sha256_file  # noqa: E402
from numerical_nextday.train.calibration import (  # noqa: E402
    fit_platt_calibrator,
)
from numerical_nextday.train.lightgbm_model import (  # noqa: E402
    LightGBMBundle,
    lightgbm_params,
)
from numerical_nextday.train.preprocessing import TabularPreprocessor  # noqa: E402
from numerical_nextday.train.v3_model import (  # noqa: E402
    V3ClassifierBundle,
    V3RankerBundle,
)


IDENTITY_COLUMNS = ["label_date", "cell_id", "latitude", "longitude", "y_fire"]
ROUTER_COLUMN = "v3_recent_fire_context"
CV_YEARS = (2022, 2023, 2024)
TOP_K_VALUES = (5, 10, 25)
CANDIDATES = {
    "global_directional_weighted": {
        "architecture": "global",
        "feature_set": "v3_directional",
        "class_weight": True,
    },
    "global_directional_unweighted": {
        "architecture": "global",
        "feature_set": "v3_directional",
        "class_weight": False,
    },
    "mixture_directional_unweighted": {
        "architecture": "mixture",
        "feature_set": "v3_directional",
        "class_weight": False,
    },
    "mixture_directional_no_s5p": {
        "architecture": "mixture",
        "feature_set": "v3_directional_no_s5p",
        "class_weight": False,
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
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features, ROUTER_COLUMN]))
    train = pd.read_parquet(data_dir / "train.parquet", columns=columns)
    validation = pd.read_parquet(data_dir / "val.parquet", columns=columns)
    frame = pd.concat([train, validation], ignore_index=True)
    del train, validation
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    frame["year"] = frame["label_date"].dt.year.astype("int16")
    return frame


def classifier_params(
    cfg: dict, target: np.ndarray, class_weight: bool
) -> dict:
    params = lightgbm_params(cfg, {"num_leaves": 31})
    if class_weight:
        positives = max(int(target.sum()), 1)
        negatives = max(int(len(target) - positives), 1)
        params["scale_pos_weight"] = negatives / positives
    else:
        params["scale_pos_weight"] = 1.0
    return params


def scored_metrics(
    frame: pd.DataFrame, score: np.ndarray, include_probability: bool = True
) -> dict:
    scored = frame[["label_date", "y_fire"]].copy()
    scored["p_fire"] = np.asarray(score, dtype="float64")
    result = (
        complete_metrics(scored, 25)
        if include_probability
        else {
            "pr_auc": float(average_precision_score(scored["y_fire"], score)),
            "roc_auc": float(roc_auc_score(scored["y_fire"], score)),
        }
    )
    result["top_k"] = {
        str(k): top_k_metrics(scored, k) for k in TOP_K_VALUES
    }
    return result


def fit_binary_fold(
    cfg: dict,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    class_weight: bool,
) -> tuple[np.ndarray, int]:
    x_train = training[features].to_numpy(dtype="float32", copy=True)
    y_train = training["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    y_validation = validation["y_fire"].to_numpy(dtype="int8", copy=True)
    params = classifier_params(cfg, y_train, class_weight)
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
        params,
        train_data,
        num_boost_round=int(cfg["training"]["num_boost_round"]),
        valid_sets=[validation_data],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(
                int(cfg["training"]["early_stopping_rounds"]), verbose=False
            ),
            lgb.log_evaluation(0),
        ],
    )
    iteration = int(booster.best_iteration)
    probability = np.asarray(
        booster.predict(x_validation, num_iteration=iteration), dtype=float
    )
    del (
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_data,
        validation_data,
        booster,
    )
    gc.collect()
    return probability, iteration


def evaluate_candidate_fold(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    validation_year: int,
    architecture: str,
    class_weight: bool,
) -> dict:
    training = development.loc[development["year"].lt(validation_year)]
    validation = development.loc[development["year"].eq(validation_year)]
    rounds = {}
    bucket_metrics = {}
    if architecture == "global":
        probability, rounds["global"] = fit_binary_fold(
            cfg, training, validation, features, class_weight
        )
    else:
        expert_features = [
            column for column in features if column != ROUTER_COLUMN
        ]
        probability = np.empty(len(validation), dtype=float)
        for bucket, value in (("context", 1), ("ignition", 0)):
            train_part = training.loc[training[ROUTER_COLUMN].eq(value)]
            validation_part = validation.loc[validation[ROUTER_COLUMN].eq(value)]
            bucket_probability, rounds[bucket] = fit_binary_fold(
                cfg,
                train_part,
                validation_part,
                expert_features,
                class_weight,
            )
            positions = np.flatnonzero(
                validation[ROUTER_COLUMN].to_numpy(dtype=int) == value
            )
            probability[positions] = bucket_probability
            bucket_metrics[bucket] = {
                **binary_metrics(
                    validation_part["y_fire"].to_numpy(), bucket_probability
                ),
                "rows": len(validation_part),
            }
            del train_part, validation_part, bucket_probability
            gc.collect()
    metrics = scored_metrics(validation, probability)
    metrics.update(
        {
            "validation_year": validation_year,
            "train_rows": len(training),
            "validation_rows": len(validation),
            "best_iterations": rounds,
            "by_expert": bucket_metrics,
        }
    )
    del training, validation, probability
    gc.collect()
    return metrics


def summarize_candidate(folds: list[dict]) -> dict:
    pr = np.array([fold["pr_auc"] for fold in folds], dtype=float)
    result = {
        "folds": folds,
        "mean_pr_auc": float(pr.mean()),
        "min_pr_auc": float(pr.min()),
        "std_pr_auc": float(pr.std()),
        "mean_top_k_precision": {
            str(k): float(
                np.mean(
                    [
                        fold["top_k"][str(k)]["alert_precision"]
                        for fold in folds
                    ]
                )
            )
            for k in TOP_K_VALUES
        },
        "median_best_iterations": {},
    }
    buckets = sorted(
        {
            bucket
            for fold in folds
            for bucket in fold["best_iterations"]
        }
    )
    for bucket in buckets:
        result["median_best_iterations"][bucket] = int(
            np.median(
                [
                    fold["best_iterations"][bucket]
                    for fold in folds
                    if bucket in fold["best_iterations"]
                ]
            )
        )
    return result


def run_classifier_experiments(
    cfg: dict,
    development: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output: Path,
) -> dict:
    results = {}
    for name, candidate in CANDIDATES.items():
        features = feature_sets[candidate["feature_set"]]
        log(
            f"[v3:classifier] {name}: {candidate['architecture']}, "
            f"{len(features)} features"
        )
        folds = [
            evaluate_candidate_fold(
                cfg,
                development,
                features,
                year,
                candidate["architecture"],
                candidate["class_weight"],
            )
            for year in CV_YEARS
        ]
        results[name] = {
            **candidate,
            "feature_count": len(features),
            **summarize_candidate(folds),
        }
        atomic_json(results, output / "classifier_experiments.partial.json")
        log(
            f"[v3:classifier] {name}: mean PR-AUC="
            f"{results[name]['mean_pr_auc']:.6f}, mean P@25="
            f"{results[name]['mean_top_k_precision']['25']:.4%}"
        )
    return results


def ranker_parameters(cfg: dict) -> dict:
    params = lightgbm_params(cfg, {"num_leaves": 31})
    params.update(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10, 25],
            "lambdarank_truncation_level": 25,
            "label_gain": [0, 1],
        }
    )
    params.pop("scale_pos_weight", None)
    return params


def date_groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("label_date", sort=False).size().to_numpy(dtype="int32")


def fit_ranker_fold(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    validation_year: int,
) -> dict:
    training = development.loc[development["year"].lt(validation_year)].sort_values(
        ["label_date", "cell_id"]
    )
    validation = development.loc[
        development["year"].eq(validation_year)
    ].sort_values(["label_date", "cell_id"])
    x_train = training[features].to_numpy(dtype="float32", copy=True)
    y_train = training["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    y_validation = validation["y_fire"].to_numpy(dtype="int8", copy=True)
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        group=date_groups(training),
        feature_name=features,
        free_raw_data=False,
    )
    validation_data = lgb.Dataset(
        x_validation,
        label=y_validation,
        group=date_groups(validation),
        reference=train_data,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        ranker_parameters(cfg),
        train_data,
        num_boost_round=int(cfg["training"]["num_boost_round"]),
        valid_sets=[validation_data],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(
                int(cfg["training"]["early_stopping_rounds"]), verbose=False
            ),
            lgb.log_evaluation(0),
        ],
    )
    iteration = int(booster.best_iteration)
    score = np.asarray(
        booster.predict(x_validation, num_iteration=iteration), dtype=float
    )
    metrics = scored_metrics(validation, score, include_probability=False)
    metrics.update(
        {
            "validation_year": validation_year,
            "best_iteration": iteration,
            "train_rows": len(training),
            "validation_rows": len(validation),
        }
    )
    del (
        training,
        validation,
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_data,
        validation_data,
        booster,
        score,
    )
    gc.collect()
    return metrics


def run_ranker_experiment(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    output: Path,
) -> dict:
    log(f"[v3:ranker] daily LambdaRank with {len(features)} features")
    folds = [
        fit_ranker_fold(cfg, development, features, year) for year in CV_YEARS
    ]
    result = {
        "feature_count": len(features),
        "folds": folds,
        "mean_pr_auc": float(np.mean([fold["pr_auc"] for fold in folds])),
        "min_pr_auc": float(np.min([fold["pr_auc"] for fold in folds])),
        "mean_top_k_precision": {
            str(k): float(
                np.mean(
                    [
                        fold["top_k"][str(k)]["alert_precision"]
                        for fold in folds
                    ]
                )
            )
            for k in TOP_K_VALUES
        },
        "median_best_iteration": int(
            np.median([fold["best_iteration"] for fold in folds])
        ),
    }
    atomic_json(result, output / "ranker_experiment.json")
    log(
        f"[v3:ranker] mean P@25="
        f"{result['mean_top_k_precision']['25']:.4%}"
    )
    return result


def fit_final_lightgbm(
    cfg: dict,
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    rounds: int,
    class_weight: bool,
    bucket: str,
) -> LightGBMBundle:
    preprocessor = TabularPreprocessor(features).fit(training, scale=False)
    x_train = preprocessor.transform(training)
    y_train = training["y_fire"].to_numpy(dtype="int8")
    params = classifier_params(cfg, y_train, class_weight)
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
    raw_calibration = np.asarray(
        booster.predict(preprocessor.transform(calibration), num_iteration=rounds),
        dtype=float,
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
        stage="V3",
        model_bucket=bucket,
        params=params,
        random_seed=int(cfg["project"]["random_seed"]),
        best_iteration=rounds,
        calibration_note=getattr(calibrator, "reason", "platt"),
    )
    del x_train, y_train, train_data, raw_calibration
    gc.collect()
    return bundle


def fit_final_classifier(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    candidate: dict,
    rounds: dict[str, int],
    output: Path,
) -> V3ClassifierBundle:
    training = development.loc[development["year"].le(2023)].reset_index(drop=True)
    calibration = development.loc[
        development["year"].eq(2024)
    ].reset_index(drop=True)
    if candidate["architecture"] == "global":
        log(f"[v3:final] global classifier, {rounds['global']} rounds")
        global_model = fit_final_lightgbm(
            cfg,
            training,
            calibration,
            features,
            rounds["global"],
            candidate["class_weight"],
            "global",
        )
        bundle = V3ClassifierBundle(
            architecture="global",
            router_column=ROUTER_COLUMN,
            global_model=global_model,
        )
    else:
        expert_features = [
            column for column in features if column != ROUTER_COLUMN
        ]
        experts = {}
        for bucket, value in (("context", 1), ("ignition", 0)):
            train_part = training.loc[training[ROUTER_COLUMN].eq(value)]
            calibration_part = calibration.loc[
                calibration[ROUTER_COLUMN].eq(value)
            ]
            log(
                f"[v3:final] {bucket} expert: {len(train_part):,} rows, "
                f"{rounds[bucket]} rounds"
            )
            experts[bucket] = fit_final_lightgbm(
                cfg,
                train_part,
                calibration_part,
                expert_features,
                rounds[bucket],
                candidate["class_weight"],
                bucket,
            )
            del train_part, calibration_part
            gc.collect()
        bundle = V3ClassifierBundle(
            architecture="mixture",
            router_column=ROUTER_COLUMN,
            context_model=experts["context"],
            ignition_model=experts["ignition"],
        )
    bundle.save(output / "models" / "classifier.joblib")
    del training, calibration
    gc.collect()
    return bundle


def fit_final_ranker(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    rounds: int,
    output: Path,
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
        ranker_parameters(cfg),
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
    bundle.save(output / "models" / "ranker.joblib")
    del training, x_train, train_data
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


def report_markdown(
    selected_name: str,
    classifier_results: dict,
    ranker_result: dict,
    ranker_selected: bool,
    test_metrics: dict,
    versions: dict,
) -> str:
    selected = classifier_results[selected_name]
    lines = [
        "# V3 Mixture/Directional Experiment Report",
        "",
        f"**Completed:** {datetime.now(UTC).isoformat()}  ",
        f"**Selected classifier:** `{selected_name}`  ",
        f"**Selected alert head:** "
        f"`{'daily_lambdarank' if ranker_selected else 'classifier_score'}`  ",
        "",
        "V3 is stored independently from V1 and V2. All V3 selections used "
        "walk-forward 2022–2024 only.",
        "",
        "## Classifier experiments",
        "",
        "| Candidate | Architecture | Features | Mean PR-AUC | Minimum PR-AUC | P@10 | P@25 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in classifier_results.items():
        lines.append(
            f"| {name} | {result['architecture']} | "
            f"{result['feature_count']} | {result['mean_pr_auc']:.6f} | "
            f"{result['min_pr_auc']:.6f} | "
            f"{result['mean_top_k_precision']['10']:.4%} | "
            f"{result['mean_top_k_precision']['25']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Daily ranking experiment",
            "",
            f"- Mean walk-forward ranker PR-AUC: "
            f"**{ranker_result['mean_pr_auc']:.6f}**",
            f"- Mean ranker P@10: "
            f"**{ranker_result['mean_top_k_precision']['10']:.4%}**",
            f"- Mean ranker P@25: "
            f"**{ranker_result['mean_top_k_precision']['25']:.4%}**",
            "",
            "## 2025 descriptive comparison",
            "",
            "The 2025 result is descriptive, not a new untouched holdout, "
            "because V2's 2025 results had already been inspected before V3.",
            "",
            "| Version | Raw PR-AUC | Calibrated PR-AUC | ROC-AUC | Brier | P@25 | Recall@25 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("V1", "V2"):
        result = versions[name]
        lines.append(
            f"| {name} | {result['raw_pr_auc']:.6f} | "
            f"{result['calibrated_pr_auc']:.6f} | {result['roc_auc']:.6f} | "
            f"{result['brier']:.6f} | {result['precision_25']:.4%} | "
            f"{result['recall_25']:.4%} |"
        )
    v3 = test_metrics
    lines.append(
        f"| V3 | {v3['classifier_raw']['pr_auc']:.6f} | "
        f"{v3['classifier_calibrated']['pr_auc']:.6f} | "
        f"{v3['classifier_calibrated']['roc_auc']:.6f} | "
        f"{v3['classifier_calibrated']['brier']:.6f} | "
        f"{v3['selected_alert_head']['top_k']['25']['alert_precision']:.4%} | "
        f"{v3['selected_alert_head']['top_k']['25']['positive_recall']:.4%} |"
    )
    lines.extend(
        [
            "",
            "## Leakage and serving contract",
            "",
            "- Target T is D+1; ERA5-derived inputs end D−5.",
            "- All fire histories and the router end T−2=D−1.",
            "- Directional features combine causal fire history with D−5 wind; "
            "they do not read the target.",
            "- The continuation/ignition route is determined before scoring.",
            "- Platt calibration was fitted on 2024 after the architecture was selected.",
            "- D−1 FIRMS history must be available operationally; otherwise V3 "
            "must be rebuilt with the actual serving cutoff.",
            "",
        ]
    )
    return "\n".join(lines)


def version_payload(metrics: dict) -> dict:
    return {
        "raw_pr_auc": metrics["raw"]["pr_auc"],
        "calibrated_pr_auc": metrics["calibrated"]["pr_auc"],
        "roc_auc": metrics["calibrated"]["roc_auc"],
        "brier": metrics["calibrated"]["brier"],
        "precision_25": metrics["calibrated"]["alert_precision"],
        "recall_25": metrics["calibrated"]["positive_recall"],
    }


def write_version_registry(
    destination: Path,
    versions: dict,
    v3_metrics: dict,
    v3_output: Path,
    v3_architecture: str,
) -> None:
    payload = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "holdout_note": (
            "Only V1's initial 2025 evaluation was fully untouched across the "
            "experiment series. V2 and V3 were designed after earlier 2025 "
            "results were inspected, so their 2025 comparisons are descriptive."
        ),
        "versions": {
            "V1": {
                **versions["V1"],
                "artifact_directory": str(
                    (destination / "lag5_full_year").resolve()
                ),
                "architecture": "Stage C global LightGBM",
            },
            "V2": {
                **versions["V2"],
                "artifact_directory": str((destination / "lag5_v2").resolve()),
                "architecture": "Leakage-safe global LightGBM",
            },
            "V3": {
                "raw_pr_auc": v3_metrics["classifier_raw"]["pr_auc"],
                "calibrated_pr_auc": v3_metrics["classifier_calibrated"][
                    "pr_auc"
                ],
                "roc_auc": v3_metrics["classifier_calibrated"]["roc_auc"],
                "brier": v3_metrics["classifier_calibrated"]["brier"],
                "precision_25": v3_metrics["selected_alert_head"]["top_k"]["25"][
                    "alert_precision"
                ],
                "recall_25": v3_metrics["selected_alert_head"]["top_k"]["25"][
                    "positive_recall"
                ],
                "artifact_directory": str(v3_output.resolve()),
                "architecture": v3_architecture,
            },
        },
    }
    atomic_json(payload, destination / "model_versions.json")
    lines = [
        "# Model Version Comparison",
        "",
        "| Version | Architecture | Raw PR-AUC | Calibrated PR-AUC | Brier | P@25 | Recall@25 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["versions"].items():
        lines.append(
            f"| {name} | {result['architecture']} | "
            f"{result['raw_pr_auc']:.6f} | "
            f"{result['calibrated_pr_auc']:.6f} | "
            f"{result['brier']:.6f} | {result['precision_25']:.4%} | "
            f"{result['recall_25']:.4%} |"
        )
    lines.extend(["", payload["holdout_note"], ""])
    (destination / "MODEL_VERSION_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--v1-metrics", required=True, type=Path)
    parser.add_argument("--v2-metrics", required=True, type=Path)
    args = parser.parse_args()
    data_dir = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    feature_sets = json.loads(
        (data_dir / "feature_sets.json").read_text(encoding="utf-8")
    )
    all_features = feature_sets["v3_directional"]
    log(f"[v3:load] development data with {len(all_features)} features")
    development = load_development(data_dir, all_features)

    classifier_results = run_classifier_experiments(
        cfg, development, feature_sets, output
    )
    selected_name = max(
        classifier_results,
        key=lambda name: (
            classifier_results[name]["mean_pr_auc"],
            classifier_results[name]["min_pr_auc"],
        ),
    )
    selected = classifier_results[selected_name]
    selected_features = feature_sets[selected["feature_set"]]
    log(f"[v3:selection] classifier={selected_name}")

    ranker_result = run_ranker_experiment(
        cfg, development, selected_features, output
    )
    classifier_p25 = selected["mean_top_k_precision"]["25"]
    classifier_p10 = selected["mean_top_k_precision"]["10"]
    ranker_selected = (
        ranker_result["mean_top_k_precision"]["25"] > classifier_p25
        and ranker_result["mean_top_k_precision"]["10"] >= classifier_p10 * 0.995
    )
    log(
        "[v3:selection] alert_head="
        + ("daily_lambdarank" if ranker_selected else "classifier_score")
    )

    classifier = fit_final_classifier(
        cfg,
        development,
        selected_features,
        selected,
        selected["median_best_iterations"],
        output,
    )
    ranker = fit_final_ranker(
        cfg,
        development,
        selected_features,
        ranker_result["median_best_iteration"],
        output,
    )
    del development
    gc.collect()

    log("[v3:test] loading 2025 after V3 choices are locked")
    test_columns = list(
        dict.fromkeys([*IDENTITY_COLUMNS, *selected_features, ROUTER_COLUMN])
    )
    test = pd.read_parquet(data_dir / "test.parquet", columns=test_columns)
    test["label_date"] = pd.to_datetime(test["label_date"]).dt.normalize()
    raw_probability = classifier.predict_raw(test)
    probability = classifier.predict_proba(test)
    rank_score = ranker.predict_score(test)
    alert_score = rank_score if ranker_selected else raw_probability
    classifier_raw = scored_metrics(test, raw_probability)
    classifier_calibrated = scored_metrics(test, probability)
    ranker_metrics = scored_metrics(test, rank_score, include_probability=False)
    alert_metrics = scored_metrics(test, alert_score, include_probability=False)
    route_metrics = {}
    for bucket, value in (("context", 1), ("ignition", 0)):
        mask = test[ROUTER_COLUMN].eq(value)
        route_metrics[bucket] = binary_metrics(
            test.loc[mask, "y_fire"].to_numpy(), probability[mask.to_numpy()]
        )
    test_metrics = {
        "classifier_raw": classifier_raw,
        "classifier_calibrated": classifier_calibrated,
        "ranker": ranker_metrics,
        "selected_alert_head": {
            "name": "daily_lambdarank" if ranker_selected else "classifier_score",
            **alert_metrics,
        },
        "by_route_calibrated": route_metrics,
    }
    scored = test[IDENTITY_COLUMNS].copy()
    scored[ROUTER_COLUMN] = test[ROUTER_COLUMN].astype("int8")
    scored["p_fire_raw"] = raw_probability.astype("float32")
    scored["p_fire"] = probability.astype("float32")
    scored["rank_score"] = rank_score.astype("float32")
    scored["alert_score"] = alert_score.astype("float32")
    scored["confidence_pct"] = (100 * probability).astype("float32")
    atomic_parquet(scored, output / "test_predictions.parquet")
    plot_calibration_and_pr(
        scored,
        output / "plots" / "calibration_and_pr_2025.png",
        f"V3 {selected['architecture']} classifier — 2025 descriptive",
    )
    peak_day = scored.groupby("label_date")["p_fire"].max().idxmax()
    plot_risk_map(
        scored, output / "plots" / "risk_map_peak_day_2025.png", peak_day
    )

    classifier_models = {
        "global": classifier.global_model,
        "context": classifier.context_model,
        "ignition": classifier.ignition_model,
    }
    for name, model in classifier_models.items():
        if model is None:
            continue
        write_importance(
            model.booster,
            model.feature_columns,
            output / "explainability" / f"classifier_{name}_importance.csv",
        )
        mask = (
            np.ones(len(test), dtype=bool)
            if name == "global"
            else test[ROUTER_COLUMN].eq(1 if name == "context" else 0).to_numpy()
        )
        write_lightgbm_explanations(
            model,
            test.loc[mask],
            output / "explainability" / f"classifier_{name}",
            max_rows=int(cfg["training"]["max_explain_rows"]),
        )
    write_importance(
        ranker.booster,
        ranker.feature_columns,
        output / "explainability" / "ranker_importance.csv",
    )

    v1 = json.loads(args.v1_metrics.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_metrics.read_text(encoding="utf-8"))
    versions = {
        "V1": version_payload(v1["test_2025"]),
        "V2": version_payload(v2["test_2025"]),
    }
    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "version": "V3",
        "selection_protocol": (
            "Classifier and alert-head selection used walk-forward 2022–2024 "
            "only. The 2025 comparison is descriptive because V2 test results "
            "were already known before V3."
        ),
        "selected_classifier": selected_name,
        "selected_architecture": selected["architecture"],
        "selected_feature_set": selected["feature_set"],
        "selected_feature_count": len(selected_features),
        "selected_alert_head": (
            "daily_lambdarank" if ranker_selected else "classifier_score"
        ),
        "classifier_experiments": classifier_results,
        "ranker_experiment": ranker_result,
        "test_2025_descriptive": test_metrics,
        "v1_v2_reference": versions,
    }
    atomic_json(payload, output / "metrics.json")
    report = report_markdown(
        selected_name,
        classifier_results,
        ranker_result,
        ranker_selected,
        test_metrics,
        versions,
    )
    (output / "TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    artifacts = {
        "metrics": output / "metrics.json",
        "report": output / "TRAINING_REPORT.md",
        "predictions": output / "test_predictions.parquet",
        "classifier": output / "models" / "classifier.joblib",
        "ranker": output / "models" / "ranker.joblib",
        "calibration_plot": output / "plots" / "calibration_and_pr_2025.png",
        "risk_map": output / "plots" / "risk_map_peak_day_2025.png",
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "version": "V3",
        "selected_classifier": selected_name,
        "selected_alert_head": payload["selected_alert_head"],
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
    write_version_registry(
        output.parent,
        versions,
        test_metrics,
        output,
        (
            f"{selected_name} with Platt calibration; alert head="
            f"{payload['selected_alert_head']}"
        ),
    )
    log(f"[v3:done] report={output / 'TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
