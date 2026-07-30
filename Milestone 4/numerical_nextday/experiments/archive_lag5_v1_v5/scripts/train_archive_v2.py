from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wildfire-v2-matplotlib")
)

from numerical_nextday.config import load_config  # noqa: E402
from numerical_nextday.evaluation.explain import (  # noqa: E402
    write_lightgbm_explanations,
)
from numerical_nextday.evaluation.metrics import (  # noqa: E402
    complete_metrics,
    metrics_by_group,
)
from numerical_nextday.evaluation.plots import (  # noqa: E402
    plot_calibration_and_pr,
    plot_risk_map,
)
from numerical_nextday.io import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    sha256_file,
)
from numerical_nextday.train.calibration import fit_calibrator  # noqa: E402
from numerical_nextday.train.lightgbm_model import (  # noqa: E402
    LightGBMBundle,
    lightgbm_params,
)
from numerical_nextday.train.preprocessing import TabularPreprocessor  # noqa: E402


IDENTITY_COLUMNS = ["label_date", "cell_id", "latitude", "longitude", "y_fire"]
FIRE_SEASON_MONTHS = {4, 5, 6, 7, 8, 9, 10, 11}
CV_YEARS = (2022, 2023, 2024)
CONFIGURATIONS = {
    "default": {},
    "leaves_31": {"num_leaves": 31},
    "regularized": {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        "lambda_l2": 5.0,
    },
    "no_class_weight": {"scale_pos_weight": 1.0},
}


def log(message: str) -> None:
    print(message, flush=True)


def compact_metrics(
    frame: pd.DataFrame, probability: np.ndarray, top_k: int
) -> dict:
    scored = frame[["label_date", "y_fire"]].copy()
    scored["p_fire"] = np.asarray(probability, dtype="float32")
    return complete_metrics(scored, top_k)


def load_development(data_dir: Path, features: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features]))
    train = pd.read_parquet(data_dir / "train.parquet", columns=columns)
    validation = pd.read_parquet(data_dir / "val.parquet", columns=columns)
    frame = pd.concat([train, validation], ignore_index=True)
    del train, validation
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    frame["year"] = frame["label_date"].dt.year.astype("int16")
    frame["month"] = frame["label_date"].dt.month.astype("int8")
    return frame


def model_parameters(
    cfg: dict,
    target: np.ndarray,
    overrides: dict,
) -> dict:
    params = lightgbm_params(cfg, overrides)
    positives = max(int(target.sum()), 1)
    negatives = max(int(len(target) - positives), 1)
    params.setdefault("scale_pos_weight", negatives / positives)
    return params


def fit_cv_fold(
    cfg: dict,
    frame: pd.DataFrame,
    features: list[str],
    validation_year: int,
    overrides: dict,
    bucket: str | None = None,
    return_probability: bool = False,
) -> tuple[dict, int, np.ndarray | None]:
    train_mask = frame["year"].lt(validation_year)
    validation_mask = frame["year"].eq(validation_year)
    if bucket == "fire_season":
        train_mask &= frame["month"].isin(FIRE_SEASON_MONTHS)
        validation_mask &= frame["month"].isin(FIRE_SEASON_MONTHS)
    elif bucket == "winter":
        train_mask &= ~frame["month"].isin(FIRE_SEASON_MONTHS)
        validation_mask &= ~frame["month"].isin(FIRE_SEASON_MONTHS)
    train = frame.loc[train_mask]
    validation = frame.loc[validation_mask]
    x_train = train[features].to_numpy(dtype="float32", copy=True)
    y_train = train["y_fire"].to_numpy(dtype="int8", copy=True)
    x_validation = validation[features].to_numpy(dtype="float32", copy=True)
    y_validation = validation["y_fire"].to_numpy(dtype="int8", copy=True)
    params = model_parameters(cfg, y_train, overrides)
    training_data = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=features,
        free_raw_data=False,
    )
    validation_data = lgb.Dataset(
        x_validation,
        label=y_validation,
        reference=training_data,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        training_data,
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
    probability = booster.predict(
        x_validation, num_iteration=booster.best_iteration
    )
    metrics = compact_metrics(
        validation,
        probability,
        int(cfg["training"]["top_k_per_day"]),
    )
    metrics.update(
        {
            "validation_year": validation_year,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "best_iteration": int(booster.best_iteration),
            "bucket": bucket or "all_months",
        }
    )
    best_iteration = int(booster.best_iteration)
    retained_probability = probability if return_probability else None
    del (
        train,
        validation,
        x_train,
        y_train,
        x_validation,
        y_validation,
        training_data,
        validation_data,
        booster,
    )
    del probability
    gc.collect()
    return metrics, best_iteration, retained_probability


def summarize_folds(folds: list[dict]) -> dict:
    pr_values = np.array([fold["pr_auc"] for fold in folds], dtype=float)
    roc_values = np.array([fold["roc_auc"] for fold in folds], dtype=float)
    iterations = [int(fold["best_iteration"]) for fold in folds]
    return {
        "folds": folds,
        "mean_pr_auc": float(pr_values.mean()),
        "min_pr_auc": float(pr_values.min()),
        "std_pr_auc": float(pr_values.std()),
        "mean_roc_auc": float(roc_values.mean()),
        "median_best_iteration": int(np.median(iterations)),
    }


def run_feature_ablation(
    cfg: dict,
    development: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output: Path,
) -> dict:
    results = {}
    for name, features in feature_sets.items():
        log(f"[cv:features] {name}: {len(features)} features")
        folds = [
            fit_cv_fold(cfg, development, features, year, {})[0]
            for year in CV_YEARS
        ]
        results[name] = summarize_folds(folds)
        atomic_json(results, output / "feature_ablation.partial.json")
        log(
            f"[cv:features] {name}: mean PR-AUC="
            f"{results[name]['mean_pr_auc']:.6f}"
        )
    return results


def run_parameter_search(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    output: Path,
) -> dict:
    results = {}
    for name, overrides in CONFIGURATIONS.items():
        log(f"[cv:params] {name}")
        folds = [
            fit_cv_fold(cfg, development, features, year, overrides)[0]
            for year in CV_YEARS
        ]
        results[name] = {
            "overrides": overrides,
            **summarize_folds(folds),
        }
        atomic_json(results, output / "parameter_search.partial.json")
        log(
            f"[cv:params] {name}: mean PR-AUC="
            f"{results[name]['mean_pr_auc']:.6f}"
        )
    return results


def run_seasonal_cv(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    overrides: dict,
    output: Path,
) -> dict:
    fold_results = []
    bucket_iterations = {"fire_season": [], "winter": []}
    for year in CV_YEARS:
        validation = development.loc[development["year"].eq(year)]
        routed = validation[["label_date", "y_fire"]].copy()
        routed["p_fire"] = np.nan
        bucket_metrics = {}
        for bucket in ("fire_season", "winter"):
            metrics, iteration, probability = fit_cv_fold(
                cfg,
                development,
                features,
                year,
                overrides,
                bucket=bucket,
                return_probability=True,
            )
            bucket_iterations[bucket].append(iteration)
            validation_mask = development["year"].eq(year)
            if bucket == "fire_season":
                validation_mask &= development["month"].isin(FIRE_SEASON_MONTHS)
            else:
                validation_mask &= ~development["month"].isin(FIRE_SEASON_MONTHS)
            bucket_validation = development.loc[validation_mask]
            if probability is None:
                raise RuntimeError("Seasonal CV did not retain fold predictions")
            routed.loc[bucket_validation.index, "p_fire"] = probability
            bucket_metrics[bucket] = metrics
            del bucket_validation, probability
            gc.collect()
        if routed["p_fire"].isna().any():
            raise RuntimeError(f"Seasonal router left rows unscored for {year}")
        overall = complete_metrics(
            routed,
            int(cfg["training"]["top_k_per_day"]),
        )
        fold_results.append(
            {
                "validation_year": year,
                "overall": overall,
                "by_bucket": bucket_metrics,
            }
        )
        atomic_json(fold_results, output / "seasonal_cv.partial.json")
    pr_values = [fold["overall"]["pr_auc"] for fold in fold_results]
    result = {
        "folds": fold_results,
        "mean_pr_auc": float(np.mean(pr_values)),
        "min_pr_auc": float(np.min(pr_values)),
        "median_best_iteration": {
            bucket: int(np.median(values))
            for bucket, values in bucket_iterations.items()
        },
    }
    atomic_json(result, output / "seasonal_cv.json")
    return result


def fit_final_bundle(
    cfg: dict,
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    overrides: dict,
    rounds: int,
    stage: str,
    bucket: str,
) -> LightGBMBundle:
    preprocessor = TabularPreprocessor(features).fit(training, scale=False)
    x_train = preprocessor.transform(training)
    y_train = training["y_fire"].to_numpy(dtype="int8")
    params = model_parameters(cfg, y_train, overrides)
    training_data = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=features,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        training_data,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    x_calibration = preprocessor.transform(calibration)
    raw_calibration = booster.predict(x_calibration, num_iteration=rounds)
    calibrator = fit_calibrator(
        raw_calibration,
        calibration["y_fire"].to_numpy(dtype="int8"),
        int(cfg["model_buckets"]["min_calibration_positives"]),
    )
    bundle = LightGBMBundle(
        booster=booster,
        calibrator=calibrator,
        preprocessor=preprocessor,
        feature_columns=features,
        stage=stage,
        model_bucket=bucket,
        params=params,
        random_seed=int(cfg["project"]["random_seed"]),
        best_iteration=rounds,
        calibration_note=getattr(calibrator, "reason", "isotonic"),
    )
    del x_train, y_train, training_data, x_calibration, raw_calibration
    gc.collect()
    return bundle


def fit_final_models(
    cfg: dict,
    development: pd.DataFrame,
    features: list[str],
    overrides: dict,
    global_rounds: int,
    seasonal_rounds: dict[str, int],
    output: Path,
) -> dict[str, LightGBMBundle]:
    training = development.loc[development["year"].le(2023)].reset_index(drop=True)
    calibration = development.loc[development["year"].eq(2024)].reset_index(drop=True)
    bundles = {}
    log(
        f"[final] global model: train through 2023, fixed {global_rounds} rounds"
    )
    bundles["global"] = fit_final_bundle(
        cfg,
        training,
        calibration,
        features,
        overrides,
        global_rounds,
        "V2",
        "all_months",
    )
    for bucket in ("fire_season", "winter"):
        if bucket == "fire_season":
            train_part = training.loc[
                training["month"].isin(FIRE_SEASON_MONTHS)
            ].reset_index(drop=True)
            calibration_part = calibration.loc[
                calibration["month"].isin(FIRE_SEASON_MONTHS)
            ].reset_index(drop=True)
        else:
            train_part = training.loc[
                ~training["month"].isin(FIRE_SEASON_MONTHS)
            ].reset_index(drop=True)
            calibration_part = calibration.loc[
                ~calibration["month"].isin(FIRE_SEASON_MONTHS)
            ].reset_index(drop=True)
        rounds = seasonal_rounds[bucket]
        log(f"[final] {bucket}: {len(train_part):,} rows, {rounds} rounds")
        bundles[bucket] = fit_final_bundle(
            cfg,
            train_part,
            calibration_part,
            features,
            overrides,
            rounds,
            "V2",
            bucket,
        )
        del train_part, calibration_part
        gc.collect()
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in bundles.items():
        bundle.save(models_dir / f"{name}.joblib")
    del training, calibration
    gc.collect()
    return bundles


def score_architecture(
    architecture: str,
    bundles: dict[str, LightGBMBundle],
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.empty(len(test), dtype="float64")
    calibrated = np.empty(len(test), dtype="float64")
    if architecture == "global":
        raw[:] = bundles["global"].predict_raw(test)
        calibrated[:] = bundles["global"].calibrator.predict(raw)
        return raw, calibrated
    for bucket in ("fire_season", "winter"):
        mask = (
            test["month"].isin(FIRE_SEASON_MONTHS)
            if bucket == "fire_season"
            else ~test["month"].isin(FIRE_SEASON_MONTHS)
        )
        bucket_raw = bundles[bucket].predict_raw(test.loc[mask])
        raw[mask.to_numpy()] = bucket_raw
        calibrated[mask.to_numpy()] = bundles[bucket].calibrator.predict(bucket_raw)
    return raw, calibrated


def write_importance(bundle: LightGBMBundle, destination: Path) -> None:
    table = pd.DataFrame(
        {
            "feature": bundle.feature_columns,
            "gain_importance": bundle.booster.feature_importance(
                importance_type="gain"
            ),
            "split_importance": bundle.booster.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)


def report_markdown(
    selected_features: str,
    selected_config: str,
    architecture: str,
    feature_ablation: dict,
    parameter_search: dict,
    seasonal_cv: dict,
    test_metrics: dict,
    baseline: dict | None,
) -> str:
    lines = [
        "# Leakage-Safe V2 Training Report",
        "",
        f"**Completed:** {datetime.now(UTC).isoformat()}  ",
        f"**Selected feature set:** `{selected_features}`  ",
        f"**Selected LightGBM configuration:** `{selected_config}`  ",
        f"**Selected architecture:** `{architecture}`  ",
        "",
        "All choices were made from walk-forward 2022–2024 validation. No V2 "
        "feature, parameter, or architecture decision used 2025.",
        "",
        "## Walk-forward feature ablation",
        "",
        "| Feature set | Features | Mean PR-AUC | Minimum PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, result in sorted(feature_ablation.items()):
        lines.append(
            f"| {name} | {result['feature_count']} | "
            f"{result['mean_pr_auc']:.6f} | {result['min_pr_auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Parameter search",
            "",
            "| Configuration | Mean PR-AUC | Minimum PR-AUC | Median rounds |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in sorted(parameter_search.items()):
        lines.append(
            f"| {name} | {result['mean_pr_auc']:.6f} | "
            f"{result['min_pr_auc']:.6f} | "
            f"{result['median_best_iteration']} |"
        )
    test = test_metrics["calibrated"]
    raw = test_metrics["raw"]
    lines.extend(
        [
            "",
            "## Architecture selection",
            "",
            f"- Global walk-forward mean PR-AUC: "
            f"**{parameter_search[selected_config]['mean_pr_auc']:.6f}**",
            f"- Seasonal-router walk-forward mean PR-AUC: "
            f"**{seasonal_cv['mean_pr_auc']:.6f}**",
            "",
            "## Locked 2025 evaluation",
            "",
            "| Probability | PR-AUC | ROC-AUC | Brier | ECE |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Raw | {raw['pr_auc']:.6f} | {raw['roc_auc']:.6f} | "
                f"{raw['brier']:.6f} | {raw['ece_10bin']:.6f} |"
            ),
            (
                f"| Calibrated | {test['pr_auc']:.6f} | "
                f"{test['roc_auc']:.6f} | {test['brier']:.6f} | "
                f"{test['ece_10bin']:.6f} |"
            ),
            "",
            (
                f"Daily top-25 precision is **{test['alert_precision']:.4%}** "
                f"and positive cell-day recall is **{test['positive_recall']:.4%}**."
            ),
        ]
    )
    if baseline:
        baseline_test = baseline["test_2025"]["calibrated"]
        lines.extend(
            [
                "",
                "## Comparison with V1",
                "",
                "| Model | PR-AUC | ROC-AUC | Brier |",
                "|---|---:|---:|---:|",
                (
                    f"| V1 Stage C LightGBM | {baseline_test['pr_auc']:.6f} | "
                    f"{baseline_test['roc_auc']:.6f} | "
                    f"{baseline_test['brier']:.6f} |"
                ),
                (
                    f"| V2 selected | {test['pr_auc']:.6f} | "
                    f"{test['roc_auc']:.6f} | {test['brier']:.6f} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Leakage controls",
            "",
            "- Direct target-day FIRMS outcome columns were excluded; only "
            "explicitly lagged aggregate fire histories were included.",
            "- Weather rolling windows end at ERA5 day D−5.",
            "- For target T=D+1, fire-history windows end at T−2=D−1; "
            "forecast-day D and target-day D+1 FIRMS outcomes are excluded.",
            "- Cell-risk history is expanding and shifted; it never uses the "
            "current or future target.",
            "- Feature/config/architecture selection used only 2022–2024.",
            "- 2025 was scored only after V2 choices were locked.",
            "",
            "The fire-history features are operationally valid only if FIRMS "
            "observations through D−1 are available when the forecast for D+1 "
            "is issued on D. If the serving feed is delayed to D−5, those "
            "features must be rebuilt with a D−5 cutoff.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--baseline-metrics", type=Path)
    args = parser.parse_args()
    data_dir = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    feature_sets = json.loads(
        (data_dir / "feature_sets.json").read_text(encoding="utf-8")
    )
    feature_sets["v1_clean_base"] = feature_sets["v2_full"][:61]
    all_features = feature_sets["v2_full"]
    log(f"[load] development data with {len(all_features)} total features")
    development = load_development(data_dir, all_features)

    feature_ablation = run_feature_ablation(
        cfg, development, feature_sets, output
    )
    for name, result in feature_ablation.items():
        result["feature_count"] = len(feature_sets[name])
    selected_features_name = max(
        feature_ablation, key=lambda name: feature_ablation[name]["mean_pr_auc"]
    )
    selected_features = feature_sets[selected_features_name]
    log(f"[selection] feature set={selected_features_name}")

    parameter_search = run_parameter_search(
        cfg, development, selected_features, output
    )
    selected_config_name = max(
        parameter_search, key=lambda name: parameter_search[name]["mean_pr_auc"]
    )
    selected_config = CONFIGURATIONS[selected_config_name]
    log(f"[selection] configuration={selected_config_name}")

    seasonal_cv = run_seasonal_cv(
        cfg, development, selected_features, selected_config, output
    )
    global_cv = parameter_search[selected_config_name]
    architecture = (
        "seasonal"
        if seasonal_cv["mean_pr_auc"] > global_cv["mean_pr_auc"]
        else "global"
    )
    log(f"[selection] architecture={architecture}")

    bundles = fit_final_models(
        cfg,
        development,
        selected_features,
        selected_config,
        int(global_cv["median_best_iteration"]),
        seasonal_cv["median_best_iteration"],
        output,
    )
    del development
    gc.collect()

    log("[test] loading 2025 after all V2 selections are locked")
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *selected_features]))
    test = pd.read_parquet(data_dir / "test.parquet", columns=columns)
    test["label_date"] = pd.to_datetime(test["label_date"]).dt.normalize()
    test["month"] = test["label_date"].dt.month.astype("int8")
    raw_probability, probability = score_architecture(
        architecture, bundles, test
    )
    raw_metrics = compact_metrics(
        test, raw_probability, int(cfg["training"]["top_k_per_day"])
    )
    calibrated_metrics = compact_metrics(
        test, probability, int(cfg["training"]["top_k_per_day"])
    )
    scored = test[IDENTITY_COLUMNS].copy()
    scored["p_fire_raw"] = raw_probability.astype("float32")
    scored["p_fire"] = probability.astype("float32")
    scored["confidence_pct"] = (100 * scored["p_fire"]).astype("float32")
    test_metrics = {
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "by_month_calibrated": metrics_by_group(
            scored.assign(month=scored["label_date"].dt.month),
            "month",
            int(cfg["training"]["top_k_per_day"]),
        ),
    }
    atomic_parquet(scored, output / "test_predictions.parquet")
    plot_calibration_and_pr(
        scored,
        output / "plots" / "calibration_and_pr_2025.png",
        f"Leakage-safe V2 {architecture} — 2025",
    )
    risk_day = scored.groupby("label_date")["p_fire"].max().idxmax()
    plot_risk_map(
        scored,
        output / "plots" / "risk_map_peak_day_2025.png",
        risk_day,
    )
    for name, bundle in bundles.items():
        write_importance(
            bundle, output / "explainability" / f"{name}_importance.csv"
        )
        if architecture == "global" and name == "global":
            write_lightgbm_explanations(
                bundle,
                test,
                output / "explainability" / "selected_model",
                max_rows=int(cfg["training"]["max_explain_rows"]),
            )
        elif architecture == "seasonal" and name in {"fire_season", "winter"}:
            mask = (
                test["month"].isin(FIRE_SEASON_MONTHS)
                if name == "fire_season"
                else ~test["month"].isin(FIRE_SEASON_MONTHS)
            )
            write_lightgbm_explanations(
                bundle,
                test.loc[mask],
                output / "explainability" / name,
                max_rows=int(cfg["training"]["max_explain_rows"]),
            )

    baseline = (
        json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
        if args.baseline_metrics and args.baseline_metrics.exists()
        else None
    )
    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "selection_protocol": (
            "Walk-forward raw PR-AUC on 2022, 2023, and 2024; "
            "2025 excluded from all V2 decisions."
        ),
        "selected_feature_set": selected_features_name,
        "selected_feature_count": len(selected_features),
        "selected_configuration": selected_config_name,
        "selected_overrides": selected_config,
        "selected_architecture": architecture,
        "feature_ablation": feature_ablation,
        "parameter_search": parameter_search,
        "seasonal_cv": seasonal_cv,
        "test_2025": test_metrics,
    }
    atomic_json(payload, output / "metrics.json")
    report = report_markdown(
        selected_features_name,
        selected_config_name,
        architecture,
        feature_ablation,
        parameter_search,
        seasonal_cv,
        test_metrics,
        baseline,
    )
    (output / "TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    artifacts = {
        "metrics": output / "metrics.json",
        "report": output / "TRAINING_REPORT.md",
        "predictions": output / "test_predictions.parquet",
        "calibration_plot": output / "plots" / "calibration_and_pr_2025.png",
        "risk_map": output / "plots" / "risk_map_peak_day_2025.png",
    }
    for name in ("global", "fire_season", "winter"):
        artifacts[f"model_{name}"] = output / "models" / f"{name}.joblib"
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_architecture": architecture,
        "selected_feature_set": selected_features_name,
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
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }
    atomic_json(manifest, output / "run_manifest.json")
    log(f"[done] report={output / 'TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
