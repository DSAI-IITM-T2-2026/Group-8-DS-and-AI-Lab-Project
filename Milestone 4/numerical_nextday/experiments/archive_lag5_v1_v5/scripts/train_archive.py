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
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wildfire-matplotlib-cache")
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
from numerical_nextday.io import atomic_json, atomic_parquet, sha256_file  # noqa: E402
from numerical_nextday.train.calibration import fit_calibrator  # noqa: E402
from numerical_nextday.train.lightgbm_model import (  # noqa: E402
    LightGBMBundle,
    fit_lightgbm,
)
from numerical_nextday.train.mlp_model import MLPBundle  # noqa: E402
from numerical_nextday.train.preprocessing import TabularPreprocessor  # noqa: E402


STAGES = ("a", "b", "c")
IDENTITY_COLUMNS = ["label_date", "cell_id", "latitude", "longitude", "y_fire"]
SOIL_COLUMNS_TO_CLIP = [
    "swvl1_mean",
    "swvl2_mean",
    "soil_moisture_index",
    "swvl1_mean_7d",
]
S2_NON_MEASUREMENT_COLUMNS = {"s2n_available"}
S5_NON_MEASUREMENT_COLUMNS = {"s5n_available"}


def log(message: str) -> None:
    print(message, flush=True)


def stage_dir(archive: Path, stage: str) -> Path:
    return archive / f"stage_{stage.lower()}"


def declared_features(archive: Path, stage: str) -> list[str]:
    path = stage_dir(archive, stage) / "metadata" / "feature_columns.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _availability_cleanup(
    frame: pd.DataFrame,
    features: list[str],
    prefix: str,
    maximum_lag: int,
) -> dict[str, int]:
    available_column = f"{prefix}_available"
    lag_column = f"{prefix}_lag_days"
    if available_column not in frame or lag_column not in frame:
        return {}
    invalid = (
        frame[available_column].ne(1)
        | frame[lag_column].isna()
        | frame[lag_column].lt(0)
        | frame[lag_column].gt(maximum_lag)
    )
    non_measurements = (
        S2_NON_MEASUREMENT_COLUMNS if prefix == "s2n" else S5_NON_MEASUREMENT_COLUMNS
    )
    measurement_columns = [
        column
        for column in features
        if column.startswith(f"{prefix}_") and column not in non_measurements
    ]
    before_available = int(frame[available_column].eq(1).sum())
    if measurement_columns:
        frame.loc[invalid, measurement_columns] = 0.0
    frame.loc[invalid, available_column] = 0
    return {
        f"{prefix}_rows_forced_unavailable": int(invalid.sum()),
        f"{prefix}_rows_newly_forced_unavailable": int(
            before_available - frame[available_column].eq(1).sum()
        ),
    }


def load_and_clean(
    archive: Path,
    stage: str,
    split: str,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = stage_dir(archive, stage) / f"{split}.parquet"
    auxiliary = ["s2n_lag_days", "s5n_lag_days"]
    available_auxiliary = []
    parquet_columns = set(
        __import__("pyarrow.parquet", fromlist=["ParquetFile"])
        .ParquetFile(path)
        .schema_arrow.names
    )
    for column in auxiliary:
        if column in parquet_columns:
            available_auxiliary.append(column)
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features, *available_auxiliary]))
    frame = pd.read_parquet(path, columns=columns)
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    cleanup: dict[str, int] = {}
    if stage in {"b", "c"}:
        cleanup.update(_availability_cleanup(frame, features, "s2n", 15))
    if stage == "c":
        cleanup.update(_availability_cleanup(frame, features, "s5n", 2))
    for column in SOIL_COLUMNS_TO_CLIP:
        if column in frame:
            count = int(frame[column].lt(0).sum())
            frame[column] = frame[column].clip(lower=0)
            cleanup[f"{column}_negative_rows_clipped"] = count
    frame[features] = frame[features].astype("float32")
    if not np.isfinite(frame[features].to_numpy(dtype="float32")).all():
        raise ValueError(f"Non-finite value remains in stage {stage} split {split}")
    return frame, cleanup


def split_validation(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tune = validation.loc[validation["label_date"].dt.year == 2023].reset_index(drop=True)
    calibration = validation.loc[
        validation["label_date"].dt.year == 2024
    ].reset_index(drop=True)
    if len(tune) != 245_280 or len(calibration) != 245_952:
        raise ValueError(
            f"Unexpected validation partition sizes: tune={len(tune)}, "
            f"calibration={len(calibration)}"
        )
    return tune, calibration


def remove_constant_features(
    train: pd.DataFrame, features: list[str]
) -> tuple[list[str], list[str]]:
    minimum = train[features].min(axis=0)
    maximum = train[features].max(axis=0)
    constants = [
        column
        for column in features
        if pd.isna(minimum[column]) or minimum[column] == maximum[column]
    ]
    return [column for column in features if column not in constants], constants


def compact_scored(
    frame: pd.DataFrame, raw_probability: np.ndarray, probability: np.ndarray
) -> pd.DataFrame:
    scored = frame[IDENTITY_COLUMNS].copy()
    scored["p_fire_raw"] = np.asarray(raw_probability, dtype="float32")
    scored["p_fire"] = np.asarray(probability, dtype="float32")
    scored["confidence_pct"] = (100 * scored["p_fire"]).astype("float32")
    return scored


def raw_and_calibrated_metrics(
    frame: pd.DataFrame,
    raw_probability: np.ndarray,
    probability: np.ndarray,
    top_k: int,
) -> dict:
    raw_scored = frame[["label_date", "y_fire"]].copy()
    raw_scored["p_fire"] = np.asarray(raw_probability, dtype="float32")
    calibrated_scored = frame[["label_date", "y_fire"]].copy()
    calibrated_scored["p_fire"] = np.asarray(probability, dtype="float32")
    return {
        "raw": complete_metrics(raw_scored, top_k),
        "calibrated": complete_metrics(calibrated_scored, top_k),
    }


def lgbm_raw(bundle: LightGBMBundle, frame: pd.DataFrame) -> np.ndarray:
    return bundle.predict_raw(frame)


def mlp_raw(bundle: MLPBundle, frame: pd.DataFrame) -> np.ndarray:
    matrix = bundle.preprocessor.transform(frame[bundle.feature_columns])
    return np.asarray(bundle.model.predict_proba(matrix)[:, 1], dtype=float)


def fit_weighted_mlp(
    cfg: dict,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
) -> MLPBundle:
    mlp_cfg = cfg["training"]["mlp"]
    preprocessor = TabularPreprocessor(features).fit(train, scale=True)
    matrix = preprocessor.transform(train)
    target = train["y_fire"].astype(int).to_numpy()
    positives = max(int(target.sum()), 1)
    negatives = max(int(len(target) - positives), 1)
    positive_weight = negatives / positives
    weights = np.where(target == 1, positive_weight, 1.0)
    model = MLPClassifier(
        hidden_layer_sizes=tuple(int(value) for value in mlp_cfg["hidden_layers"]),
        max_iter=int(mlp_cfg["max_epochs"]),
        batch_size=int(mlp_cfg["batch_size"]),
        n_iter_no_change=int(mlp_cfg["patience"]),
        learning_rate_init=float(mlp_cfg["learning_rate_init"]),
        alpha=float(mlp_cfg["alpha"]),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=int(cfg["project"]["random_seed"]),
        verbose=True,
    )
    model.fit(matrix, target, sample_weight=weights)
    del matrix, weights
    gc.collect()
    calibration_matrix = preprocessor.transform(calibration)
    raw_calibration = model.predict_proba(calibration_matrix)[:, 1]
    del calibration_matrix
    calibrator = fit_calibrator(
        raw_calibration,
        calibration["y_fire"].astype(int).to_numpy(),
        int(cfg["model_buckets"]["min_calibration_positives"]),
    )
    return MLPBundle(
        model=model,
        calibrator=calibrator,
        preprocessor=preprocessor,
        feature_columns=features,
        stage="C",
        model_bucket="all_months",
        random_seed=int(cfg["project"]["random_seed"]),
    )


def write_lgbm_importance(bundle: LightGBMBundle, path: Path) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def train_lightgbm_candidates(
    cfg: dict, archive: Path, output: Path, force: bool
) -> tuple[dict, dict[str, list[str]]]:
    candidates: dict = {}
    feature_sets: dict[str, list[str]] = {}
    for stage in STAGES:
        name = f"lightgbm_stage_{stage}"
        model_path = output / "models" / f"{name}.joblib"
        log(f"[{name}] loading archived tables")
        features = declared_features(archive, stage)
        train, train_cleanup = load_and_clean(
            archive, stage, "train", features
        )
        validation, validation_cleanup = load_and_clean(
            archive, stage, "val", features
        )
        tune, calibration = split_validation(validation)
        del validation
        features, constants = remove_constant_features(train, features)
        feature_sets[name] = features
        if model_path.exists() and not force:
            log(f"[{name}] loading existing model")
            bundle = joblib.load(model_path)
        else:
            log(
                f"[{name}] fitting {len(train):,} rows, "
                f"{len(features)} features"
            )
            bundle = fit_lightgbm(
                cfg,
                train,
                tune,
                calibration,
                features,
                stage.upper(),
                "all_months",
            )
            bundle.save(model_path)
        log(f"[{name}] evaluating 2023 tune and 2024 calibration")
        tune_raw = lgbm_raw(bundle, tune)
        tune_calibrated = bundle.calibrator.predict(tune_raw)
        calibration_raw = lgbm_raw(bundle, calibration)
        calibration_calibrated = bundle.calibrator.predict(calibration_raw)
        candidates[name] = {
            "family": "lightgbm",
            "stage": stage.upper(),
            "model_path": str(model_path),
            "feature_count": len(features),
            "constant_features_removed": constants,
            "best_iteration": int(bundle.best_iteration),
            "calibration": getattr(bundle.calibrator, "reason", "isotonic"),
            "train_cleanup": train_cleanup,
            "validation_cleanup": validation_cleanup,
            "tune_2023": raw_and_calibrated_metrics(
                tune,
                tune_raw,
                tune_calibrated,
                int(cfg["training"]["top_k_per_day"]),
            ),
            "calibration_2024": raw_and_calibrated_metrics(
                calibration,
                calibration_raw,
                calibration_calibrated,
                int(cfg["training"]["top_k_per_day"]),
            ),
        }
        write_lgbm_importance(
            bundle, output / "explainability" / f"{name}_importance.csv"
        )
        del (
            train,
            tune,
            calibration,
            tune_raw,
            tune_calibrated,
            calibration_raw,
            calibration_calibrated,
            bundle,
        )
        gc.collect()
        atomic_json(candidates, output / "candidate_metrics.partial.json")
    return candidates, feature_sets


def train_mlp_candidate(
    cfg: dict,
    archive: Path,
    output: Path,
    force: bool,
    candidates: dict,
    feature_sets: dict[str, list[str]],
) -> None:
    name = "mlp_stage_c"
    model_path = output / "models" / f"{name}.joblib"
    declared = declared_features(archive, "c")
    log(f"[{name}] loading archived tables")
    train, train_cleanup = load_and_clean(archive, "c", "train", declared)
    validation, validation_cleanup = load_and_clean(
        archive, "c", "val", declared
    )
    tune, calibration = split_validation(validation)
    del validation
    features = feature_sets["lightgbm_stage_c"]
    if model_path.exists() and not force:
        log(f"[{name}] loading existing model")
        bundle = joblib.load(model_path)
    else:
        log(
            f"[{name}] fitting {len(train):,} rows, {len(features)} features "
            "with balanced sample weights"
        )
        bundle = fit_weighted_mlp(cfg, train, calibration, features)
        bundle.save(model_path)
    log(f"[{name}] evaluating 2023 tune and 2024 calibration")
    tune_raw = mlp_raw(bundle, tune)
    tune_calibrated = bundle.calibrator.predict(tune_raw)
    calibration_raw = mlp_raw(bundle, calibration)
    calibration_calibrated = bundle.calibrator.predict(calibration_raw)
    candidates[name] = {
        "family": "mlp",
        "stage": "C",
        "model_path": str(model_path),
        "feature_count": len(features),
        "constant_features_removed": sorted(set(declared) - set(features)),
        "epochs": int(bundle.model.n_iter_),
        "calibration": getattr(bundle.calibrator, "reason", "isotonic"),
        "balanced_sample_weight": True,
        "train_cleanup": train_cleanup,
        "validation_cleanup": validation_cleanup,
        "tune_2023": raw_and_calibrated_metrics(
            tune,
            tune_raw,
            tune_calibrated,
            int(cfg["training"]["top_k_per_day"]),
        ),
        "calibration_2024": raw_and_calibrated_metrics(
            calibration,
            calibration_raw,
            calibration_calibrated,
            int(cfg["training"]["top_k_per_day"]),
        ),
    }
    del (
        train,
        tune,
        calibration,
        tune_raw,
        tune_calibrated,
        calibration_raw,
        calibration_calibrated,
        bundle,
    )
    gc.collect()
    atomic_json(candidates, output / "candidate_metrics.partial.json")


def evaluate_selected(
    cfg: dict,
    archive: Path,
    output: Path,
    candidates: dict,
    feature_sets: dict[str, list[str]],
) -> tuple[str, dict]:
    selected_name = max(
        candidates,
        key=lambda name: np.nan_to_num(
            candidates[name]["tune_2023"]["raw"]["pr_auc"], nan=-1.0
        ),
    )
    selected = candidates[selected_name]
    stage = selected["stage"].lower()
    log(
        f"[selection] {selected_name} selected by 2023 raw PR-AUC; "
        "opening 2025 test for the first time"
    )
    declared = declared_features(archive, stage)
    test, cleanup = load_and_clean(archive, stage, "test", declared)
    bundle = joblib.load(selected["model_path"])
    if selected["family"] == "lightgbm":
        raw = lgbm_raw(bundle, test)
    else:
        raw = mlp_raw(bundle, test)
    calibrated = bundle.calibrator.predict(raw)
    scored = compact_scored(test, raw, calibrated)
    test_metrics = raw_and_calibrated_metrics(
        test,
        raw,
        calibrated,
        int(cfg["training"]["top_k_per_day"]),
    )
    test_metrics["cleanup"] = cleanup
    test_metrics["by_month_calibrated"] = metrics_by_group(
        scored, "month", int(cfg["training"]["top_k_per_day"])
    ) if "month" in scored else metrics_by_group(
        scored.assign(month=scored["label_date"].dt.month),
        "month",
        int(cfg["training"]["top_k_per_day"]),
    )
    atomic_parquet(scored, output / "test_predictions.parquet")
    plot_calibration_and_pr(
        scored,
        output / "plots" / "calibration_and_pr_2025.png",
        f"{selected_name} — 2025 test",
    )
    risk_day = scored.groupby("label_date")["p_fire"].max().idxmax()
    plot_risk_map(
        scored,
        output / "plots" / "risk_map_peak_day_2025.png",
        risk_day,
    )
    if selected["family"] == "lightgbm":
        write_lightgbm_explanations(
            bundle,
            test,
            output / "explainability" / "selected_model",
            max_rows=int(cfg["training"]["max_explain_rows"]),
        )
    del test, bundle, raw, calibrated, scored
    gc.collect()
    return selected_name, test_metrics


def write_report(
    output: Path,
    selected_name: str,
    candidates: dict,
    test_metrics: dict,
) -> None:
    selected = candidates[selected_name]
    tune = selected["tune_2023"]["raw"]
    calibration = selected["calibration_2024"]["calibrated"]
    test = test_metrics["calibrated"]
    test_raw = test_metrics["raw"]
    prevalence = test["prevalence"]
    pr_lift = test["pr_auc"] / prevalence
    alert_lift = test["alert_precision"] / prevalence
    candidate_rows = []
    for name, result in sorted(candidates.items()):
        candidate_rows.append(
            "| {name} | {features} | {pr:.6f} | {roc:.6f} | {brier:.6f} |".format(
                name=name,
                features=result["feature_count"],
                pr=result["tune_2023"]["raw"]["pr_auc"],
                roc=result["tune_2023"]["raw"]["roc_auc"],
                brier=result["calibration_2024"]["calibrated"]["brier"],
            )
        )
    lines = [
        "# Archive Model Training Report",
        "",
        f"**Completed:** {datetime.now(UTC).isoformat()}  ",
        f"**Selected model:** `{selected_name}`  ",
        "**Selection rule:** highest raw PR-AUC on 2023; 2025 remained unopened until selection.",
        "",
        "## Candidate selection",
        "",
        "| Candidate | Features | 2023 PR-AUC | 2023 ROC-AUC | 2024 calibrated Brier |",
        "|---|---:|---:|---:|---:|",
        *candidate_rows,
        "",
        "## Selected-model metrics",
        "",
        "| Split | PR-AUC | ROC-AUC | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
        (
            f"| 2023 tune (raw) | {tune['pr_auc']:.6f} | {tune['roc_auc']:.6f} "
            f"| {tune['brier']:.6f} | {tune['ece_10bin']:.6f} |"
        ),
        (
            f"| 2024 calibration | {calibration['pr_auc']:.6f} "
            f"| {calibration['roc_auc']:.6f} | {calibration['brier']:.6f} "
            f"| {calibration['ece_10bin']:.6f} |"
        ),
        (
            f"| 2025 test (raw ranking) | {test_raw['pr_auc']:.6f} "
            f"| {test_raw['roc_auc']:.6f} | {test_raw['brier']:.6f} "
            f"| {test_raw['ece_10bin']:.6f} |"
        ),
        (
            f"| 2025 test (calibrated) | {test['pr_auc']:.6f} | {test['roc_auc']:.6f} "
            f"| {test['brier']:.6f} | {test['ece_10bin']:.6f} |"
        ),
        "",
        (
            f"On 2025, calibrated PR-AUC is **{pr_lift:.2f}×** the "
            f"{prevalence:.4%} prevalence baseline. The daily top-25 alerts have "
            f"**{test['alert_precision']:.4%}** precision ({alert_lift:.2f}× lift) "
            f"and capture **{test['positive_recall']:.4%}** of positive cell-days."
        ),
        "",
        "The 2024 calibration metrics are in-sample for the isotonic calibrator; "
        "the 2025 calibration metrics are the unbiased probability-quality check. "
        "Isotonic ties can lower PR-AUC even when calibration improves, so raw and "
        "calibrated 2025 ranking metrics are both reported.",
        "",
        "## Safeguards applied",
        "",
        "- Exact audited feature allowlists; FIRMS outcome columns were never model inputs.",
        "- S2 observations older than 15 days were zeroed and marked unavailable.",
        "- S5P observations older than 2 days were zeroed and marked unavailable.",
        "- Unavailable S2/S5P measurement blocks were zeroed.",
        "- Small negative soil-moisture values were clipped to zero.",
        "- Constant training features were removed.",
        "- 2021 S5P rows retained zero placeholders plus availability indicators.",
        "",
        "The test predictions, plots, explanations, model bundles, and full JSON metrics "
        "are stored beside this report.",
        "",
    ]
    (output / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_manifest(
    archive: Path,
    output: Path,
    selected_name: str,
    candidates: dict,
) -> None:
    selected_model_path = Path(candidates[selected_name]["model_path"])
    artifacts = {
        "selected_model": selected_model_path,
        "test_predictions": output / "test_predictions.parquet",
        "metrics": output / "metrics.json",
        "training_report": output / "TRAINING_REPORT.md",
        "calibration_plot": output / "plots" / "calibration_and_pr_2025.png",
        "risk_map": output / "plots" / "risk_map_peak_day_2025.png",
    }
    archive_meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected_name,
        "selection_rule": "maximum raw PR-AUC on label year 2023",
        "software": {
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "archive": str(archive),
            "stage_c_input_sha256": archive_meta["sha256"],
            "splits": archive_meta["splits"],
            "semantics": {
                "forecast_day_D": "eo_asof_date",
                "target": "y_fire on D+1",
                "era5_cutoff": "D-5",
            },
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
    atomic_json(payload, output / "run_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = args.archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    log("[run] training archive models with no GCS access")
    candidates, feature_sets = train_lightgbm_candidates(
        cfg, archive, output, args.force
    )
    if not args.skip_mlp:
        train_mlp_candidate(
            cfg, archive, output, args.force, candidates, feature_sets
        )
    selected_name, test_metrics = evaluate_selected(
        cfg, archive, output, candidates, feature_sets
    )
    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(archive),
        "selection_rule": "maximum 2023 raw PR-AUC",
        "selected_model": selected_name,
        "candidates": candidates,
        "test_2025": test_metrics,
        "data_semantics": {
            "forecast_day_D": "eo_asof_date",
            "target": "y_fire on D+1",
            "era5_cutoff": "D-5",
        },
    }
    atomic_json(payload, output / "metrics.json")
    write_report(output, selected_name, candidates, test_metrics)
    write_run_manifest(archive, output, selected_name, candidates)
    log(f"[done] selected={selected_name}; report={output / 'TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
