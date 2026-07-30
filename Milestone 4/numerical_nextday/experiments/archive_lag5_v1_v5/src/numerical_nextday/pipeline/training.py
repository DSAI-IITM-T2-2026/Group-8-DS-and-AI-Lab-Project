from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import load_feature_columns, load_splits
from ..evaluation.explain import write_lightgbm_explanations
from ..evaluation.metrics import complete_metrics, metrics_by_group
from ..evaluation.plots import plot_calibration_and_pr, plot_risk_map
from ..io import atomic_json, atomic_parquet
from ..train.lightgbm_model import LightGBMBundle, fit_lightgbm
from ..train.mlp_model import fit_mlp
from ..train.router import MonthRouter
from ..train.sweep import EXPERIMENTS

logger = logging.getLogger(__name__)


def _model_root(cfg: dict, lag: int) -> Path:
    return cfg["paths"]["artifact_dir"] / "models" / f"lag{lag}"


def _filter_bucket(splits: dict[str, pd.DataFrame], bucket: str) -> dict[str, pd.DataFrame]:
    return {
        name: frame.loc[frame["model_bucket"] == bucket].reset_index(drop=True)
        for name, frame in splits.items()
    }


def _score(bundle, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["p_fire"] = bundle.predict_proba(scored).astype("float32")
    scored["confidence_pct"] = (100 * scored["p_fire"]).astype("float32")
    return scored


def _fit_and_record(
    cfg: dict,
    stage: str,
    lag: int,
    bucket: str,
    experiment: str,
    overrides: dict | None = None,
) -> tuple[LightGBMBundle, dict]:
    splits = _filter_bucket(load_splits(cfg, stage, lag), bucket)
    features = load_feature_columns(cfg, stage, lag)
    bundle = fit_lightgbm(
        cfg,
        splits["train"],
        splits["tune"],
        splits["calibration"],
        features,
        stage,
        bucket,
        overrides,
    )
    artifact_dir = _model_root(cfg, lag) / stage.lower() / bucket / experiment
    model_path = artifact_dir / "model.joblib"
    bundle.save(model_path)
    metrics = {
        "experiment": experiment,
        "stage": stage,
        "bucket": bucket,
        "lag": lag,
        "best_iteration": bundle.best_iteration,
        "calibration": bundle.calibration_note,
        "params": bundle.params,
    }
    for split_name in ("tune", "calibration", "test"):
        scored = _score(bundle, splits[split_name])
        metrics[split_name] = complete_metrics(scored, int(cfg["training"]["top_k_per_day"]))
        if split_name == "test":
            atomic_parquet(scored, artifact_dir / "test_predictions.parquet")
    atomic_json(metrics, artifact_dir / "metrics.json")
    write_lightgbm_explanations(
        bundle,
        splits["test"],
        artifact_dir / "explainability",
        max_rows=int(cfg["training"]["max_explain_rows"]),
    )
    return bundle, {"model_path": str(model_path), **metrics}


def train_fire_season(cfg: dict, lag: int, force: bool = False) -> dict:
    selection_path = _model_root(cfg, lag) / "best_fire_season.json"
    if selection_path.exists() and not force:
        return json.loads(selection_path.read_text(encoding="utf-8"))
    defaults = []
    for stage in ("A", "B", "C"):
        _, result = _fit_and_record(cfg, stage, lag, "fire_season", f"{stage}_default")
        defaults.append(result)
    best_default = max(
        defaults,
        key=lambda result: np.nan_to_num(result["tune"]["pr_auc"], nan=-1.0),
    )
    best = best_default
    if bool(cfg["training"]["run_hyperparameter_sweep"]):
        for experiment, overrides in EXPERIMENTS.items():
            _, result = _fit_and_record(
                cfg,
                best_default["stage"],
                lag,
                "fire_season",
                experiment,
                overrides,
            )
            if np.nan_to_num(result["tune"]["pr_auc"], nan=-1.0) > np.nan_to_num(
                best["tune"]["pr_auc"], nan=-1.0
            ):
                best = result

    if bool(cfg["training"]["run_mlp"]):
        splits = _filter_bucket(load_splits(cfg, "C", lag), "fire_season")
        features = load_feature_columns(cfg, "C", lag)
        mlp = fit_mlp(
            cfg,
            splits["train"],
            splits["calibration"],
            features,
            "C",
            "fire_season",
        )
        mlp_dir = _model_root(cfg, lag) / "c" / "fire_season" / "C_mlp_default"
        mlp.save(mlp_dir / "model.joblib")
        mlp_scored = _score(mlp, splits["test"])
        atomic_json(
            complete_metrics(mlp_scored, int(cfg["training"]["top_k_per_day"])),
            mlp_dir / "metrics.json",
        )
        atomic_parquet(mlp_scored, mlp_dir / "test_predictions.parquet")
    selection = {
        "selected_by": "tune_pr_auc",
        "best_stage": best["stage"],
        "best_experiment": best["experiment"],
        "model_path": best["model_path"],
        "params": best["params"],
        "tune_metrics": best["tune"],
        "calibration_metrics": best["calibration"],
        "test_metrics": best["test"],
        "default_comparison": defaults,
    }
    atomic_json(selection, selection_path)
    return selection


def train_month_models(cfg: dict, lag: int, force: bool = False) -> dict:
    routing_path = _model_root(cfg, lag) / "routing.json"
    if routing_path.exists() and not force:
        return json.loads(routing_path.read_text(encoding="utf-8"))
    best = train_fire_season(cfg, lag, force=False)
    stage = best["best_stage"]
    routing = {
        "fire_season": {
            "model_path": best["model_path"],
            "status": "trained",
        }
    }
    minimum = int(cfg["model_buckets"]["min_train_positives"])
    for bucket in ("jan", "feb", "mar", "dec"):
        splits = _filter_bucket(load_splits(cfg, stage, lag), bucket)
        positives = int(splits["train"]["y_fire"].sum())
        tune_classes = splits["tune"]["y_fire"].nunique()
        if positives < minimum or tune_classes < 2:
            routing[bucket] = {
                "model_path": best["model_path"],
                "status": "fallback",
                "reason": (
                    f"train_positives={positives}, tune_classes={tune_classes}; "
                    "using fire_season model"
                ),
            }
            continue
        _, result = _fit_and_record(
            cfg,
            stage,
            lag,
            bucket,
            f"{stage}_{bucket}",
            overrides=_portable_overrides(
                best["params"], preserve_no_spw=best["best_experiment"] == "lgbm_no_spw"
            ),
        )
        routing[bucket] = {
            "model_path": result["model_path"],
            "status": "trained",
            "test_metrics": result["test"],
        }
    payload = {
        "lag": lag,
        "stage": stage,
        "fallback_bucket": cfg["model_buckets"]["winter_fallback"],
        "models": routing,
    }
    atomic_json(payload, routing_path)
    return payload


def _portable_overrides(params: dict, preserve_no_spw: bool = False) -> dict:
    excluded = {
        "objective",
        "metric",
        "verbosity",
        "seed",
        "num_threads",
        "scale_pos_weight",
    }
    if preserve_no_spw:
        excluded.remove("scale_pos_weight")
    return {key: value for key, value in params.items() if key not in excluded}


def evaluate_router(cfg: dict, lag: int) -> dict:
    routing = train_month_models(cfg, lag, force=False)
    stage = routing["stage"]
    models = {
        bucket: LightGBMBundle.load(Path(entry["model_path"]))
        for bucket, entry in routing["models"].items()
    }
    router = MonthRouter(models, routing["fallback_bucket"])
    test = load_splits(cfg, stage, lag)["test"].reset_index(drop=True)
    scored = test.copy()
    scored["p_fire"] = router.predict_proba(scored).astype("float32")
    scored["confidence_pct"] = (100 * scored["p_fire"]).astype("float32")
    report_dir = cfg["paths"]["report_dir"] / f"lag{lag}"
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(scored, report_dir / "routed_test_predictions.parquet")
    metrics = {
        "overall": complete_metrics(scored, int(cfg["training"]["top_k_per_day"])),
        "by_bucket": metrics_by_group(
            scored, "model_bucket", int(cfg["training"]["top_k_per_day"])
        ),
        "by_month": metrics_by_group(
            scored.assign(month=scored["label_date"].dt.month),
            "month",
            int(cfg["training"]["top_k_per_day"]),
        ),
    }
    atomic_json(metrics, report_dir / "routed_metrics.json")
    pd.DataFrame(metrics["by_bucket"]).to_csv(report_dir / "metrics_by_bucket.csv", index=False)
    pd.DataFrame(metrics["by_month"]).to_csv(report_dir / "metrics_by_month.csv", index=False)
    plot_calibration_and_pr(scored, report_dir / "calibration_and_pr.png", "Routed 2025 test")
    risk_day = scored.groupby("label_date")["p_fire"].max().idxmax()
    plot_risk_map(scored, report_dir / "risk_map_fire_season.png", risk_day)
    winter = scored.loc[scored["label_date"].dt.month.isin([1, 2, 3, 12])]
    if not winter.empty:
        winter_day = winter.groupby("label_date")["p_fire"].max().idxmax()
        plot_risk_map(scored, report_dir / "risk_map_winter.png", winter_day)
    _write_generated_report(cfg, lag, routing, metrics, report_dir)
    return metrics


def _write_generated_report(
    cfg: dict, lag: int, routing: dict, metrics: dict, report_dir: Path
) -> None:
    overall = metrics["overall"]
    lines = [
        "# Milestone 4 Generated Evaluation Report",
        "",
        f"- ERA5 lag: **{lag} days**",
        f"- Selected feature stage: **{routing['stage']}**",
        (
            f"- Label: **{cfg['task']['label_definition']}** from FIRMS confidence "
            f"≥ {cfg['task']['firms_confidence_min']}"
        ),
        f"- Test year: **{', '.join(map(str, cfg['temporal']['test_years']))}**",
        f"- Rows: **{overall['n_rows']:,}**",
        f"- Positives: **{overall['n_positives']:,}**",
        f"- PR-AUC: **{overall['pr_auc']:.6f}**",
        f"- ROC-AUC: **{overall['roc_auc']:.6f}**",
        f"- Brier score: **{overall['brier']:.6f}**",
        f"- Top-{overall['top_k_per_day']} alert precision: **{overall['alert_precision']:.6f}**",
        f"- Top-{overall['top_k_per_day']} positive recall: **{overall['positive_recall']:.6f}**",
        "",
        "## Model routing",
        "",
        "| Bucket | Status | Model |",
        "|---|---|---|",
    ]
    for bucket, entry in routing["models"].items():
        lines.append(f"| {bucket} | {entry['status']} | `{entry['model_path']}` |")
    lines.extend(
        [
            "",
            "## Interpretation warning",
            "",
            (
                "The configured label is FIRMS thermal activity, not a deduplicated "
                "ignition-event label. Performance therefore includes continuing fires and "
                "satellite detectability. Use the onset-label redesign in "
                "`WILDFIRE_MODEL_ARCHITECTURE_REVIEW.md` before claiming ignition forecasting."
            ),
            "",
        ]
    )
    (report_dir / "EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
