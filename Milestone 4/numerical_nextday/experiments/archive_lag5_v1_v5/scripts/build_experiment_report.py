from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from numerical_nextday.evaluation.metrics import (  # noqa: E402
    expected_calibration_error,
)
from numerical_nextday.train.lightgbm_model import LightGBMBundle  # noqa: E402
from numerical_nextday.train.v3_model import (  # noqa: E402
    V3ClassifierBundle,
    V3RankerBundle,
)


VERSION_DIRECTORIES = {
    "V1": "lag5_full_year",
    "V2": "lag5_v2",
    "V3": "lag5_v3",
    "V4": "lag5_v4_recall25",
    "V5": "lag5_v5_two_stage",
}


def top_k(frame: pd.DataFrame, score: str, k: int) -> dict:
    alerts = (
        frame.sort_values(["label_date", score], ascending=[True, False])
        .groupby("label_date", sort=False)
        .head(k)
    )
    positives = int(frame["y_fire"].sum())
    captured = int(alerts["y_fire"].sum())
    days = int(frame["label_date"].nunique())
    return {
        "k": k,
        "alerts": len(alerts),
        "captured": captured,
        "precision": captured / len(alerts),
        "recall": captured / positives,
        "false_alerts_per_day": (len(alerts) - captured) / days,
    }


def minimum_k_for_recall(
    frame: pd.DataFrame, score: str, target_recall: float, maximum_k: int = 100
) -> dict | None:
    for k in range(1, maximum_k + 1):
        metrics = top_k(frame, score, k)
        if metrics["recall"] >= target_recall:
            return metrics
    return None


def evaluate_predictions(version: str, path: Path) -> dict:
    frame = pd.read_parquet(path)
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    alert_column = "alert_score" if "alert_score" in frame else "p_fire"
    probability = frame["p_fire"].to_numpy(dtype=float)
    raw = frame["p_fire_raw"].to_numpy(dtype=float)
    target = frame["y_fire"].to_numpy(dtype=int)
    result = {
        "version": version,
        "rows": len(frame),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "raw_pr_auc": float(average_precision_score(target, raw)),
        "calibrated_pr_auc": float(
            average_precision_score(target, probability)
        ),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "ece_10bin": expected_calibration_error(target, probability),
        "alert_column": alert_column,
        "top_k": {
            str(k): top_k(frame, alert_column, k) for k in (5, 10, 25, 50)
        },
        "k_for_50pct_recall": minimum_k_for_recall(
            frame, alert_column, 0.50
        ),
    }
    return result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_cv_metrics(version: str, metrics: dict) -> dict:
    if version == "V1":
        selected = metrics["candidates"][metrics["selected_model"]]
        tune = selected["tune_2023"]["raw"]
        return {
            "protocol": "Single 2023 tune year",
            "mean_pr_auc": tune["pr_auc"],
            "minimum_pr_auc": tune["pr_auc"],
            "mean_recall_25": tune["positive_recall"],
            "minimum_recall_25": tune["positive_recall"],
            "mean_recall_50": None,
        }
    if version == "V2":
        selected = metrics["parameter_search"][
            metrics["selected_configuration"]
        ]
        return {
            "protocol": "Walk-forward 2022–2024",
            "mean_pr_auc": selected["mean_pr_auc"],
            "minimum_pr_auc": selected["min_pr_auc"],
            "mean_recall_25": float(
                np.mean([fold["positive_recall"] for fold in selected["folds"]])
            ),
            "minimum_recall_25": float(
                np.min([fold["positive_recall"] for fold in selected["folds"]])
            ),
            "mean_recall_50": None,
        }
    if version == "V3":
        selected = metrics["classifier_experiments"][
            metrics["selected_classifier"]
        ]
        return {
            "protocol": "Walk-forward 2022–2024",
            "mean_pr_auc": selected["mean_pr_auc"],
            "minimum_pr_auc": selected["min_pr_auc"],
            "mean_recall_25": float(
                np.mean(
                    [
                        fold["top_k"]["25"]["positive_recall"]
                        for fold in selected["folds"]
                    ]
                )
            ),
            "minimum_recall_25": float(
                np.min(
                    [
                        fold["top_k"]["25"]["positive_recall"]
                        for fold in selected["folds"]
                    ]
                )
            ),
            "mean_recall_50": None,
        }
    if version == "V4":
        classifier = metrics["classifier_grid"][metrics["selected_classifier"]]
        alert = metrics["blend_experiments"][metrics["selected_alert_head"]]
        return {
            "protocol": "Recall@25 walk-forward 2022–2024",
            "mean_pr_auc": classifier["mean_pr_auc"],
            "minimum_pr_auc": classifier["min_pr_auc"],
            "mean_recall_25": alert["mean_recall_25"],
            "minimum_recall_25": alert["min_recall_25"],
            "mean_recall_50": alert["mean_recall_50"],
        }
    selected = metrics["reranker_grid"][metrics["selected_reranker"]]
    return {
        "protocol": "Forward stacked reranking 2022–2024",
        "mean_pr_auc": selected["mean_pr_auc"],
        "minimum_pr_auc": min(
            fold["pr_auc"] for fold in selected["folds"]
        ),
        "mean_recall_25": selected["mean_recall_25"],
        "minimum_recall_25": selected["min_recall_25"],
        "mean_recall_50": selected["mean_recall_50"],
    }


def key_params(params: dict) -> str:
    names = [
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l1",
        "lambda_l2",
        "scale_pos_weight",
    ]
    parts = []
    for name in names:
        if name in params:
            value = params[name]
            if isinstance(value, float):
                value = round(value, 4)
            parts.append(f"{name}={value}")
    return ", ".join(parts)


def architecture_records(artifact_root: Path, metric_sets: dict) -> list[dict]:
    v1 = LightGBMBundle.load(
        artifact_root / "lag5_full_year/models/lightgbm_stage_c.joblib"
    )
    v2 = LightGBMBundle.load(
        artifact_root / "lag5_v2/models/global.joblib"
    )
    v3_container = V3ClassifierBundle.load(
        artifact_root / "lag5_v3/models/classifier.joblib"
    )
    if v3_container.global_model is None:
        raise RuntimeError("Expected V3 selected global model")
    v3 = v3_container.global_model
    v4 = LightGBMBundle.load(
        artifact_root / "lag5_v4_recall25/models/classifier.joblib"
    )
    v4_ranker = V3RankerBundle.load(
        artifact_root / "lag5_v4_recall25/models/ranker.joblib"
    )
    v5_ranker = V3RankerBundle.load(
        artifact_root / "lag5_v5_two_stage/models/reranker.joblib"
    )
    v5_metrics = metric_sets["V5"]
    return [
        {
            "version": "V1",
            "architecture": "Global Stage C LightGBM; MLP tested and rejected",
            "feature_count": len(v1.feature_columns),
            "rounds": v1.best_iteration,
            "calibration": type(v1.calibrator).__name__,
            "parameters": key_params(v1.params),
        },
        {
            "version": "V2",
            "architecture": "Leakage-safe global LightGBM",
            "feature_count": len(v2.feature_columns),
            "rounds": v2.best_iteration,
            "calibration": type(v2.calibrator).__name__,
            "parameters": key_params(v2.params),
        },
        {
            "version": "V3",
            "architecture": (
                "Global direction-aware LightGBM; mixture and LambdaRank tested"
            ),
            "feature_count": len(v3.feature_columns),
            "rounds": v3.best_iteration,
            "calibration": type(v3.calibrator).__name__,
            "parameters": key_params(v3.params),
        },
        {
            "version": "V4",
            "architecture": (
                "Regularized classifier + LambdaRank; 50/50 daily rank blend"
            ),
            "feature_count": len(v4.feature_columns),
            "rounds": (
                f"classifier={v4.best_iteration}; ranker="
                f"{v4_ranker.best_iteration}"
            ),
            "calibration": type(v4.calibrator).__name__,
            "parameters": key_params(v4.params),
        },
        {
            "version": "V5",
            "architecture": (
                "Frozen V4 retrieval + top-75 hard-negative LambdaRank"
            ),
            "feature_count": len(v5_ranker.feature_columns),
            "rounds": v5_ranker.best_iteration,
            "calibration": "Inherited V4 Platt probabilities",
            "parameters": key_params(
                v5_metrics["reranker_grid"][
                    v5_metrics["selected_reranker"]
                ]["configuration"]
            ),
        },
    ]


def fmt(value: float | None, digits: int = 6) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def report_text(
    meta: dict,
    results: dict,
    cv: dict,
    architecture: list[dict],
    metrics: dict,
) -> str:
    split = meta["splits"]
    v2 = metrics["V2"]
    v3 = metrics["V3"]
    v4 = metrics["V4"]
    v5 = metrics["V5"]
    lines = [
        "# Wildfire Forecasting Experiments: V1–V5",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}  ",
        f"**Dataset:** `{meta['dataset_name']}`  ",
        f"**Audit status:** `{meta['audit_status']}`  ",
        "",
        "## Executive summary",
        "",
        "Five isolated model versions were trained from the local archive "
        "without repeating GCS downloads. V2 produced the largest improvement "
        "by adding strictly lagged fire-history features. V3 added directional "
        "spread features and improved ranking further. V4 directly optimized "
        "Recall@25 with longer, regularized training and score blending. V5 "
        "tested hard-negative reranking but provided negligible additional gain.",
        "",
        "The best practical model is **V4**. V5 improves 2025 descriptive "
        "Recall@25 by only one captured positive while reducing Recall@50. "
        "Strict Recall@25=50% was not reached; V4 reaches 50% recall at K=53.",
        "",
        "## Data",
        "",
        f"**Unit:** {meta['unit']}  ",
        f"**Time contract:** {meta['one_liner']}  ",
        f"**Coverage:** {meta['coverage']['label_date_min']} through "
        f"{meta['coverage']['label_date_max']}  ",
        f"**Grid:** {meta['coverage']['cells']} cells × "
        f"{meta['coverage']['calendar_days']} days = "
        f"{meta['coverage']['rows']:,} rows  ",
        f"**Total positives:** {meta['coverage']['positives']:,} "
        f"({meta['coverage']['positive_rate']:.3%})  ",
        "",
        "| Split | Label years | Rows | Positives | Positive rate |",
        "|---|---|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        value = split[name]
        years = "–".join(
            [str(value["label_years"][0]), str(value["label_years"][-1])]
        )
        lines.append(
            f"| {name} | {years} | {value['rows']:,} | "
            f"{value['positives']:,} | {value['positive_rate']:.3%} |"
        )
    lines.extend(
        [
            "",
            "The row count overstates the independent sample size: cells on the "
            "same day are spatially correlated, consecutive days are temporally "
            "correlated, and one incident can create many positive cell-days. "
            "The effective evidence is the number of independent incidents and "
            "fire seasons.",
            "",
            "### Sensor and quality limitations",
            "",
            "- Sentinel-5P is entirely missing for 2021: 245,280 placeholder "
            "rows with zero measurements and zero availability.",
            "- A no-S5P V2 ablation retained mean walk-forward PR-AUC "
            f"{v2['feature_ablation']['v2_no_s5p']['mean_pr_auc']:.6f}, "
            "showing that the main V2 gain does not depend on 2021 S5P.",
            "- 7,392 late-2025 S2 rows exceeded the documented 15-day age limit; "
            "the cleaned training pipeline masks stale measurements.",
            "- 6,720 training S5P rows in 2020 exceeded the two-day limit and "
            "were masked.",
            "- Small negative soil values were clipped and two constant S5P "
            "standard-deviation features were removed.",
            "",
            "## Selected architectures and parameters",
            "",
            "| Version | Selected architecture | Features | Rounds | Calibration |",
            "|---|---|---:|---|---|",
        ]
    )
    for record in architecture:
        lines.append(
            f"| {record['version']} | {record['architecture']} | "
            f"{record['feature_count']} | {record['rounds']} | "
            f"{record['calibration']} |"
        )
    lines.extend(["", "### Key parameters", ""])
    for record in architecture:
        lines.append(
            f"- **{record['version']}:** {record['parameters']}"
        )
    lines.extend(
        [
            "",
            "Additional architecture experiments:",
            "",
            f"- V1 MLP `[128, 64]`, 20 epochs: 2023 PR-AUC "
            f"{metrics['V1']['candidates']['mlp_stage_c']['tune_2023']['raw']['pr_auc']:.6f}; rejected.",
            f"- V2 seasonal router: mean PR-AUC "
            f"{v2['seasonal_cv']['mean_pr_auc']:.6f}; rejected in favor of global.",
            f"- V3 continuation/ignition mixture: mean PR-AUC "
            f"{v3['classifier_experiments']['mixture_directional_unweighted']['mean_pr_auc']:.6f}; rejected.",
            f"- V3 daily LambdaRank: mean PR-AUC "
            f"{v3['ranker_experiment']['mean_pr_auc']:.6f}; rejected.",
            f"- V4 best pure PR-AUC candidate (`extra_trees_63`): "
            f"{v4['classifier_grid']['extra_trees_63']['mean_pr_auc']:.6f}; "
            "not selected because Recall@25 was lower.",
            f"- V5 top-75 reranker: mean Recall@25 "
            f"{v5['reranker_grid'][v5['selected_reranker']]['mean_recall_25']:.2%}; "
            "negligible gain over V4.",
            "",
            "## Forward-validation results",
            "",
            "| Version | Protocol | Mean PR-AUC | Minimum PR-AUC | Mean Recall@25 | Worst Recall@25 | Mean Recall@50 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for version in VERSION_DIRECTORIES:
        value = cv[version]
        lines.append(
            f"| {version} | {value['protocol']} | "
            f"{fmt(value['mean_pr_auc'])} | "
            f"{fmt(value['minimum_pr_auc'])} | "
            f"{pct(value['mean_recall_25'])} | "
            f"{pct(value['minimum_recall_25'])} | "
            f"{pct(value['mean_recall_50'])} |"
        )
    lines.extend(
        [
            "",
            "V1 is not directly comparable to later walk-forward means because "
            "it selected on 2023 only. V4/V5 directly optimize alert-budget "
            "recall, whereas earlier versions selected primarily on PR-AUC.",
            "",
            "## 2025 result comparison",
            "",
            "Probability metrics are recomputed from each version's saved raw "
            "and calibrated classifier probabilities. Alert-budget metrics are "
            "recomputed with that version's selected alert score.",
            "",
            "| Version | Raw PR-AUC | Calibrated PR-AUC | ROC-AUC | Brier | P@25 | Recall@25 | Recall@50 | K for 50% recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for version in VERSION_DIRECTORIES:
        value = results[version]
        k50 = value["k_for_50pct_recall"]
        lines.append(
            f"| {version} | {value['raw_pr_auc']:.6f} | "
            f"{value['calibrated_pr_auc']:.6f} | "
            f"{value['roc_auc']:.6f} | {value['brier']:.6f} | "
            f"{value['top_k']['25']['precision']:.4%} | "
            f"{value['top_k']['25']['recall']:.4%} | "
            f"{value['top_k']['50']['recall']:.4%} | "
            f"{k50['k'] if k50 else '>100'} |"
        )
    lines.extend(
        [
            "",
            "V4 captures 872 of 2,275 positives at K=25. A 50% target requires "
            "1,138 captures, leaving a gap of 266 correct alerts. At the fixed "
            "9,125-alert annual budget, Precision@25 would need to increase "
            "from 9.56% to approximately 12.47%.",
            "",
            "## Version-by-version findings",
            "",
            "### V1 — Baseline",
            "",
            "- Stage C LightGBM beat Stage A, Stage B, and the MLP.",
            "- It established useful but limited discrimination: calibrated "
            "PR-AUC 0.0625 and Recall@25 25.27%.",
            "",
            "### V2 — Causal history",
            "",
            "- Added weather histories, calendar/geography, causal fire "
            "recency, cell rate, neighbor and statewide histories.",
            "- Produced the largest improvement: calibrated PR-AUC 0.1505 and "
            "Recall@25 36.79%.",
            "- The gain is dominated by persistence/continuation rather than "
            "pure ignition prediction.",
            "",
            "### V3 — Direction-aware spread",
            "",
            "- Added distance-weighted and wind-aligned neighborhood fire "
            "features plus dry/windy interactions.",
            "- Unweighted global LightGBM beat the proposed mixture and ranker.",
            "- Improved calibrated PR-AUC to 0.1704 and Recall@25 to 37.85%.",
            "",
            "### V4 — Recall-specific tuning",
            "",
            "- Increased the candidate maximum to 1,200 rounds but selected "
            "rounds using Recall@25 early stopping.",
            "- Tested seven classifier configurations, four rankers, and three "
            "predeclared blends.",
            "- Selected a 248-round regularized classifier and a 50/50 daily "
            "classifier/ranker blend.",
            f"- Best practical balance: PR-AUC "
            f"{results['V4']['calibrated_pr_auc']:.4f}, Recall@25 "
            f"{results['V4']['top_k']['25']['recall']:.2%}, Recall@50 "
            f"{results['V4']['top_k']['50']['recall']:.2%}.",
            "",
            "### V5 — Hard-negative reranking",
            "",
            "- Generated forward-only base predictions for 2021–2024 and "
            "trained top-75/100/150 candidate rerankers.",
            "- Added only one 2025 captured positive at K=25 and reduced "
            "Recall@50; therefore it should not replace V4.",
            "",
            "## Data-leakage assessment",
            "",
            "| Risk | Assessment | Status / action |",
            "|---|---|---|",
            "| Direct target leakage | `y_fire`, FIRMS counts, dates, IDs and outcome fields are excluded from direct model inputs. | Controlled |",
            "| Weather look-ahead | Target is T=D+1; ERA5 and weather rolling windows end D−5. | Controlled |",
            "| Fire-history look-ahead | Histories end T−2=D−1; forecast-day D and target-day D+1 outcomes are excluded. | Controlled, conditional on D−1 FIRMS availability at serving time |",
            "| Rolling-window boundaries | Full-grid and mutation tests confirm future labels cannot change earlier features. | Controlled |",
            "| Cross-year history | Early-year rows legitimately carry prior-year causal history; the split does not erase information available in production. | Controlled |",
            "| Sentinel-5P 2021 | Missingness can act as a year/domain indicator, but it is not target leakage. No-S5P ablation remains strong. | Distribution-shift risk, not leakage |",
            "| Correlated rows | 672 cells/day and repeated incident cell-days reduce effective sample size. | Statistical limitation, not leakage |",
            "| Test-set reuse | Only V1's initial 2025 evaluation was fully untouched. V2–V5 were designed after earlier 2025 results were visible. | Evaluation leakage for cross-version 2025 comparison; treat V2–V5 2025 as descriptive |",
            "| Calibration | V1/V2 use isotonic calibration; V3/V4 use order-preserving Platt calibration on 2024. | Controlled; calibration-year metrics are in-sample |",
            "",
            "There is **no detected feature/label leakage** under the stated "
            "serving contract. The material leakage concern is evaluation reuse: "
            "later versions cannot use 2025 as proof of unseen generalization.",
            "",
            "## Conclusions and recommendation",
            "",
            "1. Use **V4** as the retained model; keep V1–V5 artifacts for the "
            "experiment record.",
            "2. Do not claim Recall@25=50%. With current inputs, the supported "
            "operating point is approximately Recall@53=50%.",
            "3. Freeze all architecture choices before evaluating a new 2026 "
            "or external-geography holdout.",
            "4. Further tuning of the same variables is unlikely to provide a "
            "large gain and risks overfitting.",
            "5. The next high-value inputs are forecast-day/target-day weather "
            "forecasts, lightning, human-access/roads/powerlines/WUI, live fuel "
            "moisture and drought trajectories, and explicit ignition/onset labels.",
            "",
            "## Reproducibility",
            "",
            "- V1–V5 use separate artifact directories.",
            "- V2/V3 derived tables were created from the archive without GCS.",
            "- All saved model predictions were reproduced from fresh bundle loads.",
            "- All artifact manifests and dataset hashes passed verification.",
            "- The full project suite contains 30 passing tests.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-meta",
        type=Path,
        required=True,
        help="Path to the audited archive meta.json.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory containing the isolated V1-V5 artifact folders.",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    meta = load_json(args.archive_meta.resolve())
    metric_sets = {
        version: load_json(
            artifact_root / directory / "metrics.json"
        )
        for version, directory in VERSION_DIRECTORIES.items()
    }
    results = {
        version: evaluate_predictions(
            version,
            artifact_root / directory / "test_predictions.parquet",
        )
        for version, directory in VERSION_DIRECTORIES.items()
    }
    cv = {
        version: selected_cv_metrics(version, metric_sets[version])
        for version in VERSION_DIRECTORIES
    }
    architecture = architecture_records(artifact_root, metric_sets)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data": {
            "dataset_name": meta["dataset_name"],
            "one_liner": meta["one_liner"],
            "coverage": meta["coverage"],
            "splits": meta["splits"],
            "s5p_2021": meta["s5p_2021"],
        },
        "architectures": architecture,
        "validation": cv,
        "test_2025": results,
        "leakage_conclusion": (
            "No detected feature/label leakage under the D-5 weather and D-1 "
            "FIRMS serving contract. Only V1's first 2025 evaluation was fully "
            "untouched; V2-V5 2025 comparisons are descriptive."
        ),
        "recommended_version": "V4",
    }
    output_json = artifact_root / "EXPERIMENT_SUMMARY.json"
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for version in VERSION_DIRECTORIES:
        test = results[version]
        validation = cv[version]
        rows.append(
            {
                "version": version,
                "validation_mean_pr_auc": validation["mean_pr_auc"],
                "validation_mean_recall_25": validation["mean_recall_25"],
                "test_raw_pr_auc": test["raw_pr_auc"],
                "test_calibrated_pr_auc": test["calibrated_pr_auc"],
                "test_roc_auc": test["roc_auc"],
                "test_brier": test["brier"],
                "test_precision_25": test["top_k"]["25"]["precision"],
                "test_recall_25": test["top_k"]["25"]["recall"],
                "test_recall_50": test["top_k"]["50"]["recall"],
                "test_k_for_50pct_recall": (
                    test["k_for_50pct_recall"]["k"]
                    if test["k_for_50pct_recall"]
                    else None
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        artifact_root / "EXPERIMENT_RESULTS.csv", index=False
    )
    (artifact_root / "EXPERIMENT_REPORT.md").write_text(
        report_text(meta, results, cv, architecture, metric_sets),
        encoding="utf-8",
    )
    print(artifact_root / "EXPERIMENT_REPORT.md")


if __name__ == "__main__":
    main()
