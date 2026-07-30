#!/usr/bin/env python3
"""Build an apples-to-apples comparison with the colleague M4 metrics."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_DIRECTORIES = {
    "Our V1": "lag5_full_year",
    "Our V2": "lag5_v2",
    "Our V3": "lag5_v3",
    "Our V4": "lag5_v4_recall25",
    "Our V5": "lag5_v5_two_stage",
}
BUCKET_MONTHS = {
    "fire_season": tuple(range(4, 12)),
    "jan": (1,),
    "feb": (2,),
    "mar": (3,),
    "dec": (12,),
}


def score(frame: pd.DataFrame, probability_column: str = "p_fire") -> dict:
    y = frame["y_fire"].astype(int).to_numpy()
    probability = frame[probability_column].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def load_our_predictions(artifact_root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    reference_keys = None
    reference_labels = None
    for name, directory in VERSION_DIRECTORIES.items():
        path = artifact_root / directory / "test_predictions.parquet"
        frame = pd.read_parquet(
            path,
            columns=["label_date", "cell_id", "y_fire", "p_fire"],
        )
        frame["label_date"] = pd.to_datetime(frame["label_date"])
        frame = frame.sort_values(["label_date", "cell_id"]).reset_index(
            drop=True
        )
        keys = frame[["label_date", "cell_id"]]
        labels = frame["y_fire"].astype(int)
        if reference_keys is None:
            reference_keys = keys
            reference_labels = labels
        else:
            if not keys.equals(reference_keys):
                raise ValueError(f"{name} does not use the same test rows")
            if not labels.equals(reference_labels):
                raise ValueError(f"{name} does not use the same test labels")
        frames[name] = frame
    return frames


def parse_colleague_mlp(report_path: Path) -> dict:
    report = report_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\|\s*`C_mlp_default`\s*\|.*?\|\s*"
        r"([0-9.]+)\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*([0-9.]+)\s*\|"
    )
    match = pattern.search(report)
    if not match:
        raise ValueError("Could not find C_mlp_default metrics in report")
    validation_pr, test_pr, test_roc = map(float, match.groups())
    return {
        "validation_pr_auc": validation_pr,
        "pr_auc": test_pr,
        "roc_auc": test_roc,
        "precision_note": "Rounded to four decimals in the colleague report",
    }


def markdown_table(
    columns: list[str], rows: list[list[str]], align: list[str] | None = None
) -> list[str]:
    if align is None:
        align = ["---"] * len(columns)
    return [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(align) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--our-artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "archive_training",
    )
    parser.add_argument(
        "--colleague-eval-metrics", required=True, type=Path
    )
    parser.add_argument(
        "--colleague-experiments-log", required=True, type=Path
    )
    parser.add_argument("--colleague-report", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "archive_training"
        / "colleague_side_by_side",
    )
    args = parser.parse_args()

    artifact_root = args.our_artifact_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    our_frames = load_our_predictions(artifact_root)

    colleague_eval = json.loads(
        args.colleague_eval_metrics.read_text(encoding="utf-8")
    )
    colleague_log = pd.read_csv(args.colleague_experiments_log)
    colleague_mlp = parse_colleague_mlp(args.colleague_report)

    fire_season_rows = []
    colleague_ids = [
        ("Colleague C default LGBM", "C_default", "Exact artifact"),
        (
            "Colleague val-selected LGBM",
            "lgbm_lr_03",
            "Exact artifact",
        ),
        (
            "Colleague highest-test LGBM",
            "lgbm_leaves_31",
            "Exact artifact; test-selected",
        ),
    ]
    for model, experiment_id, source_note in colleague_ids:
        row = colleague_log.loc[
            colleague_log["experiment_id"].eq(experiment_id)
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one row for {experiment_id}")
        item = row.iloc[0]
        fire_season_rows.append(
            {
                "model": model,
                "family": "colleague",
                "scope": "2025 Apr-Nov",
                "rows": 163_968,
                "positives": 1_624,
                "prevalence": 1_624 / 163_968,
                "pr_auc": float(item["test_pr_auc"]),
                "roc_auc": float(item["test_roc_auc"]),
                "source_note": source_note,
            }
        )
    fire_season_rows.append(
        {
            "model": "Colleague C default MLP",
            "family": "colleague",
            "scope": "2025 Apr-Nov",
            "rows": 163_968,
            "positives": 1_624,
            "prevalence": 1_624 / 163_968,
            "pr_auc": colleague_mlp["pr_auc"],
            "roc_auc": colleague_mlp["roc_auc"],
            "source_note": colleague_mlp["precision_note"],
        }
    )

    for model, frame in our_frames.items():
        subset = frame.loc[frame["label_date"].dt.month.between(4, 11)]
        metrics = score(subset)
        if metrics["rows"] != 163_968 or metrics["positives"] != 1_624:
            raise ValueError(f"{model} fire-season population does not match")
        display_name = f"{model} classifier"
        source_note = "Recomputed with colleague metric functions"
        if model == "Our V5":
            display_name = "Our V5 base classifier (inherits V4)"
            source_note += "; V5 reranker excluded from probability comparison"
        fire_season_rows.append(
            {
                "model": display_name,
                "family": "ours",
                "scope": "2025 Apr-Nov",
                **metrics,
                "source_note": source_note,
            }
        )

    best_colleague = colleague_mlp["pr_auc"]
    for row in fire_season_rows:
        row["pr_delta_vs_colleague_mlp"] = (
            float(row["pr_auc"]) - best_colleague
        )
        row["pr_ratio_vs_colleague_mlp"] = (
            float(row["pr_auc"]) / best_colleague
        )
    fire_table = pd.DataFrame(fire_season_rows)
    fire_table.to_csv(output / "fire_season_model_comparison.csv", index=False)

    bucket_rows = []
    v4 = our_frames["Our V4"]
    for bucket, months in BUCKET_MONTHS.items():
        ours = score(v4.loc[v4["label_date"].dt.month.isin(months)])
        theirs = colleague_eval["per_bucket"][bucket]
        if (
            ours["rows"] != int(theirs["n"])
            or ours["positives"] != int(theirs["n_pos"])
        ):
            raise ValueError(f"Population mismatch for bucket {bucket}")
        bucket_rows.append(
            {
                "bucket": bucket,
                "rows": ours["rows"],
                "positives": ours["positives"],
                "colleague_pr_auc": float(theirs["pr_auc"]),
                "our_v4_pr_auc": ours["pr_auc"],
                "pr_absolute_gain": ours["pr_auc"]
                - float(theirs["pr_auc"]),
                "pr_ratio": ours["pr_auc"] / float(theirs["pr_auc"]),
                "colleague_roc_auc": float(theirs["roc_auc"]),
                "our_v4_roc_auc": ours["roc_auc"],
                "roc_absolute_gain": ours["roc_auc"]
                - float(theirs["roc_auc"]),
            }
        )
    bucket_table = pd.DataFrame(bucket_rows)
    bucket_table.to_csv(output / "bucket_comparison.csv", index=False)

    monthly_rows = []
    for month in range(5, 12):
        ours = score(v4.loc[v4["label_date"].dt.month.eq(month)])
        theirs = colleague_eval["monthly_fire_season"][str(month)]
        if (
            ours["rows"] != int(theirs["n"])
            or ours["positives"] != int(theirs["n_pos"])
        ):
            raise ValueError(f"Population mismatch for month {month}")
        monthly_rows.append(
            {
                "month": month,
                "rows": ours["rows"],
                "positives": ours["positives"],
                "colleague_pr_auc": float(theirs["pr_auc"]),
                "our_v4_pr_auc": ours["pr_auc"],
                "pr_absolute_gain": ours["pr_auc"]
                - float(theirs["pr_auc"]),
                "pr_ratio": ours["pr_auc"] / float(theirs["pr_auc"]),
                "colleague_roc_auc": float(theirs["roc_auc"]),
                "our_v4_roc_auc": ours["roc_auc"],
                "roc_absolute_gain": ours["roc_auc"]
                - float(theirs["roc_auc"]),
            }
        )
    monthly_table = pd.DataFrame(monthly_rows)
    monthly_table.to_csv(output / "monthly_comparison.csv", index=False)

    our_v4_fire = fire_table.loc[
        fire_table["model"].eq("Our V4 classifier")
    ].iloc[0]
    our_v1_fire = fire_table.loc[
        fire_table["model"].eq("Our V1 classifier")
    ].iloc[0]
    comparison = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metric_contract": {
            "pr_auc": "sklearn.metrics.average_precision_score",
            "roc_auc": "sklearn.metrics.roc_auc_score",
            "test_year": 2025,
            "fire_season_months": list(range(4, 12)),
            "probability_column_for_our_models": "p_fire",
        },
        "headline": {
            "colleague_best_reported_model": "C_mlp_default",
            "colleague_pr_auc": colleague_mlp["pr_auc"],
            "colleague_roc_auc": colleague_mlp["roc_auc"],
            "our_model": "V4 classifier",
            "our_pr_auc": float(our_v4_fire["pr_auc"]),
            "our_roc_auc": float(our_v4_fire["roc_auc"]),
            "pr_absolute_gain": float(
                our_v4_fire["pr_auc"] - colleague_mlp["pr_auc"]
            ),
            "pr_ratio": float(
                our_v4_fire["pr_auc"] / colleague_mlp["pr_auc"]
            ),
            "pr_relative_gain_percent": float(
                (
                    our_v4_fire["pr_auc"] / colleague_mlp["pr_auc"]
                    - 1
                )
                * 100
            ),
            "roc_absolute_gain": float(
                our_v4_fire["roc_auc"] - colleague_mlp["roc_auc"]
            ),
        },
        "closest_no_fire_history_comparison": {
            "qualification": (
                "V1 is the closest existing test result without explicit "
                "FIRMS-history predictors; it is global rather than "
                "fire-season-specialized and therefore not architecture-identical."
            ),
            "colleague_model": "C_mlp_default",
            "colleague_pr_auc": colleague_mlp["pr_auc"],
            "our_model": "V1 classifier",
            "our_pr_auc": float(our_v1_fire["pr_auc"]),
            "colleague_absolute_gain": float(
                colleague_mlp["pr_auc"] - our_v1_fire["pr_auc"]
            ),
            "colleague_relative_gain_percent": float(
                (colleague_mlp["pr_auc"] / our_v1_fire["pr_auc"] - 1)
                * 100
            ),
        },
        "serving_contract": [
            {
                "source": "ERA5",
                "colleague_cutoff": "D-5",
                "our_v1_cutoff": "D-5",
                "our_v2_v5_cutoff": "D-5",
            },
            {
                "source": "S2/S5P observation window",
                "colleague_cutoff": "latest window_end <= D",
                "our_v1_cutoff": "latest window_end <= D in archive",
                "our_v2_v5_cutoff": (
                    "latest window_end <= D; stale S2 >15d and S5P >2d masked"
                ),
            },
            {
                "source": "FIRMS as predictor",
                "colleague_cutoff": "not used",
                "our_v1_cutoff": "not used",
                "our_v2_v5_cutoff": "history through D-1",
            },
            {
                "source": "FIRMS target",
                "colleague_cutoff": "D+1",
                "our_v1_cutoff": "D+1",
                "our_v2_v5_cutoff": "D+1",
            },
        ],
        "fire_season_models": fire_season_rows,
        "buckets": bucket_rows,
        "months": monthly_rows,
        "limitations": [
            "The colleague MLP metrics are rounded to four decimals in the report.",
            "The colleague report does not provide full-period daily top-K recall.",
            "V2-V5 and the colleague best-test comparisons reuse the 2025 benchmark.",
            "V4 fire histories require D-1 FIRMS availability at serving time.",
        ],
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Fair side-by-side model comparison",
        "",
        f"**Generated:** {comparison['generated_at_utc']}",
        "",
        "## Metric contract",
        "",
        "- Test period: 2025.",
        "- Fire season: April through November.",
        "- Identical population: 163,968 cell-days and 1,624 positives.",
        "- PR-AUC: `sklearn.metrics.average_precision_score`.",
        "- ROC-AUC: `sklearn.metrics.roc_auc_score`.",
        "- Our score column: calibrated classifier probability `p_fire`.",
        "",
        "## Fire-season model comparison",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Model", "PR-AUC", "ROC-AUC", "PR vs colleague MLP"],
            [
                [
                    row["model"],
                    f"{row['pr_auc']:.6f}",
                    f"{row['roc_auc']:.6f}",
                    f"{row['pr_ratio_vs_colleague_mlp']:.2f}×",
                ]
                for row in fire_season_rows
            ],
            ["---", "---:", "---:", "---:"],
        )
    )
    lines.extend(
        [
            "",
            "V4 improves PR-AUC over the colleague's best reported MLP by "
            f"**{comparison['headline']['pr_absolute_gain']:.4f} absolute**, "
            f"or **{comparison['headline']['pr_ratio']:.2f}×** "
            f"(**{comparison['headline']['pr_relative_gain_percent']:.1f}%** "
            "relative improvement). ROC-AUC "
            f"improves by **{comparison['headline']['roc_absolute_gain']:.4f}**.",
            "",
            "## Serving-time comparison",
            "",
            "Both pipelines enforce ERA5 through D−5. They are not identical "
            "input contracts because V2–V5 also use prior FIRMS observations.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Source",
                "Colleague",
                "Our V1",
                "Our V2–V5",
            ],
            [
                [
                    row["source"],
                    row["colleague_cutoff"],
                    row["our_v1_cutoff"],
                    row["our_v2_v5_cutoff"],
                ]
                for row in comparison["serving_contract"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "If D−1 FIRMS is available when forecasting on D, V4 is a "
            "real-time-capable, richer system and the 2.51× comparison is the "
            "appropriate best-system comparison.",
            "",
            "If explicit fire-history predictors are disallowed, V1 is our "
            "closest existing reference. The colleague MLP scores "
            f"{colleague_mlp['pr_auc']:.4f} versus V1 "
            f"{our_v1_fire['pr_auc']:.4f}: a "
            f"{comparison['closest_no_fire_history_comparison']['colleague_absolute_gain']:.4f} "
            "absolute or "
            f"{comparison['closest_no_fire_history_comparison']['colleague_relative_gain_percent']:.1f}% "
            "relative advantage for the colleague model.",
            "",
            "## Identical calendar buckets: colleague routed Stage-C LGBM vs our global V4",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Bucket",
                "Rows",
                "Positives",
                "Colleague PR",
                "Our V4 PR",
                "PR ratio",
                "Colleague ROC",
                "Our V4 ROC",
            ],
            [
                [
                    row["bucket"],
                    f"{row['rows']:,}",
                    f"{row['positives']:,}",
                    f"{row['colleague_pr_auc']:.6f}",
                    f"{row['our_v4_pr_auc']:.6f}",
                    f"{row['pr_ratio']:.2f}×",
                    f"{row['colleague_roc_auc']:.6f}",
                    f"{row['our_v4_roc_auc']:.6f}",
                ]
                for row in bucket_rows
            ],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )
    lines.extend(
        [
            "",
            "## Monthly fire-season comparison",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Month",
                "Positives",
                "Colleague PR",
                "Our V4 PR",
                "PR ratio",
                "Colleague ROC",
                "Our V4 ROC",
            ],
            [
                [
                    str(row["month"]),
                    f"{row['positives']:,}",
                    f"{row['colleague_pr_auc']:.6f}",
                    f"{row['our_v4_pr_auc']:.6f}",
                    f"{row['pr_ratio']:.2f}×",
                    f"{row['colleague_roc_auc']:.6f}",
                    f"{row['our_v4_roc_auc']:.6f}",
                ]
                for row in monthly_rows
            ],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The colleague MLP metrics are available only to four decimal places.",
            "- The colleague report does not provide full-period Recall@25, so "
            "daily alert-budget recall cannot be compared yet.",
            "- Both teams have inspected 2025 repeatedly; these are descriptive "
            "experiment comparisons, not a fresh production holdout.",
            "- V4's advantage depends on D−1 FIRMS history being available on "
            "decision day D.",
            "",
        ]
    )
    (output / "SIDE_BY_SIDE_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(output / "SIDE_BY_SIDE_COMPARISON.md")


if __name__ == "__main__":
    main()
