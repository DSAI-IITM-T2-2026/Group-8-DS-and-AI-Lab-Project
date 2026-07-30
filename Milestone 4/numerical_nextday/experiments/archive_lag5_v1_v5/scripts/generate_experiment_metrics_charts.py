#!/usr/bin/env python3
"""Regenerate V1-V5 and fair-comparison PR-AUC/ROC-AUC charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "archive_training"
PR_COLOR = "#c45c26"
ROC_COLOR = "#4a6fa5"


def _label_bars(axis, bars, digits: int = 3) -> None:
    axis.bar_label(bars, fmt=f"%.{digits}f", fontsize=7, padding=2)


def _save(figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(destination)
    return destination


def plot_version_ladder(results: pd.DataFrame, out_dir: Path) -> Path:
    x = np.arange(len(results))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9, 5))
    pr_bars = axis.bar(
        x - width / 2,
        results["test_calibrated_pr_auc"],
        width,
        label="Test PR-AUC",
        color=PR_COLOR,
    )
    roc_bars = axis.bar(
        x + width / 2,
        results["test_roc_auc"],
        width,
        label="Test ROC-AUC",
        color=ROC_COLOR,
    )
    _label_bars(axis, pr_bars)
    _label_bars(axis, roc_bars)
    axis.set_xticks(x, results["version"])
    axis.set(ylabel="Score", ylim=(0, 1.0))
    axis.set_title("Lag-5 model version ladder — test 2025")
    axis.legend()
    return _save(figure, out_dir / "metrics_stage_ladder.png")


def plot_recall_ladder(results: pd.DataFrame, out_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        results["version"],
        results["test_recall_25"],
        marker="o",
        linewidth=2,
        label="Recall@25/day",
        color=PR_COLOR,
    )
    axis.plot(
        results["version"],
        results["test_recall_50"],
        marker="s",
        linewidth=2,
        label="Recall@50/day",
        color=ROC_COLOR,
    )
    axis.axhline(
        0.5,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="50% recall target",
    )
    for _, row in results.iterrows():
        axis.annotate(
            f'{row["test_recall_25"]:.3f}',
            (row["version"], row["test_recall_25"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set(ylabel="Recall", ylim=(0, 0.65))
    axis.set_title("Daily alert-budget recall — test 2025")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    return _save(figure, out_dir / "metrics_recall_at_k.png")


def plot_fair_model_comparison(comparison: pd.DataFrame, out_dir: Path) -> Path:
    selected = comparison.loc[
        comparison["model"].isin(
            [
                "Colleague C default LGBM",
                "Colleague C default MLP",
                "Our V1 classifier",
                "Our V2 classifier",
                "Our V3 classifier",
                "Our V4 classifier",
            ]
        )
    ].copy()
    x = np.arange(len(selected))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    pr_bars = axis.bar(
        x - width / 2,
        selected["pr_auc"],
        width,
        label="PR-AUC",
        color=PR_COLOR,
    )
    roc_bars = axis.bar(
        x + width / 2,
        selected["roc_auc"],
        width,
        label="ROC-AUC",
        color=ROC_COLOR,
    )
    _label_bars(axis, pr_bars)
    _label_bars(axis, roc_bars)
    labels = (
        selected["model"]
        .str.replace("Colleague C default ", "Colleague ", regex=False)
        .str.replace(" classifier", "", regex=False)
    )
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set(ylabel="Score", ylim=(0, 1.0))
    axis.set_title(
        "Fair model comparison — same Apr–Nov 2025 population"
    )
    axis.legend()
    return _save(figure, out_dir / "metrics_model_comparison.png")


def plot_bucket_comparison(buckets: pd.DataFrame, out_dir: Path) -> Path:
    x = np.arange(len(buckets))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, metric, title in [
        (axes[0], "pr_auc", "PR-AUC"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        left = axis.bar(
            x - width / 2,
            buckets[f"colleague_{metric}"],
            width,
            label="Colleague routed C",
            color="#8aa6c8",
        )
        right = axis.bar(
            x + width / 2,
            buckets[f"our_v4_{metric}"],
            width,
            label="Our global V4",
            color=PR_COLOR,
        )
        _label_bars(axis, left)
        _label_bars(axis, right)
        axis.set_xticks(x, buckets["bucket"], rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.2)
    axes[0].set(ylabel="Test score", ylim=(0, 1.0))
    axes[1].legend(loc="upper right")
    figure.suptitle("Identical 2025 calendar buckets")
    return _save(figure, out_dir / "metrics_month_buckets.png")


def plot_monthly_comparison(monthly: pd.DataFrame, out_dir: Path) -> Path:
    labels = [
        pd.Timestamp(2025, int(month), 1).strftime("%b")
        for month in monthly["month"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, metric, title in [
        (axes[0], "pr_auc", "PR-AUC"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        axis.plot(
            labels,
            monthly[f"colleague_{metric}"],
            marker="s",
            linewidth=2,
            label="Colleague routed C",
            color="#8aa6c8",
        )
        axis.plot(
            labels,
            monthly[f"our_v4_{metric}"],
            marker="o",
            linewidth=2,
            label="Our global V4",
            color=PR_COLOR,
        )
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set(ylabel="Test score", ylim=(0, 1.0))
    axes[1].legend(loc="lower right")
    figure.suptitle("Within fire-season months — test 2025")
    return _save(figure, out_dir / "metrics_monthly_fire_season.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=None,
        help=(
            "Directory containing fire_season_model_comparison.csv, "
            "bucket_comparison.csv and monthly_comparison.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    comparison_root = (
        args.comparison_root.resolve()
        if args.comparison_root
        else artifact_root / "colleague_side_by_side"
    )
    if not comparison_root.exists():
        tracked_comparison = (
            PROJECT_ROOT / "reports" / "colleague_side_by_side"
        )
        if tracked_comparison.exists():
            comparison_root = tracked_comparison
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else artifact_root / "figures"
    )

    results = pd.read_csv(artifact_root / "EXPERIMENT_RESULTS.csv")
    comparison = pd.read_csv(
        comparison_root / "fire_season_model_comparison.csv"
    )
    buckets = pd.read_csv(comparison_root / "bucket_comparison.csv")
    monthly = pd.read_csv(comparison_root / "monthly_comparison.csv")

    plot_version_ladder(results, out_dir)
    plot_recall_ladder(results, out_dir)
    plot_fair_model_comparison(comparison, out_dir)
    plot_bucket_comparison(buckets, out_dir)
    plot_monthly_comparison(monthly, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
