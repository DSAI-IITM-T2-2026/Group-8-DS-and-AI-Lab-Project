from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    if len(y_true) == 0:
        return math.nan
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            result += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(result)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    metrics = {
        "n_rows": len(y),
        "n_positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else math.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else math.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else math.nan,
        "ece_10bin": expected_calibration_error(y, p),
    }
    if len(np.unique(y)) == 2:
        metrics["pr_auc"] = float(average_precision_score(y, p))
        metrics["roc_auc"] = float(roc_auc_score(y, p))
        precision, recall, _ = precision_recall_curve(y, p)
        metrics["max_f1"] = float(
            np.nanmax(2 * precision * recall / np.maximum(precision + recall, 1e-12))
        )
    else:
        metrics.update({"pr_auc": math.nan, "roc_auc": math.nan, "max_f1": math.nan})
    return metrics


def top_k_metrics(
    scored: pd.DataFrame,
    top_k: int,
    date_column: str = "label_date",
    target: str = "y_fire",
    probability: str = "p_fire",
) -> dict[str, float]:
    if scored.empty:
        return {
            "top_k_per_day": top_k,
            "alert_count": 0,
            "alert_precision": math.nan,
            "positive_recall": math.nan,
            "false_alerts_per_day": math.nan,
        }
    alerts = (
        scored.sort_values([date_column, probability], ascending=[True, False])
        .groupby(date_column, group_keys=False)
        .head(top_k)
    )
    captured = int(alerts[target].sum())
    all_positives = int(scored[target].sum())
    days = max(int(scored[date_column].nunique()), 1)
    return {
        "top_k_per_day": int(top_k),
        "alert_count": len(alerts),
        "alert_precision": float(captured / len(alerts)) if len(alerts) else math.nan,
        "positive_recall": float(captured / all_positives) if all_positives else math.nan,
        "false_alerts_per_day": float((len(alerts) - captured) / days),
    }


def complete_metrics(scored: pd.DataFrame, top_k: int) -> dict[str, float]:
    return {
        **binary_metrics(scored["y_fire"].to_numpy(), scored["p_fire"].to_numpy()),
        **top_k_metrics(scored, top_k),
    }


def metrics_by_group(
    scored: pd.DataFrame, group_column: str, top_k: int
) -> list[dict[str, float | str | int]]:
    records = []
    for value, group in scored.groupby(group_column, dropna=False):
        records.append({group_column: value, **complete_metrics(group, top_k)})
    return records
