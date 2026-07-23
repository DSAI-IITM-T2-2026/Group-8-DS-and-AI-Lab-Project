"""Segmentation metrics: precision, recall, F1, Dice, AUC-PR, confusion matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def dice_score(pred_bin: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred_bin * target).sum())
    return float((2 * inter + eps) / (pred_bin.sum() + target.sum() + eps))


def confusion_counts(pred: np.ndarray, targets: np.ndarray) -> dict:
    tp = int(((pred == 1) & (targets == 1)).sum())
    fp = int(((pred == 1) & (targets == 0)).sum())
    fn = int(((pred == 0) & (targets == 1)).sum())
    tn = int(((pred == 0) & (targets == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(logits, targets, threshold: float = 0.5) -> dict:
    logits = _to_numpy(logits).reshape(-1)
    targets = _to_numpy(targets).reshape(-1).astype(np.float32)
    valid = np.isfinite(logits) & np.isfinite(targets)
    logits, targets = logits[valid], targets[valid]
    probs = 1 / (1 + np.exp(-logits))
    pred = (probs >= threshold).astype(np.float32)

    cm = confusion_counts(pred, targets)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    try:
        auc_pr = float(average_precision_score(targets, probs)) if targets.sum() > 0 else float("nan")
    except ValueError:
        auc_pr = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": dice_score(pred, targets),
        "auc_pr": auc_pr,
        "accuracy": float(acc),
        "accuracy_caveat": "secondary only — uninformative under imbalance",
        "confusion_matrix": cm,
        "positive_rate_true": float(targets.mean()) if targets.size else 0.0,
        "positive_rate_pred": float(pred.mean()) if pred.size else 0.0,
    }


def metrics_from_probs(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Same metrics from probabilities (tree baseline)."""
    y_true = _to_numpy(y_true).reshape(-1).astype(np.float32)
    y_prob = _to_numpy(y_prob).reshape(-1).astype(np.float32)
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[valid], y_prob[valid]
    pred = (y_prob >= threshold).astype(np.float32)
    # fake logits via logit for reuse — compute directly
    cm = confusion_counts(pred, y_true)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    try:
        auc_pr = float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else float("nan")
    except ValueError:
        auc_pr = float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": dice_score(pred, y_true),
        "auc_pr": auc_pr,
        "accuracy": float(acc),
        "accuracy_caveat": "uninformative under severe class imbalance — not a headline metric",
        "confusion_matrix": cm,
        "positive_rate_true": float(y_true.mean()) if y_true.size else 0.0,
        "positive_rate_pred": float(pred.mean()) if y_true.size else 0.0,
    }


def classification_report_dict(y_true, y_pred_or_probs, from_probs: bool = True, threshold: float = 0.5) -> dict:
    y_true = _to_numpy(y_true).reshape(-1).astype(np.int32)
    arr = _to_numpy(y_pred_or_probs).reshape(-1)
    if from_probs:
        y_pred = (arr >= threshold).astype(np.int32)
    else:
        y_pred = arr.astype(np.int32)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["no_fire", "fire"],
        output_dict=True,
        zero_division=0,
    )


def save_confusion_artifacts(
    cm: dict,
    out_stem: Path,
    title: str = "Confusion matrix",
) -> None:
    """Write cm JSON + PNG heatmap. out_stem without extension."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stem.with_suffix(".json"), "w") as f:
        json.dump(cm, f, indent=2)

    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred no-fire", "Pred fire"])
    ax.set_yticks([0, 1], ["True no-fire", "True fire"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    plt.close(fig)


def write_eval_bundle(
    *,
    name: str,
    split: str,
    metrics: dict,
    y_true=None,
    y_prob=None,
    logits=None,
    targets=None,
    figures_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Persist metrics + confusion matrix + classification report for one model/split.
    Provide either (y_true, y_prob) or (logits, targets).
    """
    from src import config

    figures_dir = Path(figures_dir or config.FIGURES_DIR)
    metadata_dir = Path(metadata_dir or config.METADATA_DIR)
    cm_dir = figures_dir / "confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if metrics.get("confusion_matrix") is None:
        if logits is not None and targets is not None:
            metrics = compute_metrics(logits, targets, threshold=threshold)
        elif y_true is not None and y_prob is not None:
            metrics = metrics_from_probs(y_true, y_prob, threshold=threshold)

    cm = metrics["confusion_matrix"]
    stem = cm_dir / f"cm_{name}_{split}"
    save_confusion_artifacts(cm, stem, title=f"{name} — {split}")

    if y_true is not None and y_prob is not None:
        report = classification_report_dict(y_true, y_prob, from_probs=True, threshold=threshold)
    elif logits is not None and targets is not None:
        logits_np = _to_numpy(logits).reshape(-1)
        targets_np = _to_numpy(targets).reshape(-1)
        probs = 1 / (1 + np.exp(-logits_np))
        report = classification_report_dict(targets_np, probs, from_probs=True, threshold=threshold)
    else:
        report = {}

    bundle = {
        "name": name,
        "split": split,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix_png": str(stem.with_suffix(".png")),
        "confusion_matrix_json": str(stem.with_suffix(".json")),
    }
    out_json = metadata_dir / f"eval_{name}_{split}.json"
    with open(out_json, "w") as f:
        json.dump(bundle, f, indent=2)
    return bundle


def pr_curve(logits, targets):
    logits = _to_numpy(logits).reshape(-1)
    targets = _to_numpy(targets).reshape(-1)
    valid = np.isfinite(logits) & np.isfinite(targets)
    probs = 1 / (1 + np.exp(-logits[valid]))
    targets = targets[valid]
    if targets.sum() == 0:
        return None, None, None
    p, r, thr = precision_recall_curve(targets, probs)
    return p, r, thr
