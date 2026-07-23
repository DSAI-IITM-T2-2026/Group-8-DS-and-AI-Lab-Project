"""Tree baseline (HistGradientBoosting; XGBoost if stable) + always-no-fire floor."""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src import config


def flatten_last_day(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Use last day of the 7-day window only (judgment call).
    X: (N, T, H, W, C) → (N*H*W, C); y → (N*H*W,)
    """
    last = X[:, -1]  # (N, H, W, C)
    n, h, w, c = last.shape
    feats = last.reshape(-1, c)
    labels = y.reshape(-1)
    valid = np.isfinite(feats).all(axis=1) & np.isfinite(labels)
    return feats[valid], labels[valid].astype(np.int32)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    from src.training.metrics import metrics_from_probs

    return metrics_from_probs(y_true, y_prob, threshold=threshold)


def always_no_fire_metrics(y_true: np.ndarray) -> dict:
    y_prob = np.zeros_like(y_true, dtype=np.float32)
    m = evaluate_binary(y_true, y_prob)
    m["model"] = "always_no_fire"
    return m


def _make_classifier(seed: int):
    """
    Prefer sklearn HGB on macOS — XGBoost+OpenMP often segfaults (exit 139) on
    Apple Silicon / Python 3.14. Try XGBoost elsewhere when import succeeds.
    """
    if sys.platform == "darwin":
        print("  Using HistGradientBoostingClassifier (macOS-safe tree baseline)")
        return (
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.1,
                max_iter=100,
                random_state=seed,
            ),
            "hist_gradient_boosting",
        )
    try:
        from xgboost import XGBClassifier

        print("  Using XGBClassifier")
        return (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_estimators=100,
                max_depth=6,
                learning_rate=0.3,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=seed,
                n_jobs=1,
            ),
            "xgboost_default",
        )
    except Exception as exc:
        print(f"  XGBoost unavailable ({exc}); using HistGradientBoostingClassifier")
        return (
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.1,
                max_iter=100,
                random_state=seed,
            ),
            "hist_gradient_boosting",
        )


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_train_pixels: Optional[int] = 2_000_000,
    seed: int = config.SEED,
) -> dict:
    """Tree baseline on last-day per-pixel features (default hyperparameters)."""
    xt, yt = flatten_last_day(X_train, y_train)
    xv, yv = flatten_last_day(X_val, y_val)
    xs, ys = flatten_last_day(X_test, y_test)

    rng = np.random.default_rng(seed)
    if max_train_pixels is not None and len(xt) > max_train_pixels:
        pos = np.where(yt == 1)[0]
        neg = np.where(yt == 0)[0]
        n_neg = max_train_pixels - len(pos)
        if n_neg > 0 and len(neg) > n_neg:
            neg = rng.choice(neg, size=n_neg, replace=False)
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        xt, yt = xt[idx], yt[idx]

    clf, model_name = _make_classifier(seed)
    clf.fit(xt, yt)

    results = {
        "train": evaluate_binary(yt, clf.predict_proba(xt)[:, 1]),
        "val": evaluate_binary(yv, clf.predict_proba(xv)[:, 1]),
        "test": evaluate_binary(ys, clf.predict_proba(xs)[:, 1]),
        "floor_test": always_no_fire_metrics(ys),
        "n_train_pixels": int(len(xt)),
        "n_test_pixels": int(len(xs)),
        "model": model_name,
    }
    return {"model": clf, "metrics": results}
