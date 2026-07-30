"""Evaluation metrics and figures for M4."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from numerical_nextday.data.s2_s5p import _lag_prefix
from numerical_nextday.train.router import filter_bucket

logger = logging.getLogger(__name__)


def _metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "n": int(len(y)), "n_pos": int(y.sum()) if len(y) else 0}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def run_eval_figures(cfg: dict) -> int:
    art = Path(cfg["paths"]["artifacts_dir"])
    fig_dir = art / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    root = _lag_prefix(cfg)
    test = pd.read_parquet(root / "stage_c" / "test.parquet")

    # Load fire_season model if present
    model_path = art / "models" / "fire_season" / "C_default.joblib"
    results = {"s5p_2021_mode": cfg.get("s5p_2021_mode"), "per_bucket": {}, "monthly_fire_season": {}}

    if not model_path.exists():
        logger.warning("No C_default model — writing empty eval stub")
        (art / "eval_metrics.json").write_text(json.dumps(results, indent=2))
        return 0

    bundle = joblib.load(model_path)
    model, iso, feat = bundle["model"], bundle["iso"], bundle["feature_cols"]

    def predict(df):
        raw = model.predict(df[feat], num_iteration=model.best_iteration)
        return iso.predict(raw)

    # Per bucket
    for bucket in ["fire_season", "jan", "feb", "mar", "dec"]:
        sub = filter_bucket(test, bucket, cfg)
        if len(sub) == 0:
            continue
        # Prefer bucket-specific weights
        bp = art / "models" / bucket / f"C_{bucket}.joblib"
        if bucket != "fire_season" and bp.exists():
            b = joblib.load(bp)
            proba = b["iso"].predict(b["model"].predict(sub[b["feature_cols"]], num_iteration=b["model"].best_iteration))
        elif bucket != "fire_season" and (art / "models" / bucket / "FALLBACK_fire_season.txt").exists():
            proba = predict(sub)
        else:
            proba = predict(sub)
        results["per_bucket"][bucket] = _metrics(sub["y_fire"], proba)

    # Monthly May–Nov on fire_season rows
    fs = filter_bucket(test, "fire_season", cfg)
    if len(fs):
        proba = predict(fs)
        fs = fs.copy()
        fs["proba"] = proba
        for m in range(5, 12):
            sub = fs.loc[pd.to_datetime(fs["label_date"]).dt.month == m]
            results["monthly_fire_season"][str(m)] = _metrics(sub["y_fire"], sub["proba"])

        # Calibration curve
        from sklearn.calibration import calibration_curve

        try:
            frac_pos, mean_pred = calibration_curve(fs["y_fire"], fs["proba"], n_bins=10)
            plt.figure(figsize=(5, 5))
            plt.plot(mean_pred, frac_pos, marker="o", label="fire_season")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("Predicted")
            plt.ylabel("Fraction positive")
            plt.title("Calibration — Stage C fire_season test 2025")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "calibration_fire_season.png", dpi=120)
            plt.close()
        except Exception as e:
            logger.warning("Calibration plot failed: %s", e)

        # Sample risk map for one high-fire day
        day_counts = fs.groupby(fs["label_date"].astype(str))["y_fire"].sum().sort_values(ascending=False)
        if len(day_counts):
            day = day_counts.index[0]
            day_df = fs.loc[fs["label_date"].astype(str) == day]
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(
                day_df["longitude"],
                day_df["latitude"],
                c=day_df["proba"],
                s=12,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
            )
            plt.colorbar(sc, label="P(fire)")
            plt.title(f"Risk map {day}")
            plt.xlabel("lon")
            plt.ylabel("lat")
            plt.tight_layout()
            plt.savefig(fig_dir / "risk_map_sample.png", dpi=120)
            plt.close()

            top = day_df.nlargest(25, "proba")[
                ["cell_id", "latitude", "longitude", "proba", "y_fire", "region"]
            ]
            top.to_csv(art / "sample_topk_alerts.csv", index=False)

    (art / "eval_metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote eval → %s", art / "eval_metrics.json")
    return 0
