"""Evaluation metrics and figures for M4."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from numerical_nextday.data.s2_s5p import _lag_prefix
from numerical_nextday.eval.maps import load_boundary, plot_california_risk_day
from numerical_nextday.eval.metrics_charts import write_metrics_charts
from numerical_nextday.eval.scoring import score_test_frame
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

    model_path = art / "models" / "fire_season" / "C_default.joblib"
    results = {"s5p_2021_mode": cfg.get("s5p_2021_mode"), "per_bucket": {}, "monthly_fire_season": {}}

    if not model_path.exists():
        logger.warning("No C_default model — writing empty eval stub")
        (art / "eval_metrics.json").write_text(json.dumps(results, indent=2))
        return 0

    # Score full test with router (for maps + consistent metrics)
    scored = score_test_frame(cfg, test=test)

    for bucket in ["fire_season", "jan", "feb", "mar", "dec"]:
        sub = filter_bucket(scored, bucket, cfg)
        if len(sub) == 0:
            continue
        results["per_bucket"][bucket] = _metrics(sub["y_fire"], sub["proba"])

    fs = filter_bucket(scored, "fire_season", cfg).copy()
    if len(fs):
        for m in range(5, 12):
            sub = fs.loc[pd.to_datetime(fs["label_date"]).dt.month == m]
            results["monthly_fire_season"][str(m)] = _metrics(sub["y_fire"], sub["proba"])

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

        # California-outline sample maps (M3 teammate style)
        geo = Path(cfg["paths"].get("california_geojson", Path(__file__).resolve().parents[3] / "data" / "california.geojson"))
        if not geo.exists():
            geo = Path(__file__).resolve().parents[3] / "data" / "california.geojson"
        boundary = load_boundary(geo)

        fs["day"] = pd.to_datetime(fs["label_date"]).dt.strftime("%Y-%m-%d")
        day_stats = (
            fs.groupby("day")
            .agg(n_pos=("y_fire", "sum"), mean_p=("proba", "mean"), max_p=("proba", "max"))
            .sort_values(["n_pos", "max_p"], ascending=False)
        )

        day_df = None
        if len(day_stats):
            top_fire_day = str(day_stats.index[0])
            day_df = fs.loc[fs["day"] == top_fire_day]
            plot_california_risk_day(
                day_df,
                boundary,
                fig_dir / "risk_map_sample.png",
                title=f"California wildfire risk — {top_fire_day}",
            )

            mid = day_stats.loc[[d for d in day_stats.index if d[5:7] in ("07", "08", "09", "10")]]
            if len(mid):
                mid_day = str(mid.sort_values(["max_p", "n_pos"], ascending=False).index[0])
                mid_df = fs.loc[fs["day"] == mid_day]
                plot_california_risk_day(
                    mid_df,
                    boundary,
                    fig_dir / "risk_map_high_activity.png",
                    title=f"California wildfire risk — {mid_day}",
                )

                # predicted vs actual panel (keep as secondary view)
                vmax = max(0.2, float(mid_df["proba"].quantile(0.995)))
                fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
                sc0 = axes[0].scatter(
                    mid_df["longitude"], mid_df["latitude"], c=mid_df["proba"], s=16, cmap="YlOrRd", vmin=0, vmax=vmax
                )
                fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, label="P(fire)")
                axes[0].set_title(f"Predicted P(fire)\n{mid_day}")
                axes[1].scatter(mid_df["longitude"], mid_df["latitude"], c="#f5f0e6", s=12)
                fires = mid_df.loc[mid_df["y_fire"] == 1]
                axes[1].scatter(
                    fires["longitude"], fires["latitude"], s=45, c="#b71c1c", edgecolors="black", linewidths=0.35
                )
                axes[1].set_title(f"Actual FIRMS fires (n={len(fires)})\n{mid_day}")
                for ax in axes:
                    ax.set_xlabel("lon")
                    ax.set_ylabel("lat")
                    ax.set_aspect("equal", adjustable="box")
                fig.suptitle("Predicted risk vs observed fires — Stage C", y=1.02)
                fig.tight_layout()
                fig.savefig(fig_dir / "risk_vs_actual_sample.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            if day_df is not None and len(day_df):
                cols = ["cell_id", "latitude", "longitude", "proba", "y_fire"]
                if "region" in day_df.columns:
                    cols.append("region")
                day_df.nlargest(25, "proba")[cols].to_csv(art / "sample_topk_alerts.csv", index=False)

    (art / "eval_metrics.json").write_text(json.dumps(results, indent=2))
    try:
        write_metrics_charts(cfg)
    except Exception as e:
        logger.warning("Metric charts failed: %s", e)

    logger.info("Wrote eval → %s", art / "eval_metrics.json")
    return 0
