"""Metric comparison charts for Milestone 4 artifacts."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _read_experiments(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_metrics_charts(cfg: dict) -> list[Path]:
    """Write bar/line charts for ROC/PR and related results into artifacts/figures/."""
    art = Path(cfg["paths"]["artifacts_dir"])
    fig_dir = art / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- 1) Stage A/B/C + best LGBM/MLP ladder ---
    rows = _read_experiments(art / "experiments_log.csv")
    ladder_ids = ["A_default", "B_default", "C_default", "lgbm_leaves_31", "lgbm_era5_lag0"]
    by_id = {r["experiment_id"]: r for r in rows}
    labels, prs, rocs = [], [], []
    for eid in ladder_ids:
        if eid not in by_id:
            continue
        r = by_id[eid]
        labels.append(eid.replace("_default", "").replace("lgbm_", ""))
        prs.append(float(r["test_pr_auc"]))
        rocs.append(float(r["test_roc_auc"]))

    # MLP from metrics json if present
    mlp_path = art / "models" / "fire_season" / "C_mlp_default_metrics.json"
    if mlp_path.exists():
        m = json.loads(mlp_path.read_text())
        labels.append("C_mlp")
        prs.append(float(m["test"]["pr_auc"]))
        rocs.append(float(m["test"]["roc_auc"]))

    if labels:
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 5))
        w = 0.38
        ax.bar(x - w / 2, prs, w, label="Test PR-AUC", color="#c45c26")
        ax.bar(x + w / 2, rocs, w, label="Test ROC-AUC", color="#4a6fa5")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_title("Stage ladder & key models — test 2025 (fire_season)")
        ax.legend()
        ax.axhline(0.01, color="gray", ls="--", lw=0.8, label="~chance PR")
        fig.tight_layout()
        p = fig_dir / "metrics_stage_ladder.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # --- 2) Full LGBM HP sweep ---
    hp = [r for r in rows if r.get("bucket") == "fire_season" and r.get("stage") == "C"]
    if hp:
        hp = sorted(hp, key=lambda r: float(r["test_pr_auc"]), reverse=True)
        names = [r["experiment_id"] for r in hp]
        pr = [float(r["test_pr_auc"]) for r in hp]
        roc = [float(r["test_roc_auc"]) for r in hp]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
        ax.barh(y - 0.2, pr, 0.4, label="PR-AUC", color="#c45c26")
        ax.barh(y + 0.2, roc, 0.4, label="ROC-AUC", color="#4a6fa5")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Test score")
        ax.set_xlim(0, 1.0)
        ax.set_title("LightGBM Stage C hyperparameter trials — test 2025")
        ax.legend(loc="lower right")
        fig.tight_layout()
        p = fig_dir / "metrics_lgbm_hp_sweep.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # --- 3) Month buckets ---
    eval_path = art / "eval_metrics.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        pb = ev.get("per_bucket") or {}
        if pb:
            order = ["fire_season", "jan", "feb", "mar", "dec"]
            names = [k for k in order if k in pb]
            pr = [pb[k]["pr_auc"] for k in names]
            roc = [pb[k]["roc_auc"] for k in names]
            x = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(x - 0.2, pr, 0.4, label="PR-AUC", color="#c45c26")
            ax.bar(x + 0.2, roc, 0.4, label="ROC-AUC", color="#4a6fa5")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title("Month-bucket models — test 2025")
            ax.legend()
            fig.tight_layout()
            p = fig_dir / "metrics_month_buckets.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

        monthly = ev.get("monthly_fire_season") or {}
        if monthly:
            months = sorted(monthly.keys(), key=lambda m: int(m))
            labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            names = [labels[int(m) - 1] for m in months]
            pr = [monthly[m]["pr_auc"] for m in months]
            roc = [monthly[m]["roc_auc"] for m in months]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(names, pr, marker="o", label="PR-AUC", color="#c45c26")
            ax.plot(names, roc, marker="s", label="ROC-AUC", color="#4a6fa5")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title("Within fire-season months — Stage C test 2025")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = fig_dir / "metrics_monthly_fire_season.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

    # --- 4) MLP trials if metrics exist ---
    mlp_dir = art / "models" / "fire_season"
    mlp_rows = []
    for pth in sorted(mlp_dir.glob("mlp*_metrics.json")) + sorted(mlp_dir.glob("C_mlp*_metrics.json")):
        d = json.loads(pth.read_text())
        eid = d.get("experiment_id", pth.stem.replace("_metrics", ""))
        mlp_rows.append((eid, float(d["test"]["pr_auc"]), float(d["test"]["roc_auc"])))
    if mlp_rows:
        mlp_rows = sorted(mlp_rows, key=lambda t: t[1], reverse=True)
        names = [r[0] for r in mlp_rows]
        pr = [r[1] for r in mlp_rows]
        roc = [r[2] for r in mlp_rows]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(names))))
        ax.barh(y - 0.2, pr, 0.4, label="PR-AUC", color="#c45c26")
        ax.barh(y + 0.2, roc, 0.4, label="ROC-AUC", color="#4a6fa5")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Test score")
        ax.set_title("MLP Stage C trials — test 2025")
        ax.legend(loc="lower right")
        fig.tight_layout()
        p = fig_dir / "metrics_mlp_sweep.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    logger.info("Wrote %d metric charts → %s", len(written), fig_dir)
    return written
