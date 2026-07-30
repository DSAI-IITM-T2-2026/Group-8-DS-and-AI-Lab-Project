from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/numerical_nextday_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay


def plot_calibration_and_pr(scored: pd.DataFrame, destination: Path, title: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    y = scored["y_fire"].to_numpy()
    p = scored["p_fire"].to_numpy()
    if len(np.unique(y)) == 2:
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[0].plot(predicted, observed, marker="o", label="model")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
        PrecisionRecallDisplay.from_predictions(y, p, ax=axes[1], name="model")
    axes[0].set(title="Reliability", xlabel="Mean predicted probability", ylabel="Observed rate")
    axes[0].legend()
    axes[1].set_title("Precision–recall")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_importance(importance: pd.DataFrame, destination: Path, top_n: int = 20) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(top_n).sort_values("mean_abs_contribution")
    figure, axis = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
    axis.barh(top["feature"], top["mean_abs_contribution"], color="#d95f02")
    axis.set(xlabel="Mean absolute TreeSHAP contribution", title="Global feature influence")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_risk_map(scored: pd.DataFrame, destination: Path, label_date: pd.Timestamp) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    day = scored.loc[pd.to_datetime(scored["label_date"]) == pd.Timestamp(label_date)].copy()
    if day.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 8))
    points = axis.scatter(
        day["longitude"],
        day["latitude"],
        c=day["p_fire"],
        cmap="YlOrRd",
        vmin=0,
        vmax=max(float(day["p_fire"].quantile(0.99)), 1e-5),
        s=26,
    )
    positives = day.loc[day["y_fire"] == 1]
    if len(positives):
        axis.scatter(
            positives["longitude"],
            positives["latitude"],
            marker="x",
            color="black",
            s=35,
            label="FIRMS positive",
        )
        axis.legend()
    axis.set(
        title=f"Next-day wildfire risk — {pd.Timestamp(label_date).date()}",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    figure.colorbar(points, ax=axis, label="Calibrated fire probability")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
