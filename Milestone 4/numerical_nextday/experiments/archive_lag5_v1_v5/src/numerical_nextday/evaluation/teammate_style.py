"""Teammate-compatible California wildfire risk-map rendering."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def load_california_boundary(path: Path) -> list[np.ndarray]:
    """Load Polygon/MultiPolygon exterior rings from a GeoJSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = (
        data["features"]
        if data.get("type") == "FeatureCollection"
        else [{"geometry": data}]
    )
    rings: list[np.ndarray] = []
    for feature in features:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        rings.extend(np.asarray(polygon[0]) for polygon in polygons)
    if not rings:
        raise ValueError(f"No Polygon or MultiPolygon geometry in {path}")
    return rings


def plot_california_risk_day(
    day: pd.DataFrame,
    boundary: list[np.ndarray],
    destination: Path,
    *,
    title: str,
    confidence_cap: float = 50.0,
    dpi: int = 160,
) -> Path:
    """Render the established CA-outline, confidence-dot, FIRMS-ring style."""
    frame = day.copy()
    if "confidence_pct" not in frame.columns:
        if "p_fire" not in frame.columns:
            raise ValueError("Predictions need confidence_pct or p_fire")
        frame["confidence_pct"] = frame["p_fire"].astype(float) * 100.0

    figure, axis = plt.subplots(figsize=(8, 9))
    for ring in boundary:
        axis.fill(
            ring[:, 0],
            ring[:, 1],
            facecolor="#f5f5f0",
            edgecolor="#333333",
            linewidth=0.8,
            zorder=1,
        )

    # A fixed scale makes colors comparable from one day/model to another.
    color_values = frame["confidence_pct"].astype(float).clip(
        lower=0.0, upper=confidence_cap
    )
    cells = axis.scatter(
        frame["longitude"],
        frame["latitude"],
        c=color_values,
        s=36,
        cmap="YlOrRd",
        norm=Normalize(vmin=0.0, vmax=confidence_cap),
        edgecolors="black",
        linewidths=0.2,
        alpha=0.9,
        zorder=3,
    )

    positives = frame.loc[frame["y_fire"].astype(int).eq(1)]
    if not positives.empty:
        axis.scatter(
            positives["longitude"],
            positives["latitude"],
            facecolors="none",
            edgecolors="#0033aa",
            s=80,
            linewidths=1.2,
            label="FIRMS positive",
            zorder=4,
        )
        axis.legend(loc="lower left", frameon=True, fontsize=8)

    colorbar = figure.colorbar(cells, ax=axis, shrink=0.7, pad=0.02)
    colorbar.set_label("Confidence %")
    axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title=title,
    )
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination
