"""California outline risk maps (Milestone 3 teammate style)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)


def load_boundary(path: Path):
    """Load CA boundary as GeoDataFrame or list of lon/lat rings."""
    try:
        import geopandas as gpd

        return gpd.read_file(path)
    except Exception:
        data = json.loads(path.read_text())
        polys: list[np.ndarray] = []

        def _coords(geom):
            t = geom["type"]
            c = geom["coordinates"]
            if t == "Polygon":
                return [c[0]]
            if t == "MultiPolygon":
                return [p[0] for p in c]
            return []

        feats = data["features"] if data.get("type") == "FeatureCollection" else [{"geometry": data}]
        for feat in feats:
            for ring in _coords(feat["geometry"]):
                polys.append(np.asarray(ring))
        return polys


def plot_california_risk_day(
    day_df: pd.DataFrame,
    boundary,
    out_path: Path,
    title: str | None = None,
    proba_col: str = "proba",
    dpi: int = 160,
) -> Path:
    """
    Teammate-style statewide map:
    CA outline, YlOrRd confidence %, blue rings for FIRMS positives.
    """
    df = day_df.copy()
    if "confidence_pct" not in df.columns:
        df["confidence_pct"] = df[proba_col].astype(float) * 100.0

    stamp = None
    if "label_date" in df.columns and len(df):
        stamp = pd.Timestamp(df["label_date"].iloc[0]).strftime("%Y-%m-%d")
    title = title or (f"California wildfire risk — {stamp}" if stamp else "California wildfire risk")

    fig, ax = plt.subplots(figsize=(8, 9))

    if hasattr(boundary, "plot"):
        boundary.plot(ax=ax, facecolor="#f5f5f0", edgecolor="#333333", linewidth=0.8)
    else:
        for ring in boundary:
            ax.plot(ring[:, 0], ring[:, 1], color="#333333", linewidth=0.8)
            ax.fill(ring[:, 0], ring[:, 1], color="#f5f5f0", alpha=0.5)

    # Match teammate UI: Confidence % colorbar typically spans ~0–50
    vmax = max(50.0, float(df["confidence_pct"].max()) if len(df) else 50.0)
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["confidence_pct"],
        s=36,
        cmap="YlOrRd",
        norm=Normalize(vmin=0, vmax=vmax),
        edgecolors="k",
        linewidths=0.2,
        alpha=0.9,
        zorder=3,
    )
    if "y_fire" in df.columns:
        pos = df.loc[df["y_fire"] == 1]
        if len(pos):
            ax.scatter(
                pos["longitude"],
                pos["latitude"],
                facecolors="none",
                edgecolors="#0033aa",
                s=80,
                linewidths=1.2,
                label="FIRMS positive",
                zorder=4,
            )
            ax.legend(loc="lower left", frameon=True, fontsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Confidence %")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s (%d cells)", out_path, len(df))
    return out_path
