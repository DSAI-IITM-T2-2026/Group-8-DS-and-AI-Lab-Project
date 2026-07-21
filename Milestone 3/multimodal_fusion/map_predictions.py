#!/usr/bin/env python3
"""Map cell-level confidence predictions on the California outline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("map_predictions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--predictions",
        default=None,
        help="Parquet/CSV with lat/lon/confidence_pct (default: model/test_predictions.parquet)",
    )
    p.add_argument(
        "--date",
        default=None,
        help="label_date YYYY-MM-DD (default: day with most positives, else max mean risk)",
    )
    p.add_argument("--all-dates", action="store_true", help="Write a map for every test date")
    p.add_argument("--topk-source", action="store_true", help="Use test_alerts_topk.csv instead")
    return p.parse_args()


def _load_boundary(path: Path):
    try:
        import geopandas as gpd

        return gpd.read_file(path)
    except Exception:
        import json
        from matplotlib.patches import Polygon

        data = json.loads(path.read_text())
        polys = []

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


def _pick_date(df: pd.DataFrame, requested: str | None) -> pd.Timestamp:
    if requested:
        return pd.Timestamp(requested).normalize()
    if "y_fire" in df.columns and df["y_fire"].sum() > 0:
        day = (
            df.loc[df["y_fire"] == 1]
            .groupby("label_date")
            .size()
            .sort_values(ascending=False)
            .index[0]
        )
        return pd.Timestamp(day).normalize()
    day = df.groupby("label_date")["confidence_pct"].mean().sort_values(ascending=False).index[0]
    return pd.Timestamp(day).normalize()


def plot_day(
    day_df: pd.DataFrame,
    boundary,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))

    if hasattr(boundary, "plot"):
        boundary.plot(ax=ax, facecolor="#f5f5f0", edgecolor="#333333", linewidth=0.8)
    else:
        for ring in boundary:
            ax.plot(ring[:, 0], ring[:, 1], color="#333333", linewidth=0.8)
            ax.fill(ring[:, 0], ring[:, 1], color="#f5f5f0", alpha=0.5)

    sc = ax.scatter(
        day_df["longitude"],
        day_df["latitude"],
        c=day_df["confidence_pct"],
        s=36,
        cmap="YlOrRd",
        norm=Normalize(vmin=0, vmax=max(10.0, float(day_df["confidence_pct"].quantile(0.95)))),
        edgecolors="k",
        linewidths=0.2,
        alpha=0.9,
        zorder=3,
    )
    if "y_fire" in day_df.columns:
        pos = day_df.loc[day_df["y_fire"] == 1]
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s (%d cells)", out_path, len(day_df))


def write_html(day_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Lightweight HTML scatter map (no folium dependency required)."""
    rows = []
    for r in day_df.itertuples(index=False):
        region = getattr(r, "region", r.cell_id)
        rows.append(
            f"<tr><td>{region}</td><td>{r.latitude:.3f}</td><td>{r.longitude:.3f}</td>"
            f"<td>{r.confidence_pct:.1f}</td><td>{int(getattr(r, 'y_fire', -1))}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
th {{ background: #f3f3f3; }}
</style></head>
<body>
<h2>{title}</h2>
<p>Cell-level wildfire risk predictions (calibrated confidence %).</p>
<table>
<thead><tr><th>Region</th><th>Lat</th><th>Lon</th><th>Confidence %</th><th>y_fire</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""
    out_path.write_text(html)
    logger.info("Wrote %s", out_path)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_dir = Path(cfg["paths"]["output_dir"]) / "model"
    maps_dir = Path(cfg["paths"]["output_dir"]) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        pred_path = Path(args.predictions)
    elif args.topk_source:
        pred_path = model_dir / "test_alerts_topk.csv"
    else:
        pred_path = model_dir / "test_predictions.parquet"
        if not pred_path.exists():
            pred_path = model_dir / "test_alerts_topk.csv"

    if pred_path.suffix == ".csv":
        df = pd.read_csv(pred_path, parse_dates=["label_date"])
    else:
        df = pd.read_parquet(pred_path)
        df["label_date"] = pd.to_datetime(df["label_date"])

    if "confidence_pct" not in df.columns and "p_fire" in df.columns:
        df["confidence_pct"] = df["p_fire"] * 100.0

    boundary = _load_boundary(Path(cfg["paths"]["california_geojson"]))

    dates = (
        sorted(df["label_date"].dt.normalize().unique())
        if args.all_dates
        else [_pick_date(df, args.date)]
    )

    for d in dates:
        day_df = df.loc[df["label_date"].dt.normalize() == pd.Timestamp(d)].copy()
        if day_df.empty:
            logger.warning("No rows for %s", d)
            continue
        stamp = pd.Timestamp(d).strftime("%Y-%m-%d")
        title = f"California wildfire risk — {stamp}"
        plot_day(day_df, boundary, maps_dir / f"risk_{stamp}.png", title)
        write_html(day_df, maps_dir / f"risk_{stamp}.html", title)

    return 0


if __name__ == "__main__":
    sys.exit(main())
