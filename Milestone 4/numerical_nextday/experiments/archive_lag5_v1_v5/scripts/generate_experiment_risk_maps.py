#!/usr/bin/env python3
"""Render teammate-style maps directly from saved experiment predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts"
    / "archive_training"
    / "lag5_v4_recall25"
    / "test_predictions.parquet"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "archive_training"
    / "lag5_v4_recall25"
    / "maps"
    / "daily"
)
HIGHLIGHT_DATES = ("2025-10-21", "2025-09-02", "2025-04-14")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PREDICTIONS
    )
    parser.add_argument(
        "--california-geojson",
        type=Path,
        default=PROJECT_ROOT / "data" / "california.geojson",
    )
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="One or more YYYY-MM-DD values; omit for the full test year",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--confidence-cap", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    from numerical_nextday.evaluation.teammate_style import (
        load_california_boundary,
        plot_california_risk_day,
    )

    predictions_path = args.predictions.resolve()
    boundary_path = args.california_geojson.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "label_date",
        "cell_id",
        "latitude",
        "longitude",
        "y_fire",
        "p_fire",
        "confidence_pct",
    ]
    scored = pd.read_parquet(predictions_path, columns=columns)
    scored["label_date"] = pd.to_datetime(scored["label_date"]).dt.normalize()
    boundary = load_california_boundary(boundary_path)

    available_dates = sorted(scored["label_date"].unique())
    if args.date:
        requested = [pd.Timestamp(value).normalize() for value in args.date]
        missing = [value for value in requested if value not in available_dates]
        if missing:
            formatted = ", ".join(value.strftime("%Y-%m-%d") for value in missing)
            raise SystemExit(f"No prediction rows for: {formatted}")
        dates = requested
    else:
        dates = available_dates
    if args.limit is not None:
        dates = dates[: args.limit]

    written: list[Path] = []
    for index, date in enumerate(dates, start=1):
        stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
        destination = out_dir / f"risk_{stamp}.png"
        if not (args.skip_existing and destination.exists()):
            day = scored.loc[scored["label_date"].eq(date)]
            plot_california_risk_day(
                day,
                boundary,
                destination,
                title=f"California wildfire risk — {stamp}",
                confidence_cap=args.confidence_cap,
                dpi=args.dpi,
            )
        written.append(destination)
        if index % 30 == 0 or index == len(dates):
            print(f"Rendered {index}/{len(dates)}: {destination}")

    figure_dir = (
        PROJECT_ROOT / "artifacts" / "archive_training" / "figures"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    for date in dates:
        stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
        if stamp in HIGHLIGHT_DATES:
            source = out_dir / f"risk_{stamp}.png"
            shutil.copy2(source, figure_dir / f"risk_map_{stamp}.png")

    day_stats = scored.groupby("label_date").agg(
        n_pos=("y_fire", "sum"),
        max_confidence=("confidence_pct", "max"),
    )
    sample_date = day_stats.sort_values(
        ["n_pos", "max_confidence"], ascending=False
    ).index[0]
    sample_source = (
        out_dir / f"risk_{pd.Timestamp(sample_date):%Y-%m-%d}.png"
    )
    if sample_source.exists():
        shutil.copy2(sample_source, figure_dir / "risk_map_sample.png")

    manifest = {
        "model": "V4 lag-5 recall@25 blend",
        "prediction_file": str(predictions_path),
        "prediction_sha256": file_sha256(predictions_path),
        "california_geojson": str(boundary_path),
        "n_days": len(written),
        "first_date": (
            pd.Timestamp(dates[0]).strftime("%Y-%m-%d") if dates else None
        ),
        "last_date": (
            pd.Timestamp(dates[-1]).strftime("%Y-%m-%d") if dates else None
        ),
        "confidence_scale_percent": [0.0, args.confidence_cap],
        "confidence_values_above_cap_are_saturated": True,
        "style": "CA outline + YlOrRd cell dots + blue FIRMS rings",
        "target_timing": "y_fire is D+1; ERA5 predictors end at D-5.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
