#!/usr/bin/env python3
"""Generate California-outline risk maps for every Stage C test day (~365).

Example:
  cd "Milestone 4/numerical_nextday"
  source .venv/bin/activate
  export PYTHONPATH=src MPLBACKEND=Agg OMP_NUM_THREADS=1
  python scripts/generate_test_risk_maps.py
  python scripts/generate_test_risk_maps.py --limit 5
  python scripts/generate_test_risk_maps.py --date 2025-10-21
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_test_risk_maps")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--date", default=None, help="Single YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=None, help="Max number of days (debug)")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--out-dir", default=None, help="Default: artifacts/maps/daily")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--write-predictions", action="store_true")
    args = p.parse_args()

    from numerical_nextday.config import load_config
    from numerical_nextday.eval.maps import load_boundary, plot_california_risk_day
    from numerical_nextday.eval.scoring import score_test_frame

    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else art / "maps" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    geo = Path(cfg["paths"].get("california_geojson", ROOT / "data" / "california.geojson"))
    if not Path(geo).exists():
        geo = ROOT / "data" / "california.geojson"
    boundary = load_boundary(Path(geo))

    logger.info("Scoring Stage C test set (router models)…")
    pred_cache = art / "test_predictions.parquet"
    if pred_cache.exists() and not args.write_predictions and args.date is None:
        # Reuse cached scores for full-year runs when present
        scored = pd.read_parquet(pred_cache)
        scored["label_date"] = pd.to_datetime(scored["label_date"])
        if "proba" not in scored.columns:
            scored = score_test_frame(cfg)
        logger.info("Loaded cached predictions %s (%d rows)", pred_cache, len(scored))
    else:
        scored = score_test_frame(cfg)
        if args.write_predictions or args.date is None:
            scored.to_parquet(pred_cache, index=False)
            logger.info("Wrote %s", pred_cache)

    dates = sorted(scored["label_date"].dt.normalize().unique())
    if args.date:
        want = pd.Timestamp(args.date).normalize()
        dates = [d for d in dates if pd.Timestamp(d).normalize() == want]
        if not dates:
            raise SystemExit(f"No test rows for {args.date}")
    if args.limit is not None:
        dates = dates[: args.limit]

    logger.info("Rendering %d daily maps → %s", len(dates), out_dir)
    written: list[str] = []
    for i, d in enumerate(dates, 1):
        stamp = pd.Timestamp(d).strftime("%Y-%m-%d")
        out_path = out_dir / f"risk_{stamp}.png"
        if args.skip_existing and out_path.exists():
            written.append(stamp)
            continue
        day_df = scored.loc[scored["label_date"].dt.normalize() == pd.Timestamp(d).normalize()]
        plot_california_risk_day(
            day_df,
            boundary,
            out_path,
            title=f"California wildfire risk — {stamp}",
            dpi=args.dpi,
        )
        written.append(stamp)
        if i % 30 == 0 or i == len(dates):
            logger.info("Progress %d / %d", i, len(dates))

    # Copy highlight days into figures/ for report/PPT
    fig_dir = art / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    day_stats = (
        scored.groupby(scored["label_date"].dt.strftime("%Y-%m-%d"))
        .agg(n_pos=("y_fire", "sum"), max_p=("proba", "max"))
        .sort_values(["n_pos", "max_p"], ascending=False)
    )
    if len(day_stats):
        sample_day = day_stats.index[0]
        src = out_dir / f"risk_{sample_day}.png"
        if src.exists():
            shutil.copy2(src, fig_dir / "risk_map_sample.png")
        mid = day_stats.loc[[x for x in day_stats.index if x[5:7] in ("07", "08", "09", "10")]]
        if len(mid):
            mid_day = mid.sort_values(["max_p", "n_pos"], ascending=False).index[0]
            src = out_dir / f"risk_{mid_day}.png"
            if src.exists():
                shutil.copy2(src, fig_dir / "risk_map_high_activity.png")

    (out_dir / "manifest.json").write_text(
        json.dumps({"n_days": len(written), "out_dir": str(out_dir)}, indent=2)
    )
    logger.info("Done — %d maps", len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
