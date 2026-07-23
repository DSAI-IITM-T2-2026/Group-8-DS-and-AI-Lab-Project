#!/usr/bin/env python3
"""
Verify GCS wildfire datasets: file types, timestamps, gaps, preprocess notes.

Writes:
  data/processed/metadata/inventory_summary.json
  data/processed/metadata/missing_for_team.md
  data/processed/metadata/coverage_map_data.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GS_NO_SIGN_REQUEST"] = "YES"

from src import config
from src.data.inventory import (
    build_coverage_map_data,
    build_full_inventory,
    render_missing_markdown,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--no-s2-bounds",
        action="store_true",
        help="Skip opening S2 tile headers (faster; map tiles lack footprints)",
    )
    args = parser.parse_args()

    inv = build_full_inventory(
        year=args.year, s2_fetch_bounds=not args.no_s2_bounds
    )

    # Strip heavy tiles from summary JSON copy for readability — keep in coverage map
    summary = json.loads(json.dumps(inv))
    if "tiles_by_month" in summary["sources"]["Sentinel-2"]:
        # keep counts only in summary
        tbm = summary["sources"]["Sentinel-2"].pop("tiles_by_month")
        summary["sources"]["Sentinel-2"]["n_windows"] = len(tbm)
        summary["sources"]["Sentinel-2"]["windows"] = sorted(tbm.keys())

    summary_path = config.METADATA_DIR / "inventory_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    md = render_missing_markdown(inv)
    md_path = config.METADATA_DIR / "missing_for_team.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")

    map_data = build_coverage_map_data(inv)
    map_path = config.METADATA_DIR / "coverage_map_data.json"
    with open(map_path, "w") as f:
        json.dump(map_data, f, indent=2)
    # Also copy next to the HTML app for easy relative fetch
    map_dir = config.M3_ROOT / "reports" / "coverage_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    with open(map_dir / "coverage_map_data.json", "w") as f:
        json.dump(map_data, f, indent=2)
    print(f"Wrote {map_path} and {map_dir / 'coverage_map_data.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
