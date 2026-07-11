#!/usr/bin/env python3
"""Run the Copernicus DEM GLO-30 acquisition and processing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dem_pipeline.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and process Copernicus DEM GLO-30 data for wildfire modeling."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to pipeline configuration YAML (default: ./config.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: directory containing config)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and use existing raw tiles",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge and use existing merged DEM",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip clip and use existing clipped DEM",
    )
    parser.add_argument(
        "--skip-terrain",
        action="store_true",
        help="Skip terrain generation and use existing terrain rasters",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    try:
        result = run_pipeline(
            args.config,
            args.project_root,
            skip_download=args.skip_download,
            skip_merge=args.skip_merge,
            skip_clip=args.skip_clip,
            skip_terrain=args.skip_terrain,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["qa_passed"] else 2
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
