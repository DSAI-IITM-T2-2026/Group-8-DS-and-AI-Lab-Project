#!/usr/bin/env python3
"""CLI for terrain feature lookup at a latitude/longitude."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dem_pipeline.lookup import get_terrain_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Look up terrain features for a point in the California study area."
    )
    parser.add_argument("latitude", type=float, help="Latitude in decimal degrees")
    parser.add_argument("longitude", type=float, help="Longitude in decimal degrees")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    try:
        features = get_terrain_features(
            args.latitude,
            args.longitude,
            config_path=args.config,
        )
        print(json.dumps(features.to_dict(), indent=2))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
