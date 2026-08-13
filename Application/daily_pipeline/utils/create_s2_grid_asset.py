#!/usr/bin/env python3
"""Create a Sentinel-2 1 km California grid asset in Earth Engine for a GEE project.

Use this when you switch to a new GEE project_id and do not already have:
  projects/<PROJECT>/assets/california_s2_grid_1km_v3

After the EE task completes (can take a long time for ~413k cells):
  1. Copy the printed asset id into utils/config.yaml → gee.grid_asset_id
  2. Set gee.project_id to the same project
  3. python verify_gcs.py  (without --skip-ee)

Examples
--------
  python utils/create_s2_grid_asset.py
  python utils/create_s2_grid_asset.py --project MY-GEE-PROJECT
  python utils/create_s2_grid_asset.py --project MY-GEE-PROJECT --asset-name california_s2_grid_1km_v3
  python utils/create_s2_grid_asset.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_UTILS = Path(__file__).resolve().parent
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from bootstrap import bootstrap  # noqa: E402

bootstrap()

from config_loader import load_daily_config  # noqa: E402

logger = logging.getLogger("create_s2_grid_asset")

DEFAULT_ASSET_NAME = "california_s2_grid_1km_v3"
EXPECTED_ROWS = 413_115


def _add_s2_path() -> None:
    s2 = _UTILS / "vendor" / "sentinel2"
    if str(s2) not in sys.path:
        sys.path.insert(0, str(s2))


def build_cfg(project_id: str, asset_id: str):
    _add_s2_path()
    from s2_lib.config import load_config  # type: ignore

    template = _UTILS / "vendor" / "sentinel2" / "config" / "config.template.yaml"
    cfg = load_config(template)
    object.__setattr__(cfg, "project_id", project_id)
    object.__setattr__(cfg.grid, "asset_id", asset_id)
    return cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Create S2 1 km grid EE asset for a GEE project.")
    parser.add_argument(
        "--project",
        default=None,
        help="GEE project id (default: utils/config.yaml gee.project_id)",
    )
    parser.add_argument(
        "--asset-name",
        default=DEFAULT_ASSET_NAME,
        help=f"Asset name under projects/<project>/assets/ (default: {DEFAULT_ASSET_NAME})",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="Full asset id override (projects/.../assets/...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target asset id and exit without starting EE task.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start create even if asset already exists.",
    )
    args = parser.parse_args()

    daily = load_daily_config()
    project_id = args.project or daily["gee"]["project_id"]
    asset_id = args.asset_id or f"projects/{project_id}/assets/{args.asset_name}"

    logger.info("GEE project:  %s", project_id)
    logger.info("Grid asset:   %s", asset_id)
    logger.info("Expected ~%s cells when complete", f"{EXPECTED_ROWS:,}")

    if args.dry_run:
        logger.info("Dry run — put this in utils/config.yaml:")
        print(f"gee:\n  project_id: {project_id}\n  grid_asset_id: {asset_id}")
        return 0

    import ee

    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    if not args.force:
        try:
            info = ee.data.getAsset(asset_id)
            logger.info("Asset already exists: %s (type=%s)", asset_id, info.get("type"))
            logger.info("Update utils/config.yaml gee.grid_asset_id if needed. Use --force to recreate.")
            return 0
        except Exception:
            logger.info("Asset not found — creating export task...")

    _add_s2_path()
    from s2_lib.export import create_grid_export_task, initialize  # type: ignore

    initialize(project_id)
    cfg = build_cfg(project_id, asset_id)
    task = create_grid_export_task(cfg)
    task.start()
    logger.info("Started grid asset task id=%s", task.id)
    logger.info("Wait until COMPLETED in EE Tasks, then set in utils/config.yaml:")
    print(f"gee:\n  project_id: {project_id}\n  grid_asset_id: {asset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
