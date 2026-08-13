#!/usr/bin/env python3
"""Verify GCS bucket prefixes and optional Earth Engine grid asset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_UTILS = _ROOT / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from bootstrap import bootstrap  # noqa: E402

bootstrap()

from config_loader import load_daily_config  # noqa: E402
from paths import utils_root  # noqa: E402

logger = logging.getLogger("verify_gcs")

REQUIRED_PREFIXES = (
    "firms",
    "sentinel2",
    "sentinel5p",
    "era5",
    "dem",
    "stage_c_knn",
    "final_processed",
)


def _storage_client(project: str | None = None):
    from google.cloud import storage

    return storage.Client(project=project) if project else storage.Client()


def list_prefix_samples(client, bucket: str, prefix: str, limit: int = 5) -> list[str]:
    blobs = client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/", max_results=limit)
    return [b.name for b in blobs]


def ensure_placeholder(client, bucket: str, prefix: str) -> None:
    marker = f"{prefix.rstrip('/')}/.keep"
    blob = client.bucket(bucket).blob(marker)
    if not blob.exists():
        blob.upload_from_string(b"daily_pipeline placeholder\n")
        logger.info("Created placeholder: gs://%s/%s", bucket, marker)


def verify_ee_grid_asset(asset_id: str, project_id: str) -> bool:
    try:
        import ee

        ee.Initialize(project=project_id)
        info = ee.data.getAsset(asset_id)
        logger.info("EE grid asset OK: %s (type=%s)", asset_id, info.get("type"))
        return True
    except Exception as exc:
        logger.warning("EE grid asset check failed (%s): %s", asset_id, exc)
        logger.warning(
            "If you changed GEE project, run: python utils/create_s2_grid_asset.py --project %s",
            project_id,
        )
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Verify GCS layout for daily pipeline.")
    parser.add_argument("--config", type=Path, default=utils_root() / "config.yaml")
    parser.add_argument("--create-missing", action="store_true", help="Create .keep markers.")
    parser.add_argument("--skip-ee", action="store_true")
    args = parser.parse_args()

    cfg = load_daily_config(args.config)
    bucket = cfg["gcs"]["bucket"]
    project = cfg["gee"]["project_id"]

    try:
        client = _storage_client(project)
        client.get_bucket(bucket)
        logger.info("Bucket listable: gs://%s", bucket)
    except Exception as exc:
        logger.error("Cannot access bucket gs://%s: %s", bucket, exc)
        return 1

    ok = True
    prefixes = cfg["gcs"]["prefixes"]
    for key in REQUIRED_PREFIXES:
        prefix = prefixes[key]
        samples = list_prefix_samples(client, bucket, prefix)
        if samples:
            logger.info("[%s] %d sample(s): %s", prefix, len(samples), samples[:3])
        else:
            logger.warning("[%s] empty or missing", prefix)
            if args.create_missing:
                ensure_placeholder(client, bucket, prefix)
            else:
                ok = False

    if not args.skip_ee:
        if not verify_ee_grid_asset(cfg["gee"]["grid_asset_id"], project):
            ok = False

    if ok:
        logger.info("Phase 0 verify passed.")
        return 0
    logger.error("Phase 0 verify failed — run with --create-missing or fix IAM/paths.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
