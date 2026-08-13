#!/usr/bin/env python3
"""One-shot publish static DEM parquet to GCS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("download.dem")


def publish_dem(
    local_path: Path,
    *,
    bucket: str = "wildfire-detection-first",
    prefix: str = "dem",
    dest_name: str = "era5_grid_dem_features.parquet",
    project: str | None = None,
    skip_existing: bool = True,
) -> str:
    from google.cloud import storage

    if not local_path.is_file():
        raise FileNotFoundError(f"DEM file not found: {local_path}")

    blob_name = f"{prefix.rstrip('/')}/{dest_name}"
    client = storage.Client(project=project) if project else storage.Client()
    blob = client.bucket(bucket).blob(blob_name)
    uri = f"gs://{bucket}/{blob_name}"

    if skip_existing and blob.exists():
        logger.info("DEM already published: %s", uri)
        return uri

    suffix = local_path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(local_path)
        tmp = local_path.with_suffix(".parquet")
        df.to_parquet(tmp, index=False)
        local_path = tmp

    blob.upload_from_filename(str(local_path), content_type="application/octet-stream")
    logger.info("Published DEM → %s", uri)
    return uri


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Publish DEM to GCS.")
    parser.add_argument(
        "--local",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "vendor/mvp_era5_dem/data/era5_grid_dem_features.parquet",
    )
    parser.add_argument("--bucket", default="wildfire-detection-first")
    parser.add_argument("--prefix", default="dem")
    parser.add_argument("--project", default="plated-mechanic-418917")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    publish_dem(
        args.local,
        bucket=args.bucket,
        prefix=args.prefix,
        project=args.project,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
