#!/usr/bin/env python3
"""FIRMS daily GeoTIFF export to GCS via Earth Engine."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta

logger = logging.getLogger("download.firms")

DEFAULT_PROJECT = "plated-mechanic-418917"
DEFAULT_BUCKET = "wildfire-detection-first"
DEFAULT_PREFIX = "firms_daily_geotiff"


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def init_ee(project_id: str) -> None:
    from download.ee_init import initialize_ee

    initialize_ee(project_id)


def make_firms_day_image(day_str: str, aoi) -> "ee.Image":
    import ee

    day_start = ee.Date(day_str)
    day_end = day_start.advance(1, "day")
    firms_col = (
        ee.ImageCollection("FIRMS")
        .filterBounds(aoi)
        .filterDate(day_start, day_end)
    )
    confidence = firms_col.select("confidence").max().rename("firms_confidence")
    t21 = firms_col.select("T21").max().rename("firms_t21")
    label = firms_col.select("T21").count().gt(0).rename("label").toFloat()
    return ee.Image.cat([confidence, t21, label]).clip(aoi).toFloat()


def export_firms_day(
    target: date,
    *,
    project_id: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    skip_existing: bool = True,
) -> str | None:
    try:
        from download.gcs_listing import blob_listed
    except ImportError:
        from gcs_listing import blob_listed

    day_str = target.isoformat()
    blob_name = f"{prefix.rstrip('/')}/{day_str}.tif"
    gcs_uri = f"gs://{bucket}/{blob_name}"

    if skip_existing and blob_listed(bucket, blob_name, prefix=prefix, project=project_id):
        logger.info("Already exists: %s", gcs_uri)
        return gcs_uri

    import ee

    init_ee(project_id)
    california = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq("STUSPS", "CA"))
    aoi = california.geometry()

    task = ee.batch.Export.image.toCloudStorage(
        image=make_firms_day_image(day_str, aoi),
        description=f"firms_{day_str}",
        bucket=bucket,
        fileNamePrefix=f"{prefix.rstrip('/')}/{day_str}",
        region=aoi,
        scale=1000,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info("Started FIRMS export → %s (task=%s)", gcs_uri, task.id)
    return gcs_uri


def download_firms_range(
    start: date,
    end: date,
    *,
    project_id: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    skip_existing: bool = True,
    sleep_s: float = 0.5,
) -> list[str]:
    uris: list[str] = []
    current = start
    while current <= end:
        uri = export_firms_day(
            current,
            project_id=project_id,
            bucket=bucket,
            prefix=prefix,
            skip_existing=skip_existing,
        )
        if uri:
            uris.append(uri)
        current += timedelta(days=1)
        if not skip_existing:
            time.sleep(sleep_s)
    return uris


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export FIRMS daily GeoTIFF to GCS.")
    parser.add_argument("--date", type=parse_date, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    export_firms_day(
        args.date,
        project_id=args.project,
        bucket=args.bucket,
        prefix=args.prefix,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
