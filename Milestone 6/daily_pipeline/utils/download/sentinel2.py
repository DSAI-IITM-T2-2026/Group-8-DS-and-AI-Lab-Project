#!/usr/bin/env python3
"""Submit flat Sentinel-2 exports for windows overlapping a target date."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger("download.sentinel2")

S2_SRC = Path(__file__).resolve().parents[1] / "vendor" / "sentinel2"


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _add_s2_path() -> None:
    """Put vendor/sentinel2 on path so `s2_lib` is a proper package (relative imports)."""
    root = str(S2_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)


def windows_overlapping(target: date, window_days: int = 5) -> list[tuple[date, date]]:
    """Return 5-day windows (start, end) that contain target."""
    anchor = date(2018, 1, 1)
    delta = (target - anchor).days
    idx = delta // window_days
    results: list[tuple[date, date]] = []
    for offset in (-1, 0, 1):
        start = anchor + timedelta(days=(idx + offset) * window_days)
        end = start + timedelta(days=window_days - 1)
        if start <= target <= end:
            results.append((start, end))
    if not results:
        start = anchor + timedelta(days=idx * window_days)
        end = start + timedelta(days=window_days - 1)
        results.append((start, end))
    return results


def flat_stem(start: date, end: date) -> str:
    return f"s2feat_{start:%Y%m%d}_{end:%Y%m%d}"


def build_runtime_config(
    *,
    project_id: str,
    bucket: str,
    prefix: str,
    grid_asset_id: str,
    year: int,
) -> "AppConfig":
    _add_s2_path()
    from s2_lib.config import load_config  # type: ignore

    template = S2_SRC / "config" / "config.template.yaml"
    cfg = load_config(template, project_id=project_id)
    object.__setattr__(cfg, "project_id", project_id)
    object.__setattr__(cfg.grid, "asset_id", grid_asset_id)
    object.__setattr__(cfg.export, "bucket", bucket)
    object.__setattr__(cfg.export, "prefix", prefix)
    object.__setattr__(cfg.temporal, "start_year", year)
    object.__setattr__(cfg.temporal, "end_year", year)
    return cfg


def export_window(
    start: date,
    end: date,
    *,
    project_id: str = "plated-mechanic-418917",
    bucket: str = "wildfire-detection-first",
    prefix: str = "sentinel2",
    grid_asset_id: str = "projects/plated-mechanic-418917/assets/california_s2_grid_1km_v3",
    skip_existing: bool = True,
) -> str | None:
    _add_s2_path()
    from s2_lib import export as export_mod  # type: ignore
    from s2_lib.export import initialize, start_export  # type: ignore
    from s2_lib.sentinel2 import TimeWindow  # type: ignore
    from google.cloud import storage

    stem = flat_stem(start, end)
    blob_parquet = f"{prefix.rstrip('/')}/{stem}.parquet"
    gcs_uri = f"gs://{bucket}/{blob_parquet}"

    if skip_existing:
        client = storage.Client(project=project_id)
        if client.bucket(bucket).blob(blob_parquet).exists():
            logger.info("S2 already exists: %s", gcs_uri)
            return gcs_uri
        csv_blob = f"{prefix.rstrip('/')}/{stem}.csv"
        if client.bucket(bucket).blob(csv_blob).exists():
            logger.info("S2 CSV exists (parquet pending): gs://%s/%s", bucket, csv_blob)
            return f"gs://{bucket}/{csv_blob}"

    initialize(project_id)
    cfg = build_runtime_config(
        project_id=project_id,
        bucket=bucket,
        prefix=prefix,
        grid_asset_id=grid_asset_id,
        year=start.year,
    )

    def flat_export_prefix(_cfg, window):  # noqa: ANN001
        return f"{prefix.rstrip('/')}/{flat_stem(window.start_date, window.end_date)}"

    export_mod.export_prefix = flat_export_prefix  # type: ignore[attr-defined]

    window = TimeWindow(start_date=start, end_date=end)
    task_id, uri = start_export(cfg, window)
    logger.info("Started S2 export %s (task=%s)", uri, task_id)
    return uri


def download_s2_for_date(
    target: date,
    *,
    project_id: str = "plated-mechanic-418917",
    bucket: str = "wildfire-detection-first",
    prefix: str = "sentinel2",
    grid_asset_id: str = "projects/plated-mechanic-418917/assets/california_s2_grid_1km_v3",
    skip_existing: bool = True,
    window_days: int = 5,
) -> list[str]:
    uris: list[str] = []
    for start, end in windows_overlapping(target, window_days=window_days):
        uri = export_window(
            start,
            end,
            project_id=project_id,
            bucket=bucket,
            prefix=prefix,
            grid_asset_id=grid_asset_id,
            skip_existing=skip_existing,
        )
        if uri:
            uris.append(uri)
    return uris


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Submit S2 flat export for date.")
    parser.add_argument("--date", type=parse_date, required=True)
    parser.add_argument("--project", default="plated-mechanic-418917")
    parser.add_argument("--bucket", default="wildfire-detection-first")
    parser.add_argument("--prefix", default="sentinel2")
    parser.add_argument("--grid-asset", default="projects/plated-mechanic-418917/assets/california_s2_grid_1km_v3")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    download_s2_for_date(
        args.date,
        project_id=args.project,
        bucket=args.bucket,
        prefix=args.prefix,
        grid_asset_id=args.grid_asset,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
