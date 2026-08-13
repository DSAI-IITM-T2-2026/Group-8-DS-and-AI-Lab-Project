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
S2_ANCHOR = date(2018, 1, 1)
_S2_WINDOW_RESULT: dict[tuple[str, str, date, date], str | None] = {}


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _add_s2_path() -> None:
    """Put vendor/sentinel2 on path so `s2_lib` is a proper package (relative imports)."""
    root = str(S2_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)


def windows_overlapping(target: date, window_days: int = 5) -> list[tuple[date, date]]:
    """Return unique 5-day windows (start, end) on the 2018-01-01 grid that contain target."""
    delta = (target - S2_ANCHOR).days
    idx = delta // window_days
    results: list[tuple[date, date]] = []
    seen: set[tuple[date, date]] = set()
    for offset in (-1, 0, 1):
        start = S2_ANCHOR + timedelta(days=(idx + offset) * window_days)
        end = start + timedelta(days=window_days - 1)
        if start <= target <= end and (start, end) not in seen:
            seen.add((start, end))
            results.append((start, end))
    if not results:
        start = S2_ANCHOR + timedelta(days=idx * window_days)
        end = start + timedelta(days=window_days - 1)
        results.append((start, end))
    return results


def window_index(start: date, window_days: int = 5) -> int:
    """1-based window index from the 2018-01-01 production grid."""
    return (start - S2_ANCHOR).days // window_days + 1


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
    cfg = load_config(template)
    object.__setattr__(cfg, "project_id", project_id)
    object.__setattr__(cfg.grid, "asset_id", grid_asset_id)
    object.__setattr__(cfg.export, "bucket", bucket)
    object.__setattr__(cfg.export, "prefix", prefix)
    # Keep 2018 anchor so find_window / indices match production; only need this year
    # for export metadata — do not seed full-year window lists.
    object.__setattr__(cfg.temporal, "start_year", 2018)
    object.__setattr__(cfg.temporal, "end_year", max(year, 2018))
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
    as_of: date | None = None,
    window_days: int = 5,
) -> str | None:
    as_of = as_of or date.today()
    if start > as_of:
        logger.info(
            "S2 skip future window %s…%s (starts after as_of=%s)",
            start,
            end,
            as_of,
        )
        return None

    stem = flat_stem(start, end)
    blob_parquet = f"{prefix.rstrip('/')}/{stem}.parquet"
    gcs_uri = f"gs://{bucket}/{blob_parquet}"
    cache_key = (bucket, prefix.rstrip("/"), start, end)

    if skip_existing and cache_key in _S2_WINDOW_RESULT:
        return _S2_WINDOW_RESULT[cache_key]

    if skip_existing:
        try:
            from download.gcs_listing import blob_listed
        except ImportError:
            from gcs_listing import blob_listed

        if blob_listed(bucket, blob_parquet, prefix=prefix, project=project_id):
            logger.info("S2 already exists: %s", gcs_uri)
            _S2_WINDOW_RESULT[cache_key] = gcs_uri
            return gcs_uri
        csv_blob = f"{prefix.rstrip('/')}/{stem}.csv"
        if blob_listed(bucket, csv_blob, prefix=prefix, project=project_id):
            csv_uri = f"gs://{bucket}/{csv_blob}"
            logger.info("S2 CSV exists (parquet pending): %s", csv_uri)
            _S2_WINDOW_RESULT[cache_key] = csv_uri
            return csv_uri

    _add_s2_path()
    from s2_lib import export as export_mod  # type: ignore
    from s2_lib.export import initialize, start_export  # type: ignore
    from s2_lib.sentinel2 import TimeWindow  # type: ignore

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

    idx = window_index(start, window_days=window_days)
    window = TimeWindow(
        window_id=f"{start.isoformat()}_{end.isoformat()}",
        start_date=start,
        end_date=end,
        window_index=idx,
    )
    task_id, uri = start_export(cfg, window)
    logger.info("Started S2 export %s (task=%s)", uri, task_id)
    if skip_existing:
        _S2_WINDOW_RESULT[cache_key] = uri
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
    as_of: date | None = None,
) -> list[str]:
    """Export unique overlapping 5-day windows; skip windows that start after as_of."""
    as_of = as_of or date.today()
    uris: list[str] = []
    seen: set[tuple[date, date]] = set()
    for start, end in windows_overlapping(target, window_days=window_days):
        if (start, end) in seen:
            continue
        seen.add((start, end))
        uri = export_window(
            start,
            end,
            project_id=project_id,
            bucket=bucket,
            prefix=prefix,
            grid_asset_id=grid_asset_id,
            skip_existing=skip_existing,
            as_of=as_of,
            window_days=window_days,
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
    parser.add_argument(
        "--as-of",
        type=parse_date,
        default=None,
        help="Do not submit windows that start after this date (default: today).",
    )
    args = parser.parse_args()
    download_s2_for_date(
        args.date,
        project_id=args.project,
        bucket=args.bucket,
        prefix=args.prefix,
        grid_asset_id=args.grid_asset,
        skip_existing=not args.no_skip_existing,
        as_of=args.as_of,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
