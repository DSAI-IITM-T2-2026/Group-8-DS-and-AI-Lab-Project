#!/usr/bin/env python3
"""Wrap ERA5 daily CDS download → gs://wildfire-detection-first/era5/."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger("download.era5")


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def era5_script_path() -> Path:
    return Path(__file__).resolve().parent / "_era5_cds.py"


def download_era5_day(
    target: date,
    *,
    bucket: str,
    prefix: str,
    project: str | None = None,
    workdir: str = "/tmp/era5",
    skip_existing: bool = True,
) -> int:
    script = era5_script_path()
    if not script.is_file():
        raise FileNotFoundError(f"ERA5 script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--date",
        target.isoformat(),
        "--gcs-bucket",
        bucket,
        "--gcs-prefix",
        prefix,
        "--workdir",
        workdir,
    ]
    if project:
        cmd.extend(["--gcs-project", project])
    if skip_existing:
        cmd.append("--skip-existing")
    else:
        cmd.append("--no-skip-existing")

    logger.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download one ERA5 day to GCS.")
    parser.add_argument("--date", type=parse_date, required=True)
    parser.add_argument("--bucket", default="wildfire-detection-first")
    parser.add_argument("--prefix", default="era5")
    parser.add_argument("--project", default="plated-mechanic-418917")
    parser.add_argument("--workdir", default="/tmp/era5")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    return download_era5_day(
        args.date,
        bucket=args.bucket,
        prefix=args.prefix,
        project=args.project,
        workdir=args.workdir,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
