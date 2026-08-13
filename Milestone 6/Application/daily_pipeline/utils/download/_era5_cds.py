#!/usr/bin/env python3
"""Self-contained daily ERA5 download + GCS upload (Cloud Run friendly).

No local project imports — only stdlib + cdsapi + google-cloud-storage.

Examples
--------
  python download_era5_day_to_gcs.py --date 2024-07-01
  DATE=2024-07-01 python download_era5_day_to_gcs.py

Credentials
-----------
  CDS: CDS_API_KEY=UID:API_KEY  or  ~/.cdsapirc (url + key)
  GCS: Application Default Credentials / Cloud Run service account
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger("era5_day")

# ---------------------------------------------------------------------------
# Defaults (aligned with Milestone 2 ERA5/config.yaml)
# ---------------------------------------------------------------------------
DATASET = "reanalysis-era5-single-levels"
PRODUCT_TYPE = "reanalysis"
AREA = [42.01, -124.41, 32.53, -114.13]  # N, W, S, E
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "high_vegetation_cover",
    "low_vegetation_cover",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "boundary_layer_height",
]
TIMES = [f"{h:02d}:00" for h in range(24)]

DEFAULT_GCS_BUCKET = "dsai-lab-project"
DEFAULT_GCS_PREFIX = "wildfire_satellite/era5/raw"
DEFAULT_GCS_PROJECT = "iitm-dsai-lab"
DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"
DEFAULT_WORKDIR = "/tmp/era5"

MAX_RETRIES = 8
RETRY_BASE_DELAY = 60
QUEUE_LIMIT_MAX_RETRIES = 30
QUEUE_LIMIT_DELAY = 600


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def resolve_date(cli_date: date | None) -> date:
    if cli_date is not None:
        return cli_date
    env = os.environ.get("DATE") or os.environ.get("ERA5_DATE")
    if env:
        return parse_date(env)
    raise SystemExit(
        "No date provided. Pass --date YYYY-MM-DD or set DATE / ERA5_DATE."
    )


def local_filename(target: date) -> str:
    return f"era5_{target.year:04d}_{target.month:02d}_{target.day:02d}.nc"


def gcs_blob_name(prefix: str, target: date, filename: str) -> str:
    return f"{prefix.strip('/')}/{target.year:04d}/{filename}"


def build_request(target: date) -> dict:
    return {
        "product_type": PRODUCT_TYPE,
        "variable": VARIABLES,
        "year": [f"{target.year:04d}"],
        "month": [f"{target.month:02d}"],
        "day": [f"{target.day:02d}"],
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def load_cds_credentials() -> tuple[str, str]:
    """Return (url, key) from env or ~/.cdsapirc."""
    key = os.environ.get("CDS_API_KEY") or os.environ.get("CDSAPI_KEY")
    url = (
        os.environ.get("CDS_API_URL")
        or os.environ.get("CDSAPI_URL")
        or DEFAULT_CDS_URL
    )
    if key:
        return url, key

    rc_path = Path(os.path.expanduser(os.environ.get("CDS_API_RC") or "~/.cdsapirc"))
    if not rc_path.exists():
        raise SystemExit(
            "CDS credentials not found. Set CDS_API_KEY (UID:API_KEY) "
            f"or create {rc_path}."
        )

    url_from_file = DEFAULT_CDS_URL
    key_from_file = None
    for line in rc_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, value = line.split(":", 1)
        name, value = name.strip(), value.strip()
        if name == "url":
            url_from_file = value
        elif name == "key":
            key_from_file = value

    if not key_from_file:
        raise SystemExit(f"No 'key' found in CDS credentials file: {rc_path}")
    return url_from_file, key_from_file


def _is_queue_limit_error(error: str) -> bool:
    lowered = error.lower()
    return "queued requests" in lowered or "temporarily limited" in lowered


def _is_licence_error(error: str) -> bool:
    lowered = error.lower()
    return "licence" in lowered or "license" in lowered


def download_era5_day(target: date, outfile: Path) -> Path:
    import cdsapi

    url, key = load_cds_credentials()
    client = cdsapi.Client(url=url, key=key)
    payload = build_request(target)

    logger.info(
        "Requesting ERA5 for %s (area=%s, %d vars, 24h)",
        target.isoformat(),
        AREA,
        len(VARIABLES),
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)

    delay = RETRY_BASE_DELAY
    attempt = 0
    queue_attempts = 0
    last_error: str | None = None

    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            if outfile.exists():
                outfile.unlink()
            client.retrieve(DATASET, payload, str(outfile))
            if not outfile.exists() or outfile.stat().st_size == 0:
                raise RuntimeError(f"Download produced empty file: {outfile}")
            logger.info(
                "Downloaded %s (%.1f MB)", outfile.name, outfile.stat().st_size / 1e6
            )
            return outfile
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Download attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if outfile.exists():
                outfile.unlink(missing_ok=True)

            if _is_licence_error(last_error):
                raise SystemExit(
                    "CDS account has not accepted the ERA5 licence. Accept it at:\n"
                    "https://cds.climate.copernicus.eu/datasets/"
                    "reanalysis-era5-single-levels?tab=download#manage-licences"
                ) from exc

            if _is_queue_limit_error(last_error):
                queue_attempts += 1
                if queue_attempts > QUEUE_LIMIT_MAX_RETRIES:
                    break
                logger.warning(
                    "CDS queue full — sleeping %ds (queue attempt %d/%d)",
                    QUEUE_LIMIT_DELAY,
                    queue_attempts,
                    QUEUE_LIMIT_MAX_RETRIES,
                )
                time.sleep(QUEUE_LIMIT_DELAY)
                attempt -= 1
                continue

            if attempt < MAX_RETRIES:
                logger.info("Retrying in %d seconds...", delay)
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(f"ERA5 download failed for {target}: {last_error}")


def resolve_gcs_project(cli_project: str | None) -> str | None:
    if cli_project:
        return cli_project
    env = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if env:
        return env
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        path = Path(creds_path).expanduser()
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("project_id"):
                return str(data["project_id"])
    return DEFAULT_GCS_PROJECT


def storage_client(project: str | None):
    from google.cloud import storage

    return storage.Client(project=project) if project else storage.Client()


def blob_exists(client, bucket: str, blob_name: str) -> bool:
    return client.bucket(bucket).blob(blob_name).exists()


def upload_to_gcs(
    local_path: Path,
    *,
    bucket: str,
    blob_name: str,
    project: str | None,
) -> str:
    client = storage_client(project)
    gcs_uri = f"gs://{bucket}/{blob_name}"
    logger.info("Uploading %s → %s", local_path.name, gcs_uri)
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type="application/netcdf")
    logger.info("Upload complete: %s", gcs_uri)
    return gcs_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained: download one ERA5 day and upload to GCS."
    )
    parser.add_argument(
        "--date",
        type=parse_date,
        default=None,
        help="Target day YYYY-MM-DD (or set DATE / ERA5_DATE).",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=os.environ.get("GCS_BUCKET", DEFAULT_GCS_BUCKET),
    )
    parser.add_argument(
        "--gcs-prefix",
        default=os.environ.get("GCS_PREFIX", DEFAULT_GCS_PREFIX),
    )
    parser.add_argument(
        "--gcs-project",
        default=None,
        help=f"GCP project (default: env or {DEFAULT_GCS_PROJECT}).",
    )
    parser.add_argument(
        "--workdir",
        default=os.environ.get("WORKDIR", DEFAULT_WORKDIR),
        help="Local directory for the NetCDF before upload.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip work if GCS object already exists (default: true).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Download only; keep local file.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip CDS; upload existing local file from --workdir.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep local NetCDF after upload (default: delete).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print request + destination without downloading/uploading.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    target = resolve_date(args.date)
    filename = local_filename(target)
    blob_name = gcs_blob_name(args.gcs_prefix, target, filename)
    project = resolve_gcs_project(args.gcs_project)
    gcs_uri = f"gs://{args.gcs_bucket}/{blob_name}"
    local_path = Path(args.workdir) / filename

    logger.info("=== Daily ERA5 → GCS ===")
    logger.info("Date:     %s", target.isoformat())
    logger.info("Local:    %s", local_path)
    logger.info("GCS:      %s", gcs_uri)
    logger.info("Project:  %s", project)

    if args.dry_run:
        logger.info("CDS payload: %s", build_request(target))
        return 0

    if args.skip_existing and not args.upload_only:
        client = storage_client(project)
        if blob_exists(client, args.gcs_bucket, blob_name):
            logger.info("Already in bucket, nothing to do: %s", gcs_uri)
            return 0

    if not args.upload_only:
        try:
            download_era5_day(target, local_path)
        except Exception:
            logger.exception("CDS download failed")
            return 1
    else:
        if not local_path.exists() or local_path.stat().st_size == 0:
            logger.error("Local file not found for --upload-only: %s", local_path)
            return 1

    if args.skip_upload:
        logger.info("Skipping upload. Local file: %s", local_path)
        return 0

    try:
        upload_to_gcs(
            local_path,
            bucket=args.gcs_bucket,
            blob_name=blob_name,
            project=project,
        )
    except Exception:
        logger.exception("GCS upload failed")
        return 1
    finally:
        if not args.keep_local and not args.skip_upload and local_path.exists():
            local_path.unlink()
            logger.info("Removed local file %s", local_path)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
