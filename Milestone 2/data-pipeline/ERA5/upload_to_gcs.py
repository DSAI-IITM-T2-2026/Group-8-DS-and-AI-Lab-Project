#!/usr/bin/env python3
"""Stage and upload ERA5 raw NetCDF files to Google Cloud Storage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

try:
    from google.cloud import storage
except ImportError:
    storage = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class UploadConfig:
    destination: str
    gcs_bucket: str
    gcs_project: str | None
    gcs_prefix: str
    dataset_subdir: str
    years: list[str]
    source_dir: Path
    staging_dir: Path
    skip_existing: bool
    manifest_path: Path

    @classmethod
    def from_yaml(cls, config_path: Path, args: argparse.Namespace) -> UploadConfig:
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        upload = raw["upload"]
        paths = raw["paths"]

        year_start = args.year_start or upload["years"]["start"]
        year_end = args.year_end or upload["years"]["end"]
        years = [str(y) for y in range(int(year_start), int(year_end) + 1)]

        staging = Path(args.staging_dir or upload["staging_dir"])
        prefix = args.gcs_prefix or upload["gcs_prefix"]
        bucket = args.gcs_bucket or upload["gcs_bucket"]
        project = resolve_gcp_project(args, upload)

        return cls(
            destination=upload["destination"],
            gcs_bucket=bucket,
            gcs_project=project,
            gcs_prefix=prefix.strip("/"),
            dataset_subdir=upload["dataset_subdir"].strip("/"),
            years=years,
            source_dir=Path(paths["raw"]),
            staging_dir=staging,
            skip_existing=upload.get("skip_existing", True),
            manifest_path=Path(paths["logs"]) / "gcs_upload_manifest.json",
        )

    def gcs_blob_name(self, year: str, filename: str) -> str:
        return f"{self.gcs_prefix}/{self.dataset_subdir}/{year}/{filename}"

    def staging_path(self, year: str, filename: str) -> Path:
        return self.staging_dir / self.dataset_subdir / year / filename


def resolve_gcp_project(args: argparse.Namespace, upload: dict) -> str:
    """Resolve GCP project ID from CLI, config, env, or service-account JSON."""
    if args.gcs_project:
        return args.gcs_project

    if upload.get("gcs_project"):
        return str(upload["gcs_project"])

    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if env_project:
        return env_project

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        path = Path(creds_path).expanduser()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if data.get("project_id"):
                return str(data["project_id"])

    raise ValueError(
        "GCP project ID is required. Set upload.gcs_project in config.yaml, "
        "export GOOGLE_CLOUD_PROJECT, or pass --gcs-project"
    )


def collect_files(config: UploadConfig) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for year in config.years:
        pattern = f"era5_{year}_*.nc"
        for path in sorted(config.source_dir.glob(pattern)):
            if path.is_file() and path.stat().st_size > 0:
                files.append((year, path))
    return files


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"uploaded": [], "updated_at": None}


def save_manifest(path: Path, uploaded: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "uploaded": sorted(uploaded),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


def stage_files(
    config: UploadConfig,
    files: list[tuple[str, Path]],
    use_symlinks: bool = True,
) -> list[tuple[str, Path, Path]]:
    """Create local staging directory tree and return (year, source, staged) tuples."""
    staged: list[tuple[str, Path, Path]] = []

    for year, source in files:
        dest = config.staging_path(year, source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            dest.unlink()

        if use_symlinks:
            dest.symlink_to(source.resolve())
        else:
            shutil.copy2(source, dest)

        staged.append((year, source, dest))

    return staged


def blob_exists(client: storage.Client, bucket_name: str, blob_name: str) -> bool:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()


def upload_file(
    client: storage.Client,
    bucket_name: str,
    local_path: Path,
    blob_name: str,
) -> str:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type="application/netcdf")
    return f"gs://{bucket_name}/{blob_name}"


def run_upload(config: UploadConfig, dry_run: bool = False, copy_staging: bool = False) -> int:
    if config.destination != "gcs":
        logger.error("Only gcs destination is supported")
        return 1

    if storage is None and not dry_run:
        logger.error("Install google-cloud-storage: pip install google-cloud-storage")
        return 1

    files = collect_files(config)
    if not files:
        logger.error("No files found for years %s in %s", config.years, config.source_dir)
        return 1

    manifest = load_manifest(config.manifest_path)
    already_uploaded = set(manifest.get("uploaded", []))

    logger.info(
        "Found %d files for years %s → gs://%s/%s/",
        len(files),
        ", ".join(config.years),
        config.gcs_bucket,
        config.gcs_prefix,
    )

    # Stage locally
    logger.info("Staging files under %s", config.staging_dir)
    staged = stage_files(config, files, use_symlinks=not copy_staging)
    logger.info("Staged %d files", len(staged))

    if dry_run:
        for year, source, staged_path in staged:
            blob_name = config.gcs_blob_name(year, source.name)
            status = "skip" if blob_name in already_uploaded else "upload"
            print(f"[{status}] {source.name} → gs://{config.gcs_bucket}/{blob_name}")
        return 0

    client = storage.Client(project=config.gcs_project)
    logger.info("Using GCP project: %s", config.gcs_project)
    uploaded_keys = list(already_uploaded)
    uploaded_count = 0
    skipped_count = 0

    with tqdm(total=len(staged), desc="Uploading to GCS", unit="file") as pbar:
        for year, source, staged_path in staged:
            blob_name = config.gcs_blob_name(year, source.name)
            gcs_uri = f"gs://{config.gcs_bucket}/{blob_name}"

            if config.skip_existing and blob_name in already_uploaded:
                skipped_count += 1
                pbar.set_postfix({"last": source.name, "status": "manifest-skip"})
                pbar.update(1)
                continue

            if config.skip_existing and blob_exists(client, config.gcs_bucket, blob_name):
                uploaded_keys.append(blob_name)
                skipped_count += 1
                pbar.set_postfix({"last": source.name, "status": "gcs-exists"})
                pbar.update(1)
                continue

            try:
                upload_file(client, config.gcs_bucket, source, blob_name)
                uploaded_keys.append(blob_name)
                uploaded_count += 1
                logger.info("Uploaded %s → %s", source.name, gcs_uri)
                pbar.set_postfix({"last": source.name, "status": "ok"})
            except Exception as exc:
                logger.error("Failed to upload %s: %s", source.name, exc)
                pbar.set_postfix({"last": source.name, "status": "fail"})
                save_manifest(config.manifest_path, uploaded_keys)
                pbar.update(1)
                return 1

            pbar.update(1)

    save_manifest(config.manifest_path, uploaded_keys)
    logger.info(
        "Upload complete: %d uploaded, %d skipped, %d total",
        uploaded_count,
        skipped_count,
        len(staged),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload ERA5 files to GCS")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="List uploads without sending")
    parser.add_argument("--copy", action="store_true", help="Copy files to staging instead of symlinks")
    parser.add_argument("--gcs-bucket", help="Override GCS bucket name")
    parser.add_argument("--gcs-project", help="GCP project ID (billing/quota project)")
    parser.add_argument("--gcs-prefix", help="Override GCS prefix")
    parser.add_argument("--staging-dir", help="Override local staging directory")
    parser.add_argument("--year-start", type=int, help="First year to upload")
    parser.add_argument("--year-end", type=int, help="Last year to upload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = UploadConfig.from_yaml(Path(args.config), args)
    return run_upload(config, dry_run=args.dry_run, copy_staging=args.copy)


if __name__ == "__main__":
    raise SystemExit(main())
