#!/usr/bin/env python3
"""Upload processed Copernicus DEM data to Google Cloud Storage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from dem_pipeline.config import PipelineConfig
from dem_pipeline.gcs_upload import GcsUploadConfig, collect_upload_files

logger = logging.getLogger("dem_pipeline.upload")


def _upload_file(
    client: storage.Client,
    bucket_name: str,
    local_path: Path,
    gcs_key: str,
    dry_run: bool,
) -> str:
    blob_uri = f"gs://{bucket_name}/{gcs_key}"
    size_mb = local_path.stat().st_size / (1024 * 1024)
    if dry_run:
        logger.info("[dry-run] %s (%.1f MB) -> %s", local_path.name, size_mb, blob_uri)
        return blob_uri

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_key)

    logger.info("Uploading %s (%.1f MB) -> %s", local_path.name, size_mb, blob_uri)
    blob.upload_from_filename(str(local_path), timeout=3600, retry=storage.retry.DEFAULT_RETRY)
    logger.info("Done: %s", local_path.name)
    return blob_uri


def _blob_exists(client: storage.Client, bucket_name: str, gcs_key: str) -> bool:
    return client.bucket(bucket_name).blob(gcs_key).exists()


def upload_dem_to_gcs(
    config_path: Path,
    *,
    dry_run: bool = False,
    include_all: bool = False,
    remaining_only: bool = False,
    skip_existing: bool = True,
) -> dict:
    pipeline_config = PipelineConfig.from_yaml(config_path, config_path.parent)
    upload_config = GcsUploadConfig.from_yaml(config_path)

    if include_all:
        upload_config.include_raw = True
        upload_config.include_merged = True
        upload_config.include_clipped = True
        upload_config.include_terrain = True
        upload_config.include_metadata = True
        upload_config.include_logs = True
    elif remaining_only:
        upload_config.include_raw = True
        upload_config.include_merged = True
        upload_config.include_clipped = False
        upload_config.include_terrain = False
        upload_config.include_metadata = False
        upload_config.include_logs = True

    if upload_config.destination != "gcs":
        raise ValueError(f"Unsupported destination: {upload_config.destination}")

    files = collect_upload_files(pipeline_config, upload_config)
    if not files:
        raise ValueError("No files selected for upload. Check upload config flags.")

    total_bytes = sum(path.stat().st_size for path, _ in files)
    logger.info(
        "Upload target: %s (%d files, %.2f GB)",
        upload_config.gcs_base_uri,
        len(files),
        total_bytes / (1024**3),
    )

    client = None if dry_run else storage.Client(project=upload_config.gcs_project_id)
    uploaded: list[dict] = []

    for local_path, gcs_key in files:
        if skip_existing and not dry_run and _blob_exists(
            client, upload_config.gcs_bucket, gcs_key
        ):
            logger.info("Skipping (already in GCS): %s", gcs_key)
            uploaded.append(
                {
                    "local": str(local_path),
                    "gcs_uri": f"gs://{upload_config.gcs_bucket}/{gcs_key}",
                    "size_bytes": local_path.stat().st_size,
                    "skipped": True,
                }
            )
            continue

        uri = _upload_file(
            client,
            upload_config.gcs_bucket,
            local_path,
            gcs_key,
            dry_run,
        )
        uploaded.append(
            {
                "local": str(local_path),
                "gcs_uri": uri,
                "size_bytes": local_path.stat().st_size,
            }
        )

    manifest = {
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": upload_config.gcs_project_id,
        "bucket": upload_config.gcs_bucket,
        "base_uri": upload_config.gcs_base_uri,
        "year_range": upload_config.year_range,
        "region": upload_config.region,
        "file_count": len(uploaded),
        "total_bytes": total_bytes,
        "files": uploaded,
    }

    manifest_key = (
        f"{upload_config.gcs_prefix}/dem/{upload_config.year_range}/"
        f"{upload_config.region}/metadata/upload_manifest.json"
    )
    manifest_path = pipeline_config.metadata_dir / "upload_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not dry_run:
        bucket = client.bucket(upload_config.gcs_bucket)
        bucket.blob(manifest_key).upload_from_string(
            json.dumps(manifest, indent=2),
            content_type="application/json",
        )
        logger.info("Uploaded manifest -> gs://%s/%s", upload_config.gcs_bucket, manifest_key)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload California DEM terrain data to GCS (2021-2025 project path)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without uploading",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload everything including raw, merged, clipped, terrain, metadata, logs",
    )
    parser.add_argument(
        "--remaining",
        action="store_true",
        help="Upload only files not included in the default run (raw, merged, logs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload files even if they already exist in GCS",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    try:
        manifest = upload_dem_to_gcs(
            args.config,
            dry_run=args.dry_run,
            include_all=args.all,
            remaining_only=args.remaining,
            skip_existing=not args.force,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Upload failed")
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
