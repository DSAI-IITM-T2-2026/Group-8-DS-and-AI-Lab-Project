"""Local JSON progress tracking so re-running the pipeline is idempotent.

Statuses:
  empty     — window had no matching scenes
  submitted — GEE export task was started (not yet known complete)
  done      — GEE export task completed successfully
  failed    — GEE export task failed or was cancelled (will be retried)
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("sentinel5p")

TERMINAL_SKIP = frozenset({"done", "empty"})
IN_FLIGHT = frozenset({"submitted"})


def load_progress(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_progress(path: str, progress: dict) -> None:
    with open(path, "w") as f:
        json.dump(progress, f, indent=2, sort_keys=True)


def should_skip(entry: Optional[dict]) -> bool:
    """True if this window should not be processed again this run."""
    if not entry:
        return False
    return entry.get("status") in TERMINAL_SKIP | IN_FLIGHT


def upload_progress(path: str, cfg: dict) -> None:
    """Best-effort mirror of the local progress file to GCS metadata prefix."""
    meta_prefix = (cfg.get("progress") or {}).get("gcs_metadata_prefix")
    bucket_name = (cfg.get("export") or {}).get("gcs_bucket")
    if not meta_prefix or not bucket_name or not os.path.exists(path):
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"{meta_prefix.rstrip('/')}/{os.path.basename(path)}"
        bucket.blob(blob_name).upload_from_filename(path)
        logger.info(f"Progress mirrored to gs://{bucket_name}/{blob_name}")
    except Exception as exc:  # noqa: BLE001 — optional side channel must not abort run
        logger.warning(f"Could not upload progress to GCS (continuing): {exc}")
