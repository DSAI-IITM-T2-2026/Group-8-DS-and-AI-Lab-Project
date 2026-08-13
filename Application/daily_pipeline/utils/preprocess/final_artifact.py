"""GCS-first lookup for the validated preparation-to-inference artifact."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import io
import logging
from typing import Any

import pandas as pd

from preprocess.export_inference_day import validate_champion_artifact

logger = logging.getLogger("preprocess.final_artifact")


def existing_final_artifact(
    label_date: date,
    cfg: dict,
    *,
    storage_client: Any | None = None,
) -> dict | None:
    """Return metadata for a valid GCS final parquet, otherwise request a rebuild.

    The object is streamed into memory only. Storage/authentication failures are
    allowed to propagate; treating an unreachable bucket as an absent artifact
    would incorrectly start cloud downloads against the same broken dependency.
    """
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client(project=cfg["gcs"].get("project"))

    bucket_name = cfg["gcs"]["bucket"]
    prefix = cfg["gcs"]["prefixes"]["final_processed"].rstrip("/")
    object_name = f"{prefix}/{label_date.isoformat()}_test.parquet"
    blob = storage_client.bucket(bucket_name).blob(object_name)
    if not blob.exists():
        logger.info("Prepared parquet is absent: gs://%s/%s", bucket_name, object_name)
        return None

    blob.reload()
    if not blob.size:
        logger.warning(
            "Prepared parquet is empty and will be rebuilt: gs://%s/%s",
            bucket_name,
            object_name,
        )
        return None

    # Keep storage failures distinct from a malformed parquet: auth/network
    # errors must stop the job, while unreadable object contents request rebuild.
    payload = blob.download_as_bytes()
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
        feature_cols = validate_champion_artifact(frame, label_date, cfg)
    except Exception as exc:
        logger.warning(
            "Prepared parquet failed validation and will be rebuilt: gs://%s/%s (%s)",
            bucket_name,
            object_name,
            exc,
        )
        return None

    created_at = blob.updated or datetime.now().astimezone()
    logger.info("Reusing validated prepared parquet: gs://%s/%s", bucket_name, object_name)
    return {
        "objectUri": f"gs://{bucket_name}/{object_name}",
        "rowCount": int(len(frame)),
        "featureCount": len(feature_cols),
        "cellCount": int(frame["cell_id"].nunique()),
        "labelDate": label_date.isoformat(),
        "eoAsOfDate": (label_date - timedelta(days=1)).isoformat(),
        "featureEndDate": (
            label_date
            - timedelta(
                days=int(cfg["task"].get("era5_lag_days", 5))
                + int(cfg["task"].get("lead_days", 1))
            )
        ).isoformat(),
        "createdAt": created_at.isoformat(),
    }
