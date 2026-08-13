"""Locates the real 86-feature ``final_processed/<date>_test.parquet`` table.

Two real sources, tried in order, exactly matching how
``daily_pipeline/utils/preprocess/export_inference_day.py`` writes this
file:

1. The pipeline's own local cache: ``daily_pipeline/.cache/final_processed/``
   -- populated by ``python run_daily.py all --label-date ...``.
2. GCS: ``gs://<bucket>/final_processed/<date>_test.parquet`` (only if
   ``WILDFIRE_ALLOW_GCS`` is enabled and google-cloud-storage / credentials
   are actually available).

If neither has the requested day, callers get ``None`` and must raise a
documented error (409/503) -- this module never returns synthetic rows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import ensure_pipeline_on_path, get_pipeline_config, get_settings


def _local_path(label_date: date) -> Path:
    ensure_pipeline_on_path()
    from paths import resolve_path

    cfg = get_pipeline_config()
    cache = resolve_path(cfg, "local_cache")
    return cache / "final_processed" / f"{label_date.isoformat()}_test.parquet"


def _gcs_uri(label_date: date) -> str:
    cfg = get_pipeline_config()
    bucket = cfg["gcs"]["bucket"]
    prefix = cfg["gcs"]["prefixes"]["final_processed"]
    return f"gs://{bucket}/{prefix}/{label_date.isoformat()}_test.parquet"


def load_final_processed(label_date: date) -> pd.DataFrame | None:
    """Real feature table for one label date, or None if not available anywhere."""
    local = _local_path(label_date)
    if local.is_file():
        return pd.read_parquet(local)

    settings = get_settings()
    if not settings.allow_gcs_reads:
        return None
    try:
        ensure_pipeline_on_path()
        from preprocess.adapters_gcs import download_blob

        uri = _gcs_uri(label_date)
        rest = uri[5:]
        bucket_name, _, blob_name = rest.partition("/")
        from google.cloud import storage

        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            return None
        local.parent.mkdir(parents=True, exist_ok=True)
        download_blob(uri, local)
        return pd.read_parquet(local)
    except Exception:
        # No credentials, no network, bucket unreachable, or the object is
        # genuinely absent -- any of these means "not available", not an error
        # worth crashing the request over. The caller turns this into a
        # documented 503/409.
        return None


def load_historical_archive() -> pd.DataFrame | None:
    """Optional multi-year archive used only by /validation/events.

    Not fetched automatically (it is multiple GB). See
    daily_pipeline/README.md "Zero-download 2025 replay" for the
    ``gsutil cp`` command that produces a local copy, then point
    ``WILDFIRE_HISTORICAL_ARCHIVE`` at it.
    """
    settings = get_settings()
    path = settings.historical_archive_path
    if path is None or not path.is_file():
        return None
    return pd.read_parquet(path)
