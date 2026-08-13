"""Locates the real 86-feature ``final_processed/<date>_test.parquet`` table.

Two real sources, tried in order, exactly matching how
``daily_pipeline/utils/preprocess/export_inference_day.py`` writes this
file:

1. GCS: ``gs://<bucket>/final_processed/<date>_test.parquet`` (only if
   ``WILDFIRE_ALLOW_GCS`` is enabled). The object is streamed into memory.
2. The pipeline's local cache, as an offline-development fallback only.

If neither has the requested day, callers get ``None`` and must raise a
documented error (409/503) -- this module never returns synthetic rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import io
import logging
from pathlib import Path

import pandas as pd

from .config import ensure_pipeline_on_path, get_pipeline_config, get_settings

logger = logging.getLogger("wildfire_api.data_access")


class PreparedDataAccessError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedDay:
    frame: pd.DataFrame
    identity: str


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


def load_prepared_day(label_date: date) -> PreparedDay | None:
    """Load a prepared day plus a cache identity tied to its parquet object."""
    local = _local_path(label_date)
    settings = get_settings()
    if settings.allow_gcs_reads:
        try:
            ensure_pipeline_on_path()
            uri = _gcs_uri(label_date)
            rest = uri[5:]
            bucket_name, _, blob_name = rest.partition("/")
            from google.cloud import storage

            blob = storage.Client().bucket(bucket_name).blob(blob_name)
            if not blob.exists():
                if local.is_file():
                    stat = local.stat()
                    return PreparedDay(
                        frame=pd.read_parquet(local),
                        identity=f"local:{stat.st_size}:{stat.st_mtime_ns}",
                    )
                return None
            blob.reload()
            payload = blob.download_as_bytes()
            return PreparedDay(
                frame=pd.read_parquet(io.BytesIO(payload)),
                identity=f"gcs:{blob.generation}:{blob.size}",
            )
        except Exception as exc:
            try:
                from google.auth.exceptions import DefaultCredentialsError
                from google.api_core.exceptions import Forbidden, GoogleAPIError
            except ImportError:  # pragma: no cover - dependency is required in deployment
                DefaultCredentialsError = Forbidden = GoogleAPIError = ()  # type: ignore
            logger.exception("Unable to stream the prepared parquet from GCS")
            if isinstance(exc, DefaultCredentialsError):
                raise PreparedDataAccessError(
                    "cloud_authentication_failed",
                    "Cloud authentication is not configured for prepared-data access.",
                ) from exc
            if isinstance(exc, Forbidden):
                raise PreparedDataAccessError(
                    "storage_access_denied",
                    "The backend identity cannot read the prepared-data object.",
                ) from exc
            if isinstance(exc, GoogleAPIError):
                raise PreparedDataAccessError(
                    "storage_unavailable", "Prepared-data storage is temporarily unavailable."
                ) from exc
            raise PreparedDataAccessError(
                "storage_unavailable", "The prepared-data object could not be read."
            ) from exc
    if local.is_file():
        stat = local.stat()
        return PreparedDay(
            frame=pd.read_parquet(local),
            identity=f"local:{stat.st_size}:{stat.st_mtime_ns}",
        )
    return None


def load_final_processed(label_date: date) -> pd.DataFrame | None:
    """Compatibility wrapper returning only the real feature table."""
    prepared = load_prepared_day(label_date)
    return prepared.frame if prepared else None


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
