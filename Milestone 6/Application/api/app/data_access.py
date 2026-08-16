"""Locates the real 86-feature ``final_processed/<date>_test.parquet`` table.

Sources, tried in order:

1. GCS: ``gs://<bucket>/final_processed/<date>_test.parquet`` (only if
   ``WILDFIRE_ALLOW_GCS`` is enabled). The object is streamed into memory.
2. The pipeline's local cache, as an offline-development fallback.
3. For 2019–2025 only: slice the combined historical archive into a daily
   parquet (same path as Generate's ``existing_final_artifact``), then load it.

If none has the requested day, callers get ``None`` and must raise a
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


@dataclass(frozen=True)
class ObservedLabelDay:
    frame: pd.DataFrame
    source: str
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


def _from_local(local: Path) -> PreparedDay | None:
    if not local.is_file():
        return None
    stat = local.stat()
    return PreparedDay(
        frame=pd.read_parquet(local),
        identity=f"local:{stat.st_size}:{stat.st_mtime_ns}",
    )


def _materialize_historical_day(label_date: date) -> PreparedDay | None:
    """Reuse Generate's archive slice so scoring does not 409 after historical reuse."""
    ensure_pipeline_on_path()
    from preprocess.final_artifact import materialize_from_historical_archive

    artifact = materialize_from_historical_archive(label_date, get_pipeline_config())
    if artifact is None:
        return None
    local = _local_path(label_date)
    prepared = _from_local(local)
    if prepared is not None:
        return prepared
    # Materialize may have uploaded only; identity still ties to the object URI.
    return None


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
                prepared = _from_local(local)
                if prepared is not None:
                    return prepared
                return _materialize_historical_day(label_date)
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
    prepared = _from_local(local)
    if prepared is not None:
        return prepared
    return _materialize_historical_day(label_date)


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


def _historical_label_day(label_date: date) -> ObservedLabelDay | None:
    ensure_pipeline_on_path()
    from preprocess.final_artifact import historical_archive_uri, slice_historical_archive

    cfg = get_pipeline_config()
    source = historical_archive_uri(cfg)
    if not source:
        return None
    try:
        frame = slice_historical_archive(source, label_date)
    except Exception as exc:
        raise PreparedDataAccessError(
            "validation_storage_unavailable",
            "The historical FIRMS label archive could not be read.",
        ) from exc
    if frame is None or frame.empty:
        return None
    required = {"cell_id", "y_fire"}
    if not required.issubset(frame.columns):
        raise PreparedDataAccessError(
            "invalid_validation_data",
            "The historical label archive is missing required FIRMS label columns.",
        )
    return ObservedLabelDay(
        frame=frame,
        source="historical_archive",
        identity=f"archive:{source}:{label_date.isoformat()}",
    )


def _live_firms_label_day(label_date: date) -> ObservedLabelDay | None:
    settings = get_settings()
    if not settings.allow_gcs_reads:
        raise PreparedDataAccessError(
            "validation_storage_unavailable",
            "Cloud reads are disabled, so completed FIRMS labels cannot be loaded.",
        )
    cfg = get_pipeline_config()
    bucket_name = cfg["gcs"]["bucket"]
    prefix = cfg["gcs"]["prefixes"]["firms"].rstrip("/")
    object_name = f"{prefix}/{label_date.isoformat()}.tif"
    try:
        from google.cloud import storage

        blob = storage.Client(project=cfg["gcs"].get("project")).bucket(bucket_name).blob(object_name)
        if not blob.exists():
            return None
        blob.reload()
        if not blob.size:
            raise PreparedDataAccessError(
                "invalid_validation_data", "The completed FIRMS label object is empty."
            )

        ensure_pipeline_on_path()
        from config_loader import load_m4_config, setup_m4_imports

        setup_m4_imports(cfg)
        from numerical_nextday.data.m3_imports import load_mvp_modules

        firms_module = load_mvp_modules(load_m4_config(cfg))["firms_labels"]
        frame = firms_module.label_day_to_cells(
            pd.Timestamp(label_date),
            cfg["gcs"]["firms_vsigs_prefix"],
            float(cfg["task"].get("firms_confidence_min", 30)),
            strict=True,
        )
    except PreparedDataAccessError:
        raise
    except Exception as exc:
        try:
            from google.auth.exceptions import DefaultCredentialsError
            from google.api_core.exceptions import Forbidden
        except ImportError:  # pragma: no cover
            DefaultCredentialsError = Forbidden = ()  # type: ignore
        if isinstance(exc, (DefaultCredentialsError, Forbidden)):
            raise PreparedDataAccessError(
                "validation_storage_access_denied",
                "The backend identity cannot read completed FIRMS labels.",
            ) from exc
        raise PreparedDataAccessError(
            "invalid_validation_data",
            "The completed FIRMS label object could not be decoded safely.",
        ) from exc

    return ObservedLabelDay(
        frame=frame,
        source="firms_daily_geotiff",
        identity=f"gcs:{blob.generation}:{blob.size}",
    )


def load_observed_label_day(label_date: date) -> ObservedLabelDay | None:
    """Load post-event truth without mutating or trusting prediction artifacts."""
    if label_date.year <= 2025:
        return _historical_label_day(label_date)
    return _live_firms_label_day(label_date)
