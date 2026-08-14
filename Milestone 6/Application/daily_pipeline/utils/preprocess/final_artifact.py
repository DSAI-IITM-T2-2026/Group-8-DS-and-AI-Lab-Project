"""GCS-first lookup for the validated preparation-to-inference artifact."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import io
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from paths import resolve_path
from preprocess.export_inference_day import validate_champion_artifact

logger = logging.getLogger("preprocess.final_artifact")

HISTORICAL_START = date(2019, 1, 1)
HISTORICAL_END = date(2025, 12, 31)


def _gcs_writes_allowed() -> bool:
    return os.environ.get("WILDFIRE_ALLOW_GCS", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _in_historical_range(label_date: date) -> bool:
    return HISTORICAL_START <= label_date <= HISTORICAL_END


def historical_archive_uri(cfg: dict) -> str | None:
    """Resolve the combined 2019–2025 archive (env override, then config)."""
    for key in ("WILDFIRE_HISTORICAL_ARCHIVE_URI", "WILDFIRE_HISTORICAL_ARCHIVE"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    hist = (cfg.get("gcs") or {}).get("historical_archive")
    if not hist:
        return None
    hist = str(hist).strip()
    if hist.startswith("gs://") or hist.startswith("/"):
        return hist
    expanded = Path(hist).expanduser()
    if expanded.exists():
        return str(expanded)
    bucket = cfg["gcs"]["bucket"]
    return f"gs://{bucket}/{hist.lstrip('/')}"


def _artifact_dict(
    frame: pd.DataFrame,
    label_date: date,
    cfg: dict,
    *,
    object_uri: str,
    feature_cols: list[str],
    created_at: datetime,
) -> dict:
    return {
        "objectUri": object_uri,
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


def _slice_historical_archive(source: str, label_date: date) -> pd.DataFrame | None:
    """Read only rows for label_date from the combined archive (filter pushdown)."""
    day_ts = pd.Timestamp(label_date)
    filter_values: list[Any] = [day_ts, day_ts.to_pydatetime(), label_date, label_date.isoformat()]

    if source.startswith("gs://"):
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.fs as pafs

        rest = source[5:]
        bucket_name, _, blob_name = rest.partition("/")
        filesystem = pafs.GcsFileSystem()
        path = f"{bucket_name}/{blob_name}"
        dataset = ds.dataset(path, filesystem=filesystem, format="parquet")
        frame: pd.DataFrame | None = None
        for value in filter_values:
            try:
                table = dataset.to_table(filter=pc.equal(ds.field("label_date"), value))
            except Exception:
                continue
            if table.num_rows:
                frame = table.to_pandas()
                break
        if frame is None or frame.empty:
            return None
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            logger.info("Historical archive not found: %s", path)
            return None
        frame = None
        for value in filter_values:
            try:
                candidate = pd.read_parquet(path, filters=[("label_date", "==", value)])
            except Exception:
                continue
            if candidate is not None and not candidate.empty:
                frame = candidate
                break
        if frame is None or frame.empty:
            return None

    frame = frame.copy()
    if "label_date" not in frame.columns:
        logger.warning("Historical archive missing label_date column: %s", source)
        return None
    frame["label_date"] = pd.to_datetime(frame["label_date"]).dt.normalize()
    frame = frame.loc[frame["label_date"].eq(day_ts)].copy()
    if frame.empty:
        return None
    return frame.reset_index(drop=True)


def _materialize_day_parquet(
    frame: pd.DataFrame,
    label_date: date,
    cfg: dict,
    *,
    storage_client: Any | None = None,
) -> str:
    """Write local cache and optionally upload final_processed/D_test.parquet."""
    prefix = cfg["gcs"]["prefixes"]["final_processed"].rstrip("/")
    bucket_name = cfg["gcs"]["bucket"]
    object_name = f"{prefix}/{label_date.isoformat()}_test.parquet"
    object_uri = f"gs://{bucket_name}/{object_name}"

    cache = resolve_path(cfg, "local_cache") / "final_processed"
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / f"{label_date.isoformat()}_test.parquet"
    frame.to_parquet(local_path, index=False)
    logger.info("Cached historical day slice → %s (%d rows)", local_path, len(frame))

    if not _gcs_writes_allowed():
        return str(local_path)

    try:
        from preprocess.adapters_gcs import upload_parquet

        upload_parquet(frame, object_uri)
        logger.info("Uploaded historical day slice → %s", object_uri)
        return object_uri
    except Exception as exc:
        logger.warning(
            "Could not upload sliced day to GCS (%s); using local cache %s",
            exc,
            local_path,
        )
        return str(local_path)


def materialize_from_historical_archive(
    label_date: date,
    cfg: dict,
    *,
    storage_client: Any | None = None,
) -> dict | None:
    """Slice the 2019–2025 archive for one day, validate, cache, and optionally upload."""
    if not _in_historical_range(label_date):
        return None
    source = historical_archive_uri(cfg)
    if not source:
        return None

    logger.info(
        "Daily parquet missing for %s; slicing historical archive %s",
        label_date,
        source,
    )
    try:
        frame = _slice_historical_archive(source, label_date)
    except Exception as exc:
        logger.warning("Historical archive slice failed for %s: %s", label_date, exc)
        return None
    if frame is None or frame.empty:
        logger.info("Label date %s not present in historical archive", label_date)
        return None

    try:
        feature_cols = validate_champion_artifact(frame, label_date, cfg)
    except Exception as exc:
        logger.warning(
            "Historical slice for %s failed validation (%s); falling through to live pipeline",
            label_date,
            exc,
        )
        return None

    object_uri = _materialize_day_parquet(
        frame, label_date, cfg, storage_client=storage_client
    )
    return _artifact_dict(
        frame,
        label_date,
        cfg,
        object_uri=object_uri,
        feature_cols=feature_cols,
        created_at=datetime.now().astimezone(),
    )


def existing_final_artifact(
    label_date: date,
    cfg: dict,
    *,
    storage_client: Any | None = None,
) -> dict | None:
    """Return metadata for a valid prepared parquet, otherwise request a rebuild.

    Lookup order:
      1. GCS ``final_processed/YYYY-MM-DD_test.parquet``
      2. For 2019–2025 only: slice ``final_processed/2019_2025/2019-2025.parquet``

    Storage/authentication failures on the daily object are allowed to propagate;
    treating an unreachable bucket as an absent artifact would incorrectly start
    cloud downloads against the same broken dependency.
    """
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client(project=cfg["gcs"].get("project"))

    bucket_name = cfg["gcs"]["bucket"]
    prefix = cfg["gcs"]["prefixes"]["final_processed"].rstrip("/")
    object_name = f"{prefix}/{label_date.isoformat()}_test.parquet"
    blob = storage_client.bucket(bucket_name).blob(object_name)
    if blob.exists():
        blob.reload()
        if blob.size:
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
            else:
                created_at = blob.updated or datetime.now().astimezone()
                logger.info(
                    "Reusing validated prepared parquet: gs://%s/%s",
                    bucket_name,
                    object_name,
                )
                return _artifact_dict(
                    frame,
                    label_date,
                    cfg,
                    object_uri=f"gs://{bucket_name}/{object_name}",
                    feature_cols=feature_cols,
                    created_at=created_at,
                )
        else:
            logger.warning(
                "Prepared parquet is empty and will be rebuilt: gs://%s/%s",
                bucket_name,
                object_name,
            )
    else:
        logger.info("Prepared parquet is absent: gs://%s/%s", bucket_name, object_name)

    return materialize_from_historical_archive(
        label_date, cfg, storage_client=storage_client
    )
