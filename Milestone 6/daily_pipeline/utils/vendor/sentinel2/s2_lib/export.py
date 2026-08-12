"""Earth Engine helpers + FeatureCollection export to GCS."""

from __future__ import annotations

from typing import Any

import ee

from .config import AppConfig
from .grid import build_grid
from .reducers import OUTPUT_PROPERTIES, aggregate_to_grid
from .sentinel2 import TimeWindow

# EE task states
EE_READY = "READY"
EE_RUNNING = "RUNNING"
EE_COMPLETED = "COMPLETED"
EE_FAILED = "FAILED"
EE_CANCELLED = "CANCELLED"
EE_ACTIVE = frozenset({EE_READY, EE_RUNNING})

TRANSIENT_ERROR_MARKERS = (
    "quota",
    "rate limit",
    "rate exceeded",
    "user memory limit",
    "computed value is too large",
    "too large",
    "computation timed out",
    "timeout",
    "timed out",
    "unavailable",
    "temporarily",
    "connection",
    "network",
    "reset by peer",
    "503",
    "429",
    "internal error",
    "backend error",
    "server error",
    "try again",
    "concurrent",
    "cancelled",
)


def initialize(project_id: str) -> None:
    ee.Initialize(project=project_id)


def get_task_status(task_id: str) -> dict[str, Any] | None:
    try:
        status = ee.data.getTaskStatus(task_id)
    except ee.EEException:
        return None
    if isinstance(status, list):
        if not status:
            return None
        status = status[0]
    if not isinstance(status, dict):
        return None
    return status


def count_active_tasks() -> int:
    tasks = ee.data.getTaskList()
    return sum(
        1
        for t in tasks
        if str(t.get("state", "")).upper() in EE_ACTIVE
    )


def is_transient_error(message: str | None) -> bool:
    if not message:
        return True
    lower = message.lower()
    return any(m in lower for m in TRANSIENT_ERROR_MARKERS)


def export_prefix(cfg: AppConfig, window: TimeWindow) -> str:
    """sentinel2_features/year=YYYY/month=MM/window=NNN/features"""
    year = window.start_date.strftime("%Y")
    month = window.start_date.strftime("%m")
    window_num = f"{window.window_index:03d}"
    return "/".join(
        [
            cfg.export.prefix.strip("/"),
            f"year={year}",
            f"month={month}",
            f"window={window_num}",
            "features",
        ]
    )


def gcs_csv_uri(cfg: AppConfig, window: TimeWindow) -> str:
    return f"gs://{cfg.export.bucket}/{export_prefix(cfg, window)}.csv"


def gcs_parquet_uri(cfg: AppConfig, window: TimeWindow) -> str:
    return f"gs://{cfg.export.bucket}/{export_prefix(cfg, window)}.parquet"


def gcs_final_uri(cfg: AppConfig, window: TimeWindow) -> str:
    if cfg.export.format == "parquet":
        return gcs_parquet_uri(cfg, window)
    return gcs_csv_uri(cfg, window)


def create_export_task(cfg: AppConfig, window: TimeWindow) -> ee.batch.Task:
    """
    Create EE table export (CSV only — EE has no Parquet writer).

    Parquet is produced after the task completes when export.format=parquet.
    GeoTIFF is never exported. One task = one full-AOI window file.
    """
    grid = build_grid(cfg)
    features = aggregate_to_grid(cfg, window, grid)
    prefix = export_prefix(cfg, window)
    description = (
        f"s2feat_{window.start_date.strftime('%Y%m%d')}_"
        f"{window.end_date.strftime('%Y%m%d')}"
    )[:100]

    return ee.batch.Export.table.toCloudStorage(
        collection=features,
        description=description,
        bucket=cfg.export.bucket,
        fileNamePrefix=prefix,
        fileFormat=cfg.export.gee_format,
        selectors=OUTPUT_PROPERTIES,
    )


def create_grid_export_task(cfg: AppConfig) -> ee.batch.Task:
    """Create the reusable exact-AOI grid asset used by every window."""
    if not cfg.grid.asset_id:
        raise ValueError("grid.asset_id must be configured before exporting it")
    grid = build_grid(cfg, use_asset=False).select(
        ["grid_id", "ix", "iy", "latitude", "longitude"]
    )
    return ee.batch.Export.table.toAsset(
        collection=grid,
        description="california_1km_grid_v3",
        assetId=cfg.grid.asset_id,
    )


def start_export(cfg: AppConfig, window: TimeWindow) -> tuple[str, str]:
    task = create_export_task(cfg, window)
    task.start()
    task_id = task.id
    if not task_id:
        status = task.status()
        task_id = str(status.get("id") or "")
    if not task_id:
        raise RuntimeError(f"No task id for window {window.window_id}")
    return task_id, gcs_final_uri(cfg, window)


def convert_csv_to_parquet(cfg: AppConfig, window: TimeWindow) -> int:
    """Convert EE CSV → Parquet through local staging. Returns row count.

    Reading and writing DataFrames directly through ``gs://`` can turn a
    sequential 250+ MB transfer into many slow remote range requests.  Stage
    both files locally, validate Parquet metadata, upload the final object, and
    only then remove the source CSV.
    """
    if cfg.export.format != "parquet":
        return -1

    import tempfile
    from pathlib import Path

    import gcsfs
    import pandas as pd
    import pyarrow.parquet as pq

    csv_uri = gcs_csv_uri(cfg, window)
    parquet_uri = gcs_parquet_uri(cfg, window)
    csv_path = csv_uri.removeprefix("gs://")
    parquet_path = parquet_uri.removeprefix("gs://")
    fs = gcsfs.GCSFileSystem()

    with tempfile.TemporaryDirectory(prefix="s2_parquet_") as tmp:
        local_csv = Path(tmp) / "features.csv"
        local_parquet = Path(tmp) / "features.parquet"

        print(f"Downloading {csv_uri} ({fs.size(csv_path):,} bytes)", flush=True)
        fs.get(csv_path, str(local_csv))

        df = pd.read_csv(local_csv, low_memory=False)
        cols = [c for c in OUTPUT_PROPERTIES if c in df.columns]
        extras = [c for c in df.columns if c not in cols]
        df = df[cols + extras]
        rows = int(len(df))
        df.to_parquet(local_parquet, index=False)

        metadata_rows = int(pq.ParquetFile(local_parquet).metadata.num_rows)
        if metadata_rows != rows:
            raise RuntimeError(
                f"Parquet row-count mismatch: dataframe={rows}, metadata={metadata_rows}"
            )

        print(
            f"Uploading {parquet_uri} ({local_parquet.stat().st_size:,} bytes)",
            flush=True,
        )
        fs.put(str(local_parquet), parquet_path)
        if not fs.exists(parquet_path) or fs.size(parquet_path) <= 0:
            raise RuntimeError(f"Parquet upload verification failed: {parquet_uri}")

    # The source is deleted only after local metadata and remote-object checks.
    try:
        if fs.exists(csv_path):
            fs.rm(csv_path)
    except Exception:
        pass

    return rows
