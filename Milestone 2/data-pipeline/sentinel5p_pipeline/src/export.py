"""GEE -> GCS export submission, task-status checking, and progress reconcile."""

import logging
from typing import Dict, Optional

import ee

from .config import get_aoi

logger = logging.getLogger("sentinel5p")

_ACTIVE = frozenset({"READY", "RUNNING"})
_SUCCESS = frozenset({"COMPLETED"})
_FAILED = frozenset({"FAILED", "CANCELLED"})


def submit_export(image, label: str, cfg: dict) -> str:
    mode = cfg.get("logic_mode", "confirmed")
    tag = "" if mode == "confirmed" else "ALT_DESIGN_UNVERIFIED_"
    description = f"{tag}s5p_{label}"
    exp = cfg["export"]
    prefix = f"{exp['gcs_prefix']}/raw/sentinel5p/{description}"

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        bucket=exp["gcs_bucket"],
        fileNamePrefix=prefix,
        region=get_aoi(cfg),
        scale=cfg["sentinel5p"]["export_scale_m"],
        maxPixels=exp["max_pixels"],
        fileFormat="GeoTIFF",
        crs="EPSG:4326",
    )
    task.start()
    logger.info(
        f"[sentinel5p:{mode}] Export task submitted: {description} -> "
        f"gs://{exp['gcs_bucket']}/{prefix}.tif"
    )
    return description


def _tasks_by_description() -> Dict[str, object]:
    """Map task description -> most recent Task object."""
    by_desc: Dict[str, object] = {}
    for task in ee.batch.Task.list():
        status = task.status()
        desc = status.get("description")
        if desc and desc not in by_desc:
            by_desc[desc] = task
    return by_desc


def reconcile_progress(progress: dict) -> dict:
    """
    Update local progress entries that are still 'submitted' based on live
    GEE task state. COMPLETED -> done, FAILED/CANCELLED -> failed.
    """
    if not any(e.get("status") == "submitted" for e in progress.values()):
        return progress

    by_desc = _tasks_by_description()
    updated = 0
    for entry in progress.values():
        if entry.get("status") != "submitted":
            continue
        desc = entry.get("task")
        task = by_desc.get(desc) if desc else None
        if task is None:
            logger.warning(
                f"Submitted task not found in GEE task list (leaving as submitted): {desc}"
            )
            continue
        state = task.state
        if state in _SUCCESS:
            entry["status"] = "done"
            entry.pop("error", None)
            updated += 1
        elif state in _FAILED:
            status = task.status()
            entry["status"] = "failed"
            entry["error"] = status.get("error_message") or state
            updated += 1
            logger.warning(f"Export failed ({desc}): {entry['error']}")
    if updated:
        logger.info(f"Reconciled {updated} submitted task(s) from GEE status.")
    return progress


def check_tasks(progress: Optional[dict] = None) -> None:
    tasks = ee.batch.Task.list()
    active = [t for t in tasks if t.state in _ACTIVE]
    failed = [t for t in tasks if t.state in _FAILED]

    logger.info(f"{len(active)} active task(s) in this GEE account:")
    for t in active[:50]:
        logger.info(f"  {t.status().get('description')}: {t.state}")

    if failed:
        logger.info(f"{len(failed)} recent failed/cancelled task(s) (showing up to 20):")
        for t in failed[:20]:
            status = t.status()
            logger.info(
                f"  {status.get('description')}: {t.state} "
                f"— {status.get('error_message') or ''}"
            )

    if progress is not None:
        submitted = sum(1 for e in progress.values() if e.get("status") == "submitted")
        done = sum(1 for e in progress.values() if e.get("status") == "done")
        empty = sum(1 for e in progress.values() if e.get("status") == "empty")
        failed_local = sum(1 for e in progress.values() if e.get("status") == "failed")
        logger.info(
            f"Local progress: {done} done, {submitted} submitted, "
            f"{failed_local} failed, {empty} empty."
        )
