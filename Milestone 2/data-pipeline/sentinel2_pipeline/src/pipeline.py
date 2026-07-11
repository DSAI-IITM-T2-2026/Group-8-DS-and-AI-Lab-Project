"""Top-level orchestration: loops the date windows and drives the run."""

import logging

from .composite import build_composite
from .config import get_aoi
from .dates import descending_date_steps
from .export import check_tasks, reconcile_progress, submit_export
from .progress import load_progress, save_progress, should_skip, upload_progress

logger = logging.getLogger("sentinel2")


def run(cfg: dict) -> None:
    aoi = get_aoi(cfg)
    mode = cfg.get("logic_mode", "confirmed")
    progress_path = cfg["progress"]["local_file"]
    progress = load_progress(progress_path)
    progress = reconcile_progress(progress)
    save_progress(progress_path, progress)

    submitted, skipped, empty, retried = 0, 0, 0, 0

    for start, end, label in descending_date_steps(cfg):
        key = f"{mode}:{label}"
        prior = progress.get(key)
        if should_skip(prior):
            skipped += 1
            continue
        if prior and prior.get("status") == "failed":
            retried += 1

        composite, count = build_composite(start, end, aoi, cfg)

        if count == 0:
            logger.info(f"[sentinel2:{mode}] {label}: no scenes (start={start}, end={end}) — marking empty")
            progress[key] = {"status": "empty", "start": start, "end": end, "mode": mode}
            empty += 1
            save_progress(progress_path, progress)
            continue

        description = submit_export(composite, label, cfg)
        progress[key] = {
            "status": "submitted",
            "start": start,
            "end": end,
            "mode": mode,
            "task": description,
        }
        submitted += 1
        save_progress(progress_path, progress)

    upload_progress(progress_path, cfg)
    logger.info(
        f"\n[sentinel2:{mode}] Done. {submitted} export tasks submitted this run "
        f"({retried} retries of prior failures), {skipped} skipped "
        f"(done/empty/in-flight), {empty} marked empty. "
        f"Re-run later (or --check-tasks) to reconcile submitted -> done/failed. "
        f"Check status with: python main.py --check-tasks"
    )
    check_tasks(progress)
