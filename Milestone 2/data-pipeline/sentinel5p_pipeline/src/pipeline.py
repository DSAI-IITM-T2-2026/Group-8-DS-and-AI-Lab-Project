"""Top-level orchestration: loops the date windows and drives the run."""

import logging

from .composite import build_composite
from .config import get_active_products, get_aoi
from .dates import descending_date_steps
from .export import check_tasks, reconcile_progress, submit_export
from .progress import load_progress, save_progress, should_skip, upload_progress

logger = logging.getLogger("sentinel5p")


def run(cfg: dict) -> None:
    aoi = get_aoi(cfg)
    mode = cfg.get("logic_mode", "confirmed")
    products = get_active_products(cfg)
    logger.info(f"[sentinel5p:{mode}] Active products this run: {[p['name'] for p in products]}")

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

        composite, total_scenes = build_composite(start, end, aoi, cfg)

        if composite is None:
            logger.info(f"[sentinel5p:{mode}] {label}: no data (start={start}, end={end}) — marking empty")
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
            "scenes": total_scenes,
        }
        submitted += 1
        save_progress(progress_path, progress)

    upload_progress(progress_path, cfg)
    logger.info(
        f"\n[sentinel5p:{mode}] Done. {submitted} export tasks submitted this run "
        f"({retried} retries of prior failures), {skipped} skipped "
        f"(done/empty/in-flight), {empty} marked empty "
        f"(pre-2018 windows are expected to be empty — TROPOMI pre-launch). "
        f"Re-run later (or --check-tasks) to reconcile submitted -> done/failed. "
        f"Check status with: python main.py --check-tasks"
    )
    check_tasks(progress)
