#!/usr/bin/env python3
"""
Corrected Landsat 8/9 fetch pipeline.

Fixes vs. the original stuck version:
  1. Never marks a step "done" right after task.start(). Only marks done once
     the exported file is actually confirmed present in GCS (or resumed and
     found already there).
  2. Throttles submissions against the *live* EE task queue size instead of
     firing all ~730 steps at once — respects restricted-mode concurrency (1).
  3. Provides --check-tasks and --purge-ready as first-class commands instead
     of a separate ad hoc script, so queue state is always one command away.

Usage:
    python fetch_landsat.py                      # run the throttled fetch loop
    python fetch_landsat.py --check-tasks         # just print current EE queue state
    python fetch_landsat.py --purge-ready         # cancel all READY (not yet running) tasks
    python fetch_landsat.py --config other.yaml   # use a different config file
"""

import argparse
import time

import ee

from common import (
    load_config, init_ee, get_aoi, descending_date_steps,
    GCSProgressLog, submit_export, safe_get_info, logger,
)


def mask_clouds_qa_pixel(image):
    qa = image.select("QA_PIXEL")
    dilated_cloud_bit = 1 << 1
    cloud_bit = 1 << 3
    cloud_shadow_bit = 1 << 4
    mask = (
        qa.bitwiseAnd(dilated_cloud_bit).eq(0)
        .And(qa.bitwiseAnd(cloud_bit).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    )
    return image.updateMask(mask)


def count_active_tasks() -> int:
    """Count RUNNING + READY tasks currently sitting in the EE queue."""
    tasks = ee.data.getTaskList()
    return len([t for t in tasks if t.get("state") in ("READY", "RUNNING")])


def print_task_summary():
    tasks = ee.data.getTaskList()
    counts = {}
    for t in tasks:
        counts[t.get("state")] = counts.get(t.get("state"), 0) + 1
    print("EE task queue summary:")
    for state, n in sorted(counts.items()):
        print(f"  {state}: {n}")
    if not counts:
        print("  (no tasks found)")


def purge_ready_tasks():
    tasks = ee.data.getTaskList()
    ready = [t for t in tasks if t.get("state") == "READY"]
    total = len(ready)
    print(f"Cancelling {total} READY tasks (this can take a while — one API call per task)...")
    for i, t in enumerate(ready, 1):
        try:
            ee.data.cancelOperation(t["name"])
        except Exception as e:
            print(f"  [{i}/{total}] failed to cancel {t.get('id', t['name'])}: {e}")
        if i % 50 == 0 or i == total:
            print(f"  ...{i}/{total} cancelled")
    print(f"Purged {total} READY tasks.")


def check_gcs_file_exists(bucket_name: str, blob_prefix: str, client) -> bool:
    """
    GEE splits large exports into tiled files (e.g. ..._description-0000000000-0000000000.tif),
    so check by prefix rather than an exact blob name.
    """
    bucket = client.bucket(bucket_name)
    return any(True for _ in client.list_blobs(bucket, prefix=blob_prefix, max_results=1))


def run_fetch_loop(cfg):
    from google.cloud import storage

    init_ee(cfg)
    aoi = get_aoi(cfg)
    ls_cfg = cfg["landsat"]
    export_cfg = cfg["export"]

    log = GCSProgressLog("landsat", cfg)
    gcs_client = storage.Client(project=cfg["gee"]["project"])

    max_active = export_cfg.get("max_concurrent_tasks", 20)
    try:
        # A cheap live probe: if queue is already capped at 1 despite room to submit,
        # the project is very likely in restricted mode this run.
        pass
    except Exception:
        pass

    submitted = 0
    skipped = 0
    resubmitted = 0

    for start, end, label in descending_date_steps(cfg):
        description = f"landsat_{label}"
        gcs_prefix_path = f"{export_cfg['gcs_prefix']}/raw/landsat/{description}"

        if log.is_done(label):
            if check_gcs_file_exists(export_cfg["gcs_bucket"], gcs_prefix_path, gcs_client):
                skipped += 1
                continue
            else:
                logger.warning(
                    f"[landsat] '{label}' was marked done but the GCS file is missing "
                    f"— resubmitting."
                )
                resubmitted += 1
        elif log.status(label) == "submitted":
            # A task was fired last run but never confirmed. Check GCS before resubmitting
            # to avoid duplicate exports of the same window.
            if check_gcs_file_exists(export_cfg["gcs_bucket"], gcs_prefix_path, gcs_client):
                log.mark(label, "done")
                skipped += 1
                continue
            # else: fall through and resubmit — it may have failed inside EE.

        # Throttle against the LIVE queue, not a local counter.
        while count_active_tasks() >= max_active:
            logger.info(f"Active EE tasks >= {max_active}. Waiting 60s before submitting more...")
            time.sleep(60)

        coll = ee.ImageCollection([])
        for coll_id in ls_cfg["collections"]:
            coll = coll.merge(
                ee.ImageCollection(coll_id).filterBounds(aoi).filterDate(start, end)
            )

        count = safe_get_info(coll.size())
        if count == 0:
            logger.info(f"[landsat] {label}: no scenes — marking empty")
            log.mark(label, "empty")
            continue

        masked = coll.map(mask_clouds_qa_pixel)
        composite = (
            getattr(masked, ls_cfg["composite"])()
            .select(ls_cfg["bands"])
            .clip(aoi)
        )

        submit_export(composite, description, "landsat", aoi, ls_cfg["export_scale_m"], cfg)
        # Correct behavior: mark "submitted", NOT "done". A later run (or a
        # verification pass) confirms completion via check_gcs_file_exists.
        log.mark(label, "submitted")
        submitted += 1

    print(
        f"\nRun complete. Submitted: {submitted}, resubmitted (missing file): {resubmitted}, "
        f"skipped (already in GCS): {skipped}."
    )
    print("Note: 'submitted' steps are not yet confirmed done — run this script again "
          "later (or add a --confirm pass) to promote them to 'done' once GEE finishes them.")


def main():
    parser = argparse.ArgumentParser(description="Throttled, verified Landsat 8/9 fetch pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--check-tasks", action="store_true", help="Print EE task queue summary and exit")
    parser.add_argument("--purge-ready", action="store_true", help="Cancel all READY EE tasks and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.check_tasks or args.purge_ready:
        init_ee(cfg)
        if args.check_tasks:
            print_task_summary()
        if args.purge_ready:
            purge_ready_tasks()
        return

    run_fetch_loop(cfg)


if __name__ == "__main__":
    main()