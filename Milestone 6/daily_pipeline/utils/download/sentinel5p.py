#!/usr/bin/env python3
"""Submit Sentinel-5P daily export and flatten to s5pfeat_YYYYMMDD_YYYYMMDD.parquet."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger("download.sentinel5p")


def _gcs_listing():
    try:
        from download import gcs_listing as gl
    except ImportError:
        import gcs_listing as gl  # type: ignore
    return gl

S5P_DIR = Path(__file__).resolve().parents[1] / "vendor" / "sentinel5p"
TASK_REGISTRY = Path(__file__).resolve().parents[2] / ".cache" / "s5p_ee_tasks.json"

_PENDING_STATES = {"READY", "RUNNING", "CANCEL_REQUESTED"}
_DONE_OK = {"COMPLETED"}
_TERMINAL_BAD = {"FAILED", "CANCELLED"}


def _init_ee(project_id: str) -> None:
    try:
        from download.ee_init import initialize_ee
    except ImportError:
        from ee_init import initialize_ee  # type: ignore

    initialize_ee(project_id)


def _download_gcs_to_file(uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rest = uri[5:]
    bucket_name, _, blob_name = rest.partition("/")
    from google.cloud import storage

    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))


def hive_prefix(day: date, gcs_prefix: str) -> str:
    window_number = day.timetuple().tm_yday
    return (
        f"{gcs_prefix.rstrip('/')}/year={day.year:04d}/month={day.month:02d}/"
        f"window={window_number:03d}/features"
    )


def flat_stem(day: date) -> str:
    stamp = day.strftime("%Y%m%d")
    return f"s5pfeat_{stamp}_{stamp}"


def task_description(day: date) -> str:
    return f"s5pfeat_{day:%Y%m%d}_{day:%Y%m%d}"


def blob_exists(bucket: str, blob_name: str, project: str | None = None) -> bool:
    from google.cloud import storage

    client = storage.Client(project=project) if project else storage.Client()
    return client.bucket(bucket).blob(blob_name).exists()


def upload_parquet(df, bucket: str, blob_name: str, project: str | None = None) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import storage

    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    client = storage.Client(project=project) if project else storage.Client()
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_string(buf.getvalue(), content_type="application/octet-stream")
    _gcs_listing().remember(bucket, blob_name)
    return f"gs://{bucket}/{blob_name}"


def _csv_to_parquet_df(local_csv: Path):
    import pandas as pd

    df = pd.read_csv(local_csv)
    if "window_start" in df.columns:
        df["window_start"] = pd.to_datetime(df["window_start"]).dt.normalize()
    if "window_end" in df.columns:
        df["window_end"] = pd.to_datetime(df["window_end"]).dt.normalize()
    return df


def _load_task_registry() -> dict:
    if not TASK_REGISTRY.exists():
        return {}
    try:
        return json.loads(TASK_REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_task_registry(reg: dict) -> None:
    TASK_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    TASK_REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _remember_task(day: date, task_id: str) -> None:
    reg = _load_task_registry()
    reg[day.isoformat()] = {
        "task_id": task_id,
        "description": task_description(day),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save_task_registry(reg)


def clear_task_registry_day(day: date) -> None:
    reg = _load_task_registry()
    if day.isoformat() in reg:
        reg.pop(day.isoformat(), None)
        _save_task_registry(reg)


def _ee_task_status(task_id: str) -> dict | None:
    import ee

    try:
        rows = ee.data.getTaskStatus(task_id)
        if isinstance(rows, list) and rows:
            return rows[0]
        if isinstance(rows, dict):
            return rows
    except Exception as exc:
        logger.debug("getTaskStatus(%s) failed: %s", task_id, exc)
    return None


def _find_ee_task_for_day(day: date) -> tuple[str | None, str | None]:
    """Return (task_id, state). Prefer pending/completed; also report CANCELLED/FAILED."""
    import ee

    desc = task_description(day)
    key = day.isoformat()
    found_bad: tuple[str | None, str | None] = (None, None)

    reg = _load_task_registry()
    if key in reg and reg[key].get("task_id"):
        tid = reg[key]["task_id"]
        st = _ee_task_status(tid)
        if st:
            state = str(st.get("state", "")).upper()
            if state in _PENDING_STATES or state in _DONE_OK:
                return tid, state
            if state in _TERMINAL_BAD:
                found_bad = (tid, state)

    # Scan recent EE tasks so busy accounts still see ours.
    try:
        try:
            listed = ee.batch.Task.list(count=500)
        except TypeError:
            listed = ee.batch.Task.list()
        for task in listed:
            st = task.status()
            if st.get("description") != desc:
                continue
            state = str(st.get("state", "")).upper()
            tid = st.get("id") or getattr(task, "id", None)
            if state in _PENDING_STATES or state in _DONE_OK:
                if tid:
                    _remember_task(day, tid)
                return tid, state
            if state in _TERMINAL_BAD and found_bad[0] is None:
                found_bad = (tid, state)
    except Exception as exc:
        logger.warning("Could not list EE tasks: %s", exc)

    return found_bad


def convert_hive_csv_to_flat_parquet(
    day: date,
    *,
    source_bucket: str,
    source_prefix: str,
    dest_bucket: str,
    dest_prefix: str,
    project: str | None = None,
) -> str:
    hive_blob = f"{hive_prefix(day, source_prefix)}.csv"
    flat_blob = f"{dest_prefix.rstrip('/')}/{flat_stem(day)}.parquet"
    if _gcs_listing().blob_listed(dest_bucket, flat_blob, prefix=dest_prefix, project=project):
        uri = f"gs://{dest_bucket}/{flat_blob}"
        logger.info("Flat S5P already exists: %s", uri)
        return uri

    with tempfile.TemporaryDirectory() as tmp:
        local_csv = Path(tmp) / "features.csv"
        uri = f"gs://{source_bucket}/{hive_blob}"
        logger.info("Downloading %s", uri)
        _download_gcs_to_file(uri, local_csv)
        df = _csv_to_parquet_df(local_csv)
        out_uri = upload_parquet(df, dest_bucket, flat_blob, project)
        logger.info("Uploaded flat S5P: %s", out_uri)
        return out_uri


def _flatten_dest_csv_if_present(
    day: date,
    *,
    dest_bucket: str,
    dest_prefix: str,
    project: str | None = None,
) -> str | None:
    flat_blob = f"{dest_prefix.rstrip('/')}/{flat_stem(day)}.parquet"
    if blob_exists(dest_bucket, flat_blob, project):
        return f"gs://{dest_bucket}/{flat_blob}"

    from google.cloud import storage

    client = storage.Client(project=project) if project else storage.Client()
    prefix = f"{dest_prefix.rstrip('/')}/{flat_stem(day)}"
    candidates = [
        f"{prefix}.csv",
        f"{prefix}-00000.csv",
        f"{prefix}/features.csv",
    ]
    blobs = list(client.bucket(dest_bucket).list_blobs(prefix=prefix, max_results=50))
    for b in blobs:
        if b.name.endswith(".csv"):
            candidates.append(b.name)

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        if not blob_exists(dest_bucket, name, project):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            local_csv = Path(tmp) / "features.csv"
            uri = f"gs://{dest_bucket}/{name}"
            logger.info("Flattening S5P CSV %s → parquet", uri)
            _download_gcs_to_file(uri, local_csv)
            df = _csv_to_parquet_df(local_csv)
            out = upload_parquet(df, dest_bucket, flat_blob, project)
            clear_task_registry_day(day)
            return out
    return None


def submit_s5p_day(
    target: date,
    *,
    project_id: str = "plated-mechanic-418917",
    source_bucket: str = "plated-mechanic-s5p-2016-2025",
    source_prefix: str = "sentinel5p_features_daily",
    dest_bucket: str = "wildfire-detection-first",
    dest_prefix: str = "sentinel5p",
    skip_existing: bool = True,
    force: bool = False,
) -> str | None:
    """Export one calendar day. Never duplicates pending/completed tasks unless force=True."""
    flat_blob = f"{dest_prefix.rstrip('/')}/{flat_stem(target)}.parquet"
    if skip_existing and _gcs_listing().blob_listed(
        dest_bucket, flat_blob, prefix=dest_prefix, project=project_id
    ):
        uri = f"gs://{dest_bucket}/{flat_blob}"
        logger.info("S5P already exists: %s", uri)
        return uri

    flattened = _flatten_dest_csv_if_present(
        target, dest_bucket=dest_bucket, dest_prefix=dest_prefix, project=project_id
    )
    if flattened:
        logger.info("S5P flattened from existing CSV: %s", flattened)
        return flattened

    hive_blob = f"{hive_prefix(target, source_prefix)}.csv"
    if blob_exists(source_bucket, hive_blob, project_id):
        return convert_hive_csv_to_flat_parquet(
            target,
            source_bucket=source_bucket,
            source_prefix=source_prefix,
            dest_bucket=dest_bucket,
            dest_prefix=dest_prefix,
            project=project_id,
        )

    _add_s5p_path()
    import ee  # type: ignore

    _init_ee(project_id)

    existing_id, existing_state = _find_ee_task_for_day(target)

    if existing_id and existing_state in _PENDING_STATES:
        logger.info(
            "S5P EE task already %s for %s (task=%s) — not re-submitting.",
            existing_state,
            target.isoformat(),
            existing_id,
        )
        return None

    if existing_id and existing_state in _DONE_OK:
        flattened = _flatten_dest_csv_if_present(
            target, dest_bucket=dest_bucket, dest_prefix=dest_prefix, project=project_id
        )
        if flattened:
            return flattened
        logger.warning(
            "S5P EE task COMPLETED for %s (task=%s) but CSV not visible yet — not re-submitting.",
            target.isoformat(),
            existing_id,
        )
        return None

    if existing_id and existing_state in _TERMINAL_BAD and not force:
        logger.error(
            "S5P EE task %s for %s (task=%s). Not auto-resubmitting. "
            "Re-run with --force-s5p (or download.s5p_force: true) after fixing EE, "
            "or delete .cache/s5p_ee_tasks.json entry for this day.",
            existing_state,
            target.isoformat(),
            existing_id,
        )
        return None

    if force:
        clear_task_registry_day(target)
        logger.info("force-s5p: cleared local registry for %s; submitting new EE export", target)

    from s5p_lib import aggregate_day, build_grid  # type: ignore

    grid = build_grid()
    fc = aggregate_day(target, grid)
    prefix = f"{dest_prefix.rstrip('/')}/{flat_stem(target)}"
    desc = task_description(target)
    task = ee.batch.Export.table.toCloudStorage(
        collection=fc,
        description=desc,
        bucket=dest_bucket,
        fileNamePrefix=prefix,
        fileFormat="CSV",
    )
    task.start()
    _remember_task(target, task.id)
    logger.info(
        "Started S5P EE export for %s → gs://%s/%s*.csv (task=%s).",
        target.isoformat(),
        dest_bucket,
        prefix,
        task.id,
    )
    return None


def _add_s5p_path() -> None:
    path = str(S5P_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def download_s5p_for_date(
    target: date,
    *,
    project_id: str = "plated-mechanic-418917",
    dest_bucket: str = "wildfire-detection-first",
    dest_prefix: str = "sentinel5p",
    skip_existing: bool = True,
    force: bool = False,
) -> str | None:
    return submit_s5p_day(
        target,
        project_id=project_id,
        dest_bucket=dest_bucket,
        dest_prefix=dest_prefix,
        skip_existing=skip_existing,
        force=force,
    )


def wait_for_s5p_days(
    days: list[date],
    *,
    project_id: str = "plated-mechanic-418917",
    dest_bucket: str = "wildfire-detection-first",
    dest_prefix: str = "sentinel5p",
    poll_seconds: int = 60,
    timeout_seconds: int = 6 * 3600,
) -> dict[date, str]:
    """Poll until every day has flat parquet. Does not start new EE tasks."""
    import time

    _init_ee(project_id)

    pending = list(days)
    done: dict[date, str] = {}
    t0 = time.time()

    while pending:
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for S5P after {timeout_seconds}s. Still pending: "
                + ", ".join(d.isoformat() for d in pending)
            )

        still: list[date] = []
        n_ready = n_running = n_unknown = 0
        cancelled: list[str] = []

        for day in pending:
            flat_blob = f"{dest_prefix.rstrip('/')}/{flat_stem(day)}.parquet"
            if blob_exists(dest_bucket, flat_blob, project_id):
                done[day] = f"gs://{dest_bucket}/{flat_blob}"
                logger.info("S5P ready: %s", done[day])
                continue

            flattened = _flatten_dest_csv_if_present(
                day, dest_bucket=dest_bucket, dest_prefix=dest_prefix, project=project_id
            )
            if flattened:
                done[day] = flattened
                logger.info("S5P ready (flattened): %s", flattened)
                continue

            tid, state = _find_ee_task_for_day(day)
            if state in _TERMINAL_BAD:
                cancelled.append(f"{day.isoformat()}={state}({tid})")
                continue
            if state == "READY":
                n_ready += 1
            elif state == "RUNNING":
                n_running += 1
            else:
                n_unknown += 1
            still.append(day)

        if cancelled:
            raise RuntimeError(
                "S5P EE task(s) CANCELLED/FAILED with no GCS output: "
                + ", ".join(cancelled)
                + ". Re-run with --force-s5p to resubmit after checking EE console."
            )

        pending = still
        if pending:
            logger.info(
                "S5P wait: %d pending (%d READY, %d RUNNING, %d other) — sleep %ss",
                len(pending),
                n_ready,
                n_running,
                n_unknown,
                max(15, poll_seconds),
            )
            time.sleep(max(15, poll_seconds))

    return done


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="S5P daily export + flat parquet.")
    parser.add_argument("--date", type=parse_date, required=True)
    parser.add_argument("--project", default="plated-mechanic-418917")
    parser.add_argument("--dest-bucket", default="wildfire-detection-first")
    parser.add_argument("--dest-prefix", default="sentinel5p")
    parser.add_argument("--flatten-only", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Resubmit after CANCELLED/FAILED.")
    args = parser.parse_args()
    if args.flatten_only:
        convert_hive_csv_to_flat_parquet(
            args.date,
            source_bucket="plated-mechanic-s5p-2016-2025",
            source_prefix="sentinel5p_features_daily",
            dest_bucket=args.dest_bucket,
            dest_prefix=args.dest_prefix,
            project=args.project,
        )
    else:
        download_s5p_for_date(
            args.date,
            project_id=args.project,
            dest_bucket=args.dest_bucket,
            dest_prefix=args.dest_prefix,
            skip_existing=not args.no_skip_existing,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
