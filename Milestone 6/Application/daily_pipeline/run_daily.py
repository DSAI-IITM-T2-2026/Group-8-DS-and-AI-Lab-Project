#!/usr/bin/env python3
"""Single CLI: download | preprocess | export_day | all.

Date conventions (Milestone 4 / live prediction):
  label_date D          → day we predict / write *_test.parquet
  eo_asof_date          → D − 1  (S2 / S5P causal join)
  feature_end_date      → D − (era5_lag + lead) = D − 6  (ERA5; 7d history ending there)
  FIRMS neighbor history → through D − 1 (y_fire on D is not a model input; export uses 0)
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_UTILS = _ROOT / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from bootstrap import bootstrap  # noqa: E402

bootstrap()

from config_loader import load_daily_config, load_m4_config, pipeline_today  # noqa: E402
from download.dem import publish_dem  # noqa: E402
from download.era5 import download_era5_day  # noqa: E402
from download.firms import export_firms_day, wait_for_firms_days  # noqa: E402
from download.sentinel2 import download_s2_for_date, wait_for_s2_days  # noqa: E402
from download.sentinel5p import download_s5p_for_date, wait_for_s5p_days  # noqa: E402
from pipeline_events import emit_event  # noqa: E402
from preprocess.build_stage_c_day import run_stage_c_pipeline  # noqa: E402
from preprocess.export_inference_day import export_champion_day  # noqa: E402
from preprocess.final_artifact import existing_final_artifact  # noqa: E402

logger = logging.getLogger("run_daily")


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def eo_asof_date(label_date: date) -> date:
    """M4: S2/S5P join on label_date − 1."""
    return label_date - timedelta(days=1)


def era5_feature_end(label_date: date, era5_lag: int = 5, lead: int = 1) -> date:
    """M4: feature_end_date = label − (lag + lead)."""
    return label_date - timedelta(days=era5_lag + lead)


# Back-compat alias used in logs / older call sites → means eo_asof (D−1), not D−7.
def decision_date_from_label(label_date: date, era5_lag: int = 5, lead: int = 1) -> date:
    """Map label_date → eo_asof_date (D−1). era5_lag/lead unused; kept for call-site compat."""
    _ = (era5_lag, lead)
    return eo_asof_date(label_date)


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end-date {end} is before start-date {start}")
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def resolve_label_dates(args: argparse.Namespace) -> list[date]:
    """Prefer --start-date/--end-date; else single --label-date / --date."""
    start = getattr(args, "start_date", None)
    end = getattr(args, "end_date", None)
    if start or end:
        if not (start and end):
            raise SystemExit("Provide both --start-date and --end-date for a range.")
        return date_range(start, end)
    single = getattr(args, "label_date", None) or getattr(args, "date", None)
    if single is None:
        raise SystemExit("Provide --label-date YYYY-MM-DD or --start-date/--end-date.")
    return [single]


def label_window(labels: list[date], cfg: dict) -> list[date]:
    lookback = int(cfg["task"].get("lookback_days", 30))
    label_start = min(labels) - timedelta(days=lookback)
    label_end = max(labels)
    return date_range(label_start, label_end)


def eo_asof_dates_needed(labels: list[date], cfg: dict, *, as_of: date | None = None) -> list[date]:
    """Unique eo_asof (= label−1) days for S2/S5P; never past as_of (default today)."""
    as_of = as_of or pipeline_today(cfg)
    days = sorted({eo_asof_date(d) for d in label_window(labels, cfg)})
    return [d for d in days if d <= as_of]


def firms_label_dates_needed(labels: list[date], cfg: dict, *, as_of: date | None = None) -> list[date]:
    """FIRMS geotiffs for neighbor fire history through D−1 (not predict-day D)."""
    as_of = as_of or pipeline_today(cfg)
    lookback = int(cfg["task"].get("lookback_days", 30))
    # Neighbor features use lag2 → need history through min(D−1, as_of), not D.
    history_end = min(max(labels) - timedelta(days=1), as_of)
    history_start = min(labels) - timedelta(days=lookback)
    if history_end < history_start:
        return []
    return date_range(history_start, history_end)


def era5_days_needed(labels: list[date], cfg: dict, *, as_of: date | None = None) -> list[date]:
    """ERA5 calendar days ending at each label's feature_end (D−6); never past as_of−6."""
    as_of = as_of or pipeline_today(cfg)
    lookback = int(cfg["task"].get("lookback_days", 30))
    history = int(cfg["task"].get("history_days", 7))
    lag = int(cfg["task"]["era5_lag_days"])
    lead = int(cfg["task"]["lead_days"])
    label_start = min(labels) - timedelta(days=lookback)
    label_end = min(max(labels), as_of)
    era5_end = min(label_end - timedelta(days=lag + lead), as_of - timedelta(days=lag + lead))
    era5_start = label_start - timedelta(days=history + lag + lead)
    if era5_end < era5_start:
        return []
    return date_range(era5_start, era5_end)


def _fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {s:.0f}s"


def _month_blob_exists(bucket: str, prefix: str, day: date) -> bool:
    """True if a monthly ERA5 object exists for this calendar month."""
    from download.gcs_listing import blob_listed

    stem = f"era5_{day.year}_{day.month:02d}.nc"
    base = prefix.rstrip("/")
    year = day.year
    for name, pfx in (
        (f"{base}/{year}/{stem}", f"{base}/{year}"),
        (f"{base}/raw/{year}/{stem}", f"{base}/raw/{year}"),
    ):
        if blob_listed(bucket, name, prefix=pfx):
            return True
    return False


def _era5_daily_exists(bucket: str, prefix: str, day: date) -> bool:
    from download.gcs_listing import blob_listed, era5_daily_blob

    name = era5_daily_blob(prefix, day)
    return blob_listed(bucket, name, prefix=f"{prefix.rstrip('/')}/{day.year:04d}")


def build_source_inventory(labels: list[date], cfg: dict, *, as_of: date) -> dict[str, dict[str, int]]:
    """Summarize raw-object readiness after the GCS prefix prefetch."""
    from download.gcs_listing import blob_listed
    from download.sentinel2 import flat_stem as s2_stem, windows_overlapping
    from download.sentinel5p import flat_stem as s5p_stem

    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    firms_days = firms_label_dates_needed(labels, cfg, as_of=as_of)
    eo_days = eo_asof_dates_needed(labels, cfg, as_of=as_of)
    era5_days = era5_days_needed(labels, cfg, as_of=as_of)
    s2_windows = sorted(
        {window for day in eo_days for window in windows_overlapping(day, cfg["download"].get("s2_window_days", 5))}
    )

    era5_ready = sum(
        1
        for day in era5_days
        if _month_covers_day(bucket, prefixes["era5"], day, as_of=as_of)
        or _era5_daily_exists(bucket, prefixes["era5"], day)
    )
    firms_ready = sum(
        1
        for day in firms_days
        if blob_listed(
            bucket,
            f"{prefixes['firms'].rstrip('/')}/{day.isoformat()}.tif",
            prefix=prefixes["firms"],
            project=cfg["gee"]["project_id"],
        )
    )
    s2_ready = sum(
        1
        for start, end in s2_windows
        if blob_listed(
            bucket,
            f"{prefixes['sentinel2'].rstrip('/')}/{s2_stem(start, end)}.parquet",
            prefix=prefixes["sentinel2"],
            project=cfg["gee"]["project_id"],
        )
    )
    s5p_ready = sum(
        1
        for day in eo_days
        if blob_listed(
            bucket,
            f"{prefixes['sentinel5p'].rstrip('/')}/{s5p_stem(day)}.parquet",
            prefix=prefixes["sentinel5p"],
            project=cfg["gee"]["project_id"],
        )
    )

    def summary(required: int, available: int) -> dict[str, int]:
        return {
            "required": required,
            "available": available,
            "missing": max(0, required - available),
            "scheduled": 0,
            "pending": 0,
        }

    return {
        "era5": summary(len(era5_days), era5_ready),
        "firms": summary(len(firms_days), firms_ready),
        "sentinel2": summary(len(s2_windows), s2_ready),
        "sentinel5p": summary(len(eo_days), s5p_ready),
    }


def _month_is_finished(day: date, *, as_of: date) -> bool:
    """True when the calendar month containing `day` has fully elapsed before as_of."""
    last = date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])
    return last < as_of


def _month_covers_day(
    bucket: str, prefix: str, day: date, *, as_of: date | None = None
) -> bool:
    """Skip daily CDS only if a monthly file exists and that month is finished.

    A current-month object (e.g. era5_2026_08.nc on 13 Aug) is often partial.
    Treating it as covering later August days would skip CDS for days not in the file.
    """
    as_of = as_of or pipeline_today(cfg)
    if not _month_is_finished(day, as_of=as_of):
        return False
    return _month_blob_exists(bucket, prefix, day)


def _download_flags(args: argparse.Namespace, cfg: dict) -> tuple[bool, bool]:
    skip = cfg["download"].get("skip_existing", True)
    if getattr(args, "no_skip_existing", False):
        skip = False
    force_s5p = bool(getattr(args, "force_s5p", False) or cfg["download"].get("s5p_force", False))
    return skip, force_s5p


def cmd_download_eo_day(
    eo_day: date,
    cfg: dict,
    *,
    skip: bool,
    force_s5p: bool,
    as_of: date | None = None,
) -> int:
    """Download S2 + S5P for one eo_asof day. Returns 2 if S5P still pending."""
    as_of = as_of or pipeline_today(cfg)
    if eo_day > as_of:
        logger.info("Skip EO eo_asof=%s (after as_of=%s)", eo_day, as_of)
        return 0

    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    project = cfg["gee"]["project_id"]

    emit_event("sentinel2", "running", f"Ensuring Sentinel-2 data for {eo_day}")
    t = time.perf_counter()
    download_s2_for_date(
        eo_day,
        project_id=project,
        bucket=bucket,
        prefix=prefixes["sentinel2"],
        grid_asset_id=cfg["gee"]["grid_asset_id"],
        skip_existing=skip,
        window_days=cfg["download"].get("s2_window_days", 5),
        as_of=as_of,
    )
    logger.info("Timing S2 eo_asof=%s: %s", eo_day, _fmt_secs(time.perf_counter() - t))

    emit_event("sentinel5p", "running", f"Ensuring Sentinel-5P data for {eo_day}")
    t = time.perf_counter()
    s5p_uri = download_s5p_for_date(
        eo_day,
        project_id=project,
        dest_bucket=bucket,
        dest_prefix=prefixes["sentinel5p"],
        skip_existing=skip,
        force=force_s5p,
    )
    logger.info(
        "Timing S5P eo_asof=%s: %s (result=%s)",
        eo_day,
        _fmt_secs(time.perf_counter() - t),
        s5p_uri or "EE pending",
    )
    if s5p_uri is None:
        return 2
    return 0


def cmd_download(args: argparse.Namespace, cfg: dict) -> int:
    """Legacy single-day download: --date is treated as eo_asof (S2/S5P).

    Prefer `download --start-date/--end-date` (label range) or `all --label-date`.
    FIRMS for the implied label day is skipped (live prediction does not need D's label).
    """
    skip, force_s5p = _download_flags(args, cfg)
    target = args.date
    as_of = pipeline_today(cfg)
    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    project = cfg["gee"]["project_id"]
    t0 = time.perf_counter()

    # For a lone --date, pull ERA5 for feature_end as if label = eo_asof + 1
    label = target + timedelta(days=1)
    lag = int(cfg["task"]["era5_lag_days"])
    lead = int(cfg["task"]["lead_days"])
    era5_day = era5_feature_end(label, lag, lead)
    logger.info(
        "Download eo_asof=%s (implies label=%s, ERA5 feature_end=%s) skip_existing=%s",
        target,
        label,
        era5_day,
        skip,
    )

    t = time.perf_counter()
    if skip and _month_covers_day(bucket, prefixes["era5"], era5_day, as_of=as_of):
        logger.info("ERA5 monthly covers %s — skip daily CDS", era5_day)
        era5_rc = 0
    elif skip and _era5_daily_exists(bucket, prefixes["era5"], era5_day):
        logger.info(
            "ERA5 daily already in bucket — skip CDS %s",
            era5_day,
        )
        era5_rc = 0
    else:
        if skip and _month_blob_exists(bucket, prefixes["era5"], era5_day):
            logger.info(
                "ERA5 monthly era5_%04d_%02d.nc exists but %04d-%02d is still open as of %s — daily CDS",
                era5_day.year,
                era5_day.month,
                era5_day.year,
                era5_day.month,
                as_of,
            )
        era5_rc = download_era5_day(
            era5_day,
            bucket=bucket,
            prefix=prefixes["era5"],
            project=project,
            workdir=cfg["download"].get("era5_workdir", "/tmp/era5"),
            skip_existing=skip,
        )
    logger.info("Timing ERA5: %s", _fmt_secs(time.perf_counter() - t))
    if era5_rc != 0:
        return era5_rc

    # Neighbor history: FIRMS for eo_asof day (= D−1), not the predict day.
    t = time.perf_counter()
    export_firms_day(
        target,
        project_id=project,
        bucket=bucket,
        prefix=prefixes["firms"],
        skip_existing=skip,
    )
    logger.info("Timing FIRMS history_day=%s: %s", target, _fmt_secs(time.perf_counter() - t))

    rc = cmd_download_eo_day(target, cfg, skip=skip, force_s5p=force_s5p, as_of=as_of)
    if rc == 2:
        wait = bool(cfg["download"].get("s5p_wait", True))
        if wait:
            wait_for_s5p_days(
                [target],
                project_id=project,
                dest_bucket=bucket,
                dest_prefix=prefixes["sentinel5p"],
                poll_seconds=int(cfg["download"].get("s5p_poll_seconds", 60)),
                timeout_seconds=int(cfg["download"].get("s5p_timeout_seconds", 6 * 3600)),
            )
            rc = 0
    logger.info("Timing download total: %s", _fmt_secs(time.perf_counter() - t0))
    return rc


def cmd_download_for_labels(args: argparse.Namespace, cfg: dict, labels: list[date]) -> int:
    """Download ERA5 + FIRMS(through D−1) + S2/S5P(eo_asof≤today) for the lookback window."""
    skip, force_s5p = _download_flags(args, cfg)
    as_of = pipeline_today(cfg)
    # Cap requested labels at today for live demos (no future Aug 13–31).
    capped = [d for d in labels if d <= as_of]
    if not capped:
        logger.error("All requested labels are after as_of=%s", as_of)
        return 1
    if len(capped) < len(labels):
        logger.warning(
            "Capped label range to as_of=%s (%d → %d days); skipped future labels",
            as_of,
            len(labels),
            len(capped),
        )
    labels = capped

    firms_days = firms_label_dates_needed(labels, cfg, as_of=as_of)
    eo_days = eo_asof_dates_needed(labels, cfg, as_of=as_of)
    era5_days = era5_days_needed(labels, cfg, as_of=as_of)

    logger.info(
        "Range download: %d label(s) %s…%s | FIRMS history=%d (%s…%s) | EO asof=%d (%s…%s) | ERA5 %s…%s",
        len(labels),
        min(labels),
        max(labels),
        len(firms_days),
        firms_days[0] if firms_days else None,
        firms_days[-1] if firms_days else None,
        len(eo_days),
        eo_days[0] if eo_days else None,
        eo_days[-1] if eo_days else None,
        era5_days[0] if era5_days else None,
        era5_days[-1] if era5_days else None,
    )

    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    project = cfg["gee"]["project_id"]

    if skip:
        from download.gcs_listing import prefetch_download_prefixes

        years = sorted(
            {d.year for d in era5_days}
            | {d.year for d in firms_days}
            | {d.year for d in eo_days}
        )
        prefetch_download_prefixes(bucket, prefixes, years=years, project=project)

    inventory = build_source_inventory(labels, cfg, as_of=as_of)
    emit_event(
        "inventory",
        "running",
        "Checked cloud storage for required causal inputs.",
        inventory=inventory,
    )

    logged_open_months: set[tuple[int, int]] = set()
    era5_skip_month = 0
    era5_skip_daily = 0
    emit_event("era5", "running", "Ensuring ERA5 history.", completed=0, total=len(era5_days))
    for day_index, day in enumerate(era5_days, 1):
        if skip and _month_covers_day(bucket, prefixes["era5"], day, as_of=as_of):
            era5_skip_month += 1
            continue
        if skip and _era5_daily_exists(bucket, prefixes["era5"], day):
            era5_skip_daily += 1
            continue
        if skip and _month_blob_exists(bucket, prefixes["era5"], day):
            key = (day.year, day.month)
            if key not in logged_open_months:
                logger.info(
                    "ERA5 monthly era5_%04d_%02d.nc exists but %04d-%02d is still open as of %s — daily CDS for uncovered days",
                    day.year,
                    day.month,
                    day.year,
                    day.month,
                    as_of,
                )
                logged_open_months.add(key)
        rc = download_era5_day(
            day,
            bucket=bucket,
            prefix=prefixes["era5"],
            project=project,
            workdir=cfg["download"].get("era5_workdir", "/tmp/era5"),
            skip_existing=skip,
        )
        if rc != 0:
            logger.error("ERA5 download failed for %s (exit=%s)", day, rc)
            return rc
        emit_event("era5", "running", f"ERA5 ready for {day}", completed=day_index, total=len(era5_days))

    if era5_skip_month or era5_skip_daily:
        logger.info(
            "ERA5 skipped %d day(s) via finished monthly file, %d day(s) already in GCS",
            era5_skip_month,
            era5_skip_daily,
        )

    for i, firms_day in enumerate(firms_days, 1):
        emit_event("firms", "running", f"Ensuring FIRMS history for {firms_day}", completed=i - 1, total=len(firms_days))
        logger.info("[%d/%d] FIRMS history_day=%s", i, len(firms_days), firms_day)
        export_firms_day(
            firms_day,
            project_id=project,
            bucket=bucket,
            prefix=prefixes["firms"],
            skip_existing=skip,
        )

    pending_s5p: list[date] = []
    for i, eo_day in enumerate(eo_days, 1):
        logger.info("[%d/%d] EO eo_asof=%s (S2+S5P)", i, len(eo_days), eo_day)
        rc = cmd_download_eo_day(eo_day, cfg, skip=skip, force_s5p=force_s5p, as_of=as_of)
        if rc == 2:
            pending_s5p.append(eo_day)
            continue
        if rc != 0:
            return rc

    wait_poll = int(cfg["download"].get("ee_poll_seconds", 60))
    wait_timeout = int(cfg["download"].get("ee_timeout_seconds", 6 * 3600))

    # Downloads above may have converted already-exported CSVs to parquet. Check
    # GCS again before initializing Earth Engine: a fully ready source must not
    # require EE credentials merely to pass through a no-op wait function.
    inventory = build_source_inventory(labels, cfg, as_of=as_of)
    for source in ("firms", "sentinel2", "sentinel5p"):
        missing = inventory[source]["missing"]
        inventory[source]["scheduled"] = missing
        inventory[source]["pending"] = missing
    emit_event(
        "inventory",
        "running",
        "Missing cloud inputs were submitted or reconciled.",
        inventory=inventory,
    )
    if firms_days and inventory["firms"]["missing"]:
        emit_event("firms", "waiting_external", "Waiting for FIRMS Earth Engine exports.", total=len(firms_days))
        wait_for_firms_days(
            firms_days,
            project_id=project,
            bucket=bucket,
            prefix=prefixes["firms"],
            poll_seconds=wait_poll,
            timeout_seconds=wait_timeout,
        )
        inventory["firms"].update(available=inventory["firms"]["required"], missing=0, pending=0)
        emit_event("firms", "running", "All FIRMS inputs are ready.", completed=len(firms_days), total=len(firms_days))

    if eo_days and inventory["sentinel2"]["missing"]:
        emit_event("sentinel2", "waiting_external", "Waiting for Sentinel-2 exports and parquet conversion.", total=len(eo_days))
        wait_for_s2_days(
            eo_days,
            project_id=project,
            bucket=bucket,
            prefix=prefixes["sentinel2"],
            grid_asset_id=cfg["gee"]["grid_asset_id"],
            window_days=int(cfg["download"].get("s2_window_days", 5)),
            poll_seconds=wait_poll,
            timeout_seconds=wait_timeout,
        )
        inventory["sentinel2"].update(available=inventory["sentinel2"]["required"], missing=0, pending=0)
        emit_event("sentinel2", "running", "All Sentinel-2 inputs are ready.", completed=len(eo_days), total=len(eo_days))

    if pending_s5p:
        wait = bool(cfg["download"].get("s5p_wait", True))
        if not wait:
            logger.error(
                "S5P pending for %d day(s); s5p_wait=false so exiting. Re-run later.",
                len(pending_s5p),
            )
            return 2
        logger.info(
            "Submitted/queued %d S5P day(s). EE runs them in parallel; waiting until COMPLETED…",
            len(pending_s5p),
        )
        emit_event(
            "sentinel5p",
            "waiting_external",
            "Waiting for Sentinel-5P Earth Engine exports.",
            total=len(pending_s5p),
        )
        wait_for_s5p_days(
            pending_s5p,
            project_id=project,
            dest_bucket=bucket,
            dest_prefix=prefixes["sentinel5p"],
            poll_seconds=int(cfg["download"].get("s5p_poll_seconds", 60)),
            timeout_seconds=int(cfg["download"].get("s5p_timeout_seconds", 6 * 3600)),
        )
    inventory["sentinel5p"].update(available=inventory["sentinel5p"]["required"], missing=0, pending=0)
    emit_event(
        "inventory",
        "running",
        "All required causal inputs are ready.",
        inventory=inventory,
    )
    emit_event("sentinel5p", "running", "All Sentinel-5P inputs are ready.", completed=len(eo_days), total=len(eo_days))
    return 0


def cmd_preprocess(args: argparse.Namespace, cfg: dict) -> int:
    m4_cfg = load_m4_config(cfg)
    t0 = time.perf_counter()
    emit_event("preprocessing", "running", "Building the causal Stage C feature panel.")
    run_stage_c_pipeline(args.label_date, cfg, m4_cfg, force=args.force)
    logger.info("Timing preprocess: %s", _fmt_secs(time.perf_counter() - t0))
    return 0


def cmd_export_day(args: argparse.Namespace, cfg: dict) -> int:
    from paths import resolve_path
    import pandas as pd

    label_date = args.label_date
    cache = resolve_path(cfg, "local_cache")
    history_path = cache / "m4_shared_cache" / "stage_c_knn" / "history" / "panel.parquet"
    if not history_path.exists():
        raise FileNotFoundError(f"Run preprocess first: missing {history_path}")
    history = pd.read_parquet(history_path)
    t0 = time.perf_counter()
    emit_event("exporting", "running", "Validating and exporting the 86-feature parquet.")
    frame, destination = export_champion_day(label_date, history, cfg, upload=not args.local_only)
    logger.info(
        "Timing export_day: %s → %s",
        _fmt_secs(time.perf_counter() - t0),
        destination,
    )
    lag = int(cfg["task"].get("era5_lag_days", 5))
    lead = int(cfg["task"].get("lead_days", 1))
    emit_event(
        "completed",
        "succeeded",
        "Prediction data is ready.",
        artifact={
            "objectUri": destination,
            "rowCount": int(len(frame)),
            "featureCount": 86,
            "cellCount": int(frame["cell_id"].nunique()),
            "labelDate": label_date.isoformat(),
            "eoAsOfDate": (label_date - timedelta(days=1)).isoformat(),
            "featureEndDate": (label_date - timedelta(days=lag + lead)).isoformat(),
            "createdAt": datetime.now().astimezone().isoformat(),
        },
    )
    return 0


def cmd_all_one(args: argparse.Namespace, cfg: dict, label_date: date, *, skip_download: bool = False) -> int:
    """Run full pipeline for one label date with stage timings."""
    day_t0 = time.perf_counter()
    eo = eo_asof_date(label_date)
    feat = era5_feature_end(
        label_date, cfg["task"]["era5_lag_days"], cfg["task"]["lead_days"]
    )
    logger.info(
        "=== Day pipeline label_date=%s eo_asof=%s era5_feature_end=%s ===",
        label_date,
        eo,
        feat,
    )
    args.label_date = label_date
    args.date = eo

    timings: dict[str, float] = {}

    if not skip_download:
        t = time.perf_counter()
        rc = cmd_download_for_labels(args, cfg, [label_date])
        timings["download"] = time.perf_counter() - t
        if rc != 0:
            return rc
    else:
        timings["download"] = 0.0

    t = time.perf_counter()
    cmd_preprocess(args, cfg)
    timings["preprocess"] = time.perf_counter() - t

    t = time.perf_counter()
    cmd_export_day(args, cfg)
    timings["export_day"] = time.perf_counter() - t

    total = time.perf_counter() - day_t0
    logger.info(
        "=== Timing summary label_date=%s | download=%s | preprocess=%s | export=%s | TOTAL=%s ===",
        label_date,
        _fmt_secs(timings["download"]),
        _fmt_secs(timings["preprocess"]),
        _fmt_secs(timings["export_day"]),
        _fmt_secs(total),
    )
    return 0


def cmd_all(args: argparse.Namespace, cfg: dict) -> int:
    labels = resolve_label_dates(args)
    emit_event("validating", "running", "Validated the requested prediction date.")
    as_of = pipeline_today(cfg)
    capped = [d for d in labels if d <= as_of]
    if not capped:
        logger.error("All requested labels are after as_of=%s", as_of)
        return 1
    if len(capped) < len(labels):
        logger.warning(
            "Capped label range to as_of=%s (%d → %d days); skipped future labels",
            as_of,
            len(labels),
            len(capped),
        )
    labels = capped
    logger.info("Running all for %d label date(s): %s … %s", len(labels), labels[0], labels[-1])
    wall0 = time.perf_counter()

    # GCS final parquet is the authoritative handoff to inference. Reuse it
    # only after the same contract applied to a fresh export passes. --force
    # and --local-only intentionally bypass this cloud short circuit.
    if not getattr(args, "force", False) and not getattr(args, "local_only", False):
        missing_or_invalid: list[date] = []
        reused: list[tuple[date, dict]] = []
        for label in labels:
            emit_event(
                "inventory",
                "running",
                f"Checking for prepared prediction data for {label}.",
                completed=0,
                total=1,
            )
            artifact = existing_final_artifact(label, cfg)
            if artifact is None:
                missing_or_invalid.append(label)
            else:
                reused.append((label, artifact))

        if not missing_or_invalid:
            # The application submits one date. For range CLI calls, the final
            # event represents the last requested artifact and logs cover all.
            _, artifact = reused[-1]
            emit_event(
                "completed",
                "succeeded",
                "Existing prediction data is ready.",
                completed=1,
                total=1,
                artifact=artifact,
            )
            logger.info("Reused %d validated final parquet object(s); no raw inventory needed", len(reused))
            return 0
        labels = missing_or_invalid
        emit_event(
            "inventory",
            "running",
            "Prepared prediction data is missing or invalid; checking required source files.",
        )

    t_dl = time.perf_counter()
    rc = cmd_download_for_labels(args, cfg, labels)
    logger.info("Timing range download: %s", _fmt_secs(time.perf_counter() - t_dl))
    if rc != 0:
        return rc

    for i, label in enumerate(labels, 1):
        logger.info("[%d/%d] label_date=%s", i, len(labels), label)
        rc = cmd_all_one(args, cfg, label, skip_download=True)
        if rc != 0:
            return rc
    logger.info(
        "All dates done (%d days). Wall clock: %s",
        len(labels),
        _fmt_secs(time.perf_counter() - wall0),
    )
    return 0


def cmd_download_range(args: argparse.Namespace, cfg: dict) -> int:
    """Download for label range, or single --date as eo_asof."""
    if getattr(args, "start_date", None) and getattr(args, "end_date", None):
        labels = date_range(args.start_date, args.end_date)
        return cmd_download_for_labels(args, cfg, labels)
    if getattr(args, "date", None):
        return cmd_download(args, cfg)
    raise SystemExit(
        "download needs --date (eo_asof day) or --start-date/--end-date (label range)"
    )


def cmd_publish_dem(args: argparse.Namespace, cfg: dict) -> int:
    from paths import resolve_path

    publish_dem(
        resolve_path(cfg, "dem_local"),
        bucket=cfg["gcs"]["bucket"],
        prefix=cfg["gcs"]["prefixes"]["dem"],
        dest_name=cfg["paths"]["dem_gcs_name"],
        project=cfg["gee"]["project_id"],
        skip_existing=not args.force,
    )
    return 0


def _add_range_args(p: argparse.ArgumentParser, *, label: bool) -> None:
    if label:
        p.add_argument(
            "--label-date",
            type=parse_date,
            help="Single label date (YYYY-MM-DD).",
        )
    p.add_argument("--start-date", type=parse_date, help="Range start (inclusive).")
    p.add_argument("--end-date", type=parse_date, help="Range end (inclusive).")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Unified daily wildfire pipeline.")
    parser.add_argument("--config", type=Path, default=_UTILS / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser(
        "download",
        help="Download raw sources. Label range preferred; --date = eo_asof day.",
    )
    p_dl.add_argument("--date", type=parse_date, help="Single eo_asof date (S2/S5P).")
    _add_range_args(p_dl, label=False)
    p_dl.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download raw layers even if present in GCS.",
    )
    p_dl.add_argument(
        "--force-s5p",
        action="store_true",
        help="Re-submit S5P EE export even after CANCELLED/FAILED (clears local registry).",
    )

    p_pp = sub.add_parser("preprocess", help="Build Stage C KNN history for label date.")
    p_pp.add_argument("--label-date", type=parse_date, required=True)
    p_pp.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Stage A/EO caches even if intermediate files exist.",
    )

    p_ex = sub.add_parser(
        "export_day",
        help="Engineer + prune 86 features and write final_processed.",
    )
    p_ex.add_argument("--label-date", type=parse_date, required=True)
    p_ex.add_argument("--local-only", action="store_true")

    p_all = sub.add_parser(
        "all",
        help="Reuse a valid final parquet; otherwise inventory/download + preprocess + export_day.",
    )
    _add_range_args(p_all, label=True)
    p_all.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of final parquet and Stage C intermediate caches.",
    )
    p_all.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download raw layers even if present.",
    )
    p_all.add_argument(
        "--force-s5p",
        action="store_true",
        help="Re-submit S5P after CANCELLED/FAILED.",
    )
    p_all.add_argument("--local-only", action="store_true", help="Skip GCS upload of final parquet.")

    p_dem = sub.add_parser("publish_dem", help="One-shot DEM publish to GCS.")
    p_dem.add_argument("--force", action="store_true")

    args = parser.parse_args()
    cfg = load_daily_config(args.config)

    handlers = {
        "download": cmd_download_range,
        "preprocess": cmd_preprocess,
        "export_day": cmd_export_day,
        "all": cmd_all,
        "publish_dem": cmd_publish_dem,
    }
    return handlers[args.command](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
