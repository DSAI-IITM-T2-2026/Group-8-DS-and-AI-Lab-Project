#!/usr/bin/env python3
"""Single CLI: download | preprocess | export_day | all.

Date conventions (Milestone 4):
  label_date D          → day we predict / write *_test.parquet
  eo_asof_date          → D − 1  (S2 / S5P causal join)
  feature_end_date      → D − (era5_lag + lead) = D − 6  (ERA5)
  FIRMS y_fire          → on label_date D
"""

from __future__ import annotations

import argparse
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

from config_loader import load_daily_config, load_m4_config  # noqa: E402
from download.dem import publish_dem  # noqa: E402
from download.era5 import download_era5_day  # noqa: E402
from download.firms import export_firms_day  # noqa: E402
from download.sentinel2 import download_s2_for_date  # noqa: E402
from download.sentinel5p import download_s5p_for_date, wait_for_s5p_days  # noqa: E402
from preprocess.build_stage_c_day import run_stage_c_pipeline  # noqa: E402
from preprocess.export_inference_day import export_champion_day  # noqa: E402

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
    lookback = int(cfg["task"].get("lookback_days", 7))
    label_start = min(labels) - timedelta(days=lookback)
    label_end = max(labels)
    return date_range(label_start, label_end)


def era5_days_needed(labels: list[date], cfg: dict) -> list[date]:
    """ERA5 calendar days for Stage A: history ending at each label's feature_end."""
    lookback = int(cfg["task"].get("lookback_days", 7))
    history = int(cfg["task"].get("history_days", 7))
    lag = int(cfg["task"]["era5_lag_days"])
    lead = int(cfg["task"]["lead_days"])
    label_start = min(labels) - timedelta(days=lookback)
    label_end = max(labels)
    era5_end = label_end - timedelta(days=lag + lead)
    era5_start = label_start - timedelta(days=history + lag + lead)
    return date_range(era5_start, era5_end)


def eo_asof_dates_needed(labels: list[date], cfg: dict) -> list[date]:
    """Unique eo_asof (= label−1) days for S2/S5P over the label lookback window."""
    return sorted({eo_asof_date(d) for d in label_window(labels, cfg)})


def firms_label_dates_needed(labels: list[date], cfg: dict) -> list[date]:
    """FIRMS geotiffs keyed by label_date (y_fire / neighbor history)."""
    return label_window(labels, cfg)


def _fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {s:.0f}s"


def _month_uri_exists(bucket: str, prefix: str, day: date) -> bool:
    """True if monthly ERA5 covering this day exists (era5/YYYY or era5/raw/YYYY)."""
    from google.cloud import storage

    stem = f"era5_{day.year}_{day.month:02d}.nc"
    client = storage.Client()
    b = client.bucket(bucket)
    for name in (
        f"{prefix.rstrip('/')}/{day.year}/{stem}",
        f"{prefix.rstrip('/')}/raw/{day.year}/{stem}",
    ):
        if b.blob(name).exists():
            return True
    return False


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
) -> int:
    """Download S2 + S5P for one eo_asof day. Returns 2 if S5P still pending."""
    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    project = cfg["gee"]["project_id"]

    t = time.perf_counter()
    download_s2_for_date(
        eo_day,
        project_id=project,
        bucket=bucket,
        prefix=prefixes["sentinel2"],
        grid_asset_id=cfg["gee"]["grid_asset_id"],
        skip_existing=skip,
        window_days=cfg["download"].get("s2_window_days", 5),
    )
    logger.info("Timing S2 eo_asof=%s: %s", eo_day, _fmt_secs(time.perf_counter() - t))

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
    """Legacy single-day download: --date is treated as eo_asof (S2/S5P) + FIRMS that day.

    Prefer `download --start-date/--end-date` (label range) or `all --label-date`.
    """
    skip, force_s5p = _download_flags(args, cfg)
    target = args.date
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
    if skip and _month_uri_exists(bucket, prefixes["era5"], era5_day):
        logger.info("ERA5 monthly covers %s — skip daily CDS", era5_day)
        era5_rc = 0
    else:
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

    t = time.perf_counter()
    export_firms_day(
        label,
        project_id=project,
        bucket=bucket,
        prefix=prefixes["firms"],
        skip_existing=skip,
    )
    logger.info("Timing FIRMS label=%s: %s", label, _fmt_secs(time.perf_counter() - t))

    rc = cmd_download_eo_day(target, cfg, skip=skip, force_s5p=force_s5p)
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
    """Download ERA5 + FIRMS(label) + S2/S5P(eo_asof) for the label lookback window."""
    skip, force_s5p = _download_flags(args, cfg)
    firms_days = firms_label_dates_needed(labels, cfg)
    eo_days = eo_asof_dates_needed(labels, cfg)
    era5_days = era5_days_needed(labels, cfg)

    logger.info(
        "Range download: %d label(s) %s…%s | FIRMS labels=%d | EO asof=%d (%s…%s) | ERA5 %s…%s",
        len(labels),
        min(labels),
        max(labels),
        len(firms_days),
        len(eo_days),
        eo_days[0],
        eo_days[-1],
        era5_days[0],
        era5_days[-1],
    )

    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    project = cfg["gee"]["project_id"]

    for day in era5_days:
        if skip and _month_uri_exists(bucket, prefixes["era5"], day):
            continue
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

    for i, label in enumerate(firms_days, 1):
        logger.info("[%d/%d] FIRMS label_date=%s", i, len(firms_days), label)
        export_firms_day(
            label,
            project_id=project,
            bucket=bucket,
            prefix=prefixes["firms"],
            skip_existing=skip,
        )

    pending_s5p: list[date] = []
    for i, eo_day in enumerate(eo_days, 1):
        logger.info("[%d/%d] EO eo_asof=%s (S2+S5P)", i, len(eo_days), eo_day)
        rc = cmd_download_eo_day(eo_day, cfg, skip=skip, force_s5p=force_s5p)
        if rc == 2:
            pending_s5p.append(eo_day)
            continue
        if rc != 0:
            return rc

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
        wait_for_s5p_days(
            pending_s5p,
            project_id=project,
            dest_bucket=bucket,
            dest_prefix=prefixes["sentinel5p"],
            poll_seconds=int(cfg["download"].get("s5p_poll_seconds", 60)),
            timeout_seconds=int(cfg["download"].get("s5p_timeout_seconds", 6 * 3600)),
        )
    return 0


def cmd_preprocess(args: argparse.Namespace, cfg: dict) -> int:
    m4_cfg = load_m4_config(cfg)
    t0 = time.perf_counter()
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
    path = export_champion_day(label_date, history, cfg, upload=not args.local_only)
    logger.info(
        "Timing export_day: %s → %s",
        _fmt_secs(time.perf_counter() - t0),
        path[1] if isinstance(path, tuple) else path,
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
    logger.info("Running all for %d label date(s): %s … %s", len(labels), labels[0], labels[-1])
    wall0 = time.perf_counter()

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
        help="Engineer + prune 86 features (always regenerates final_processed).",
    )
    p_ex.add_argument("--label-date", type=parse_date, required=True)
    p_ex.add_argument("--local-only", action="store_true")

    p_all = sub.add_parser(
        "all",
        help="download + preprocess + export_day. Raw skip-if-exists; final always regenerated.",
    )
    _add_range_args(p_all, label=True)
    p_all.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of Stage C intermediate caches.",
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
