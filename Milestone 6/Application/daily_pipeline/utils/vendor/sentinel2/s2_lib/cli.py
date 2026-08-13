"""CLI for the Sentinel-2 feature generation pipeline."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config, resolve_path
from src.logging import setup_logging
from src.scheduler import FeatureScheduler, estimate_completion
from src.state import CANCELLED, COMPLETED, FAILED, PENDING, RETRYING, RUNNING, StateDB


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Sentinel-2 ML feature generation pipeline (GEE → GCS)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: config/config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Start the export scheduler")
    run_p.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll/submit cycle then exit",
    )
    run_p.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help=(
            "Test mode: export only the temporal window that contains this day "
            "(e.g. 2025-12-31). Does not queue the full multi-year run."
        ),
    )
    run_p.add_argument(
        "--year",
        type=int,
        metavar="YYYY",
        help=(
            "Export all windows that start in this calendar year "
            "(e.g. 2025). Uses a year-specific state DB."
        ),
    )
    run_p.add_argument(
        "--month",
        metavar="YYYY-MM",
        help=(
            "Export all windows that start in a calendar month, newest first "
            "(e.g. 2018-12). Uses that year's state DB."
        ),
    )
    run_p.add_argument(
        "--submit-all",
        action="store_true",
        help=(
            "Submit every pending window to Earth Engine, then exit. "
            "EE keeps running after you shut the laptop. "
            "Re-run later (without --submit-all) to convert CSV→Parquet."
        ),
    )
    run_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --date, --month, or --year: re-queue target windows even "
            "if already COMPLETED in the state DB."
        ),
    )

    sub.add_parser("resume", help="Resume interrupted exports (same as run)")
    sub.add_parser(
        "export-grid",
        help="One-time export of the exact California 1 km grid to an EE asset",
    )
    status_p = sub.add_parser("status", help="Show progress summary")
    status_p.add_argument(
        "--year",
        type=int,
        metavar="YYYY",
        help="Show status for a year run",
    )
    sub.add_parser("retry", help="Re-queue FAILED windows as RETRYING")

    return parser


def _scheduler(
    args: argparse.Namespace, *, state_db: Path | None = None
) -> FeatureScheduler:
    cfg = load_config(args.config)
    logger = setup_logging(resolve_path(cfg, cfg.scheduler.log_dir))
    db = StateDB(state_db) if state_db is not None else None
    return FeatureScheduler(cfg, logger, db=db)


def _year_state_db(cfg, year: int) -> Path:
    base = Path(cfg.scheduler.state_db)
    name = f"{base.stem}_{year}{base.suffix or '.db'}"
    return resolve_path(cfg, str(base.with_name(name)))


def cmd_run(args: argparse.Namespace) -> int:
    only_date = None
    only_year = getattr(args, "year", None)
    only_month = None
    if getattr(args, "date", None):
        try:
            only_date = date.fromisoformat(args.date)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --date '{args.date}'; expected YYYY-MM-DD"
            ) from exc
    if getattr(args, "month", None):
        try:
            month_date = date.fromisoformat(f"{args.month}-01")
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --month '{args.month}'; expected YYYY-MM"
            ) from exc
        if len(args.month) != 7:
            raise SystemExit(f"Invalid --month '{args.month}'; expected YYYY-MM")
        only_month = (month_date.year, month_date.month)
    if sum(value is not None for value in (only_date, only_year, only_month)) > 1:
        raise SystemExit("Use only one of --date / --month / --year")
    if (
        getattr(args, "force", False)
        and only_date is None
        and only_year is None
        and only_month is None
    ):
        raise SystemExit("--force requires --date, --month, or --year")

    cfg = load_config(args.config)
    state_db = None
    state_year = only_year if only_year is not None else (
        only_month[0] if only_month is not None else None
    )
    if state_year is not None:
        state_db = _year_state_db(cfg, state_year)

    _scheduler(args, state_db=state_db).run(
        once=getattr(args, "once", False),
        only_date=only_date,
        only_year=only_year,
        only_month=only_month,
        force=getattr(args, "force", False),
        submit_all=getattr(args, "submit_all", False),
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    _scheduler(args).run(once=False)
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    sched = _scheduler(args)
    n = sched.retry_failed()
    print(f"Re-queued {n} FAILED window(s) as RETRYING")
    return 0


def cmd_export_grid(args: argparse.Namespace) -> int:
    from src import export as export_mod

    cfg = load_config(args.config)
    export_mod.initialize(cfg.project_id)
    task = export_mod.create_grid_export_task(cfg)
    task.start()
    print(f"Grid export started: {task.id}")
    print(f"Asset: {cfg.grid.asset_id}")
    print("Wait for this task to complete before starting window exports.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    year = getattr(args, "year", None)
    db_path = (
        _year_state_db(cfg, year)
        if year is not None
        else resolve_path(cfg, cfg.scheduler.state_db)
    )
    db = StateDB(db_path)

    counts = db.count_by_status()
    total = db.total_windows()
    completed = counts.get(COMPLETED, 0)
    running = counts.get(RUNNING, 0)
    pending = counts.get(PENDING, 0) + counts.get(RETRYING, 0)
    failed = counts.get(FAILED, 0)
    cancelled = counts.get(CANCELLED, 0)
    remaining = total - completed - failed - cancelled
    eta = estimate_completion(
        counts,
        cfg.scheduler.poll_interval_seconds,
        cfg.scheduler.max_running_tasks,
    )

    print("Sentinel-2 Feature Pipeline Status")
    print("=================================")
    print(f"Config:     {cfg.config_path}")
    print(f"State DB:   {db.db_path}")
    print(f"AOI:        N={cfg.aoi.north} S={cfg.aoi.south} W={cfg.aoi.west} E={cfg.aoi.east}")
    print(
        f"Temporal:   {cfg.temporal.start_year}–{cfg.temporal.end_year} "
        f"({cfg.temporal.window_days}-day windows)"
    )
    print(f"Grid:       {cfg.grid.resolution_m:g} m ({cfg.grid.crs})")
    print(f"Collection: {cfg.sentinel2.collection}")
    print(f"Export:     gs://{cfg.export.bucket}/{cfg.export.prefix}/ ({cfg.export.format})")
    print()
    print(f"Completed:  {completed}")
    print(f"Running:    {running}")
    print(f"Pending:    {pending}")
    print(f"Failed:     {failed}")
    print(f"Cancelled:  {cancelled}")
    print(f"Remaining:  {remaining}")
    print(f"Total:      {total}")
    print(f"Estimated completion: {eta}")

    if failed:
        print()
        print("Failed windows (up to 20):")
        for rec in db.list_by_status(FAILED)[:20]:
            print(f"  - {rec.window_id}: {rec.error_message}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "run": cmd_run,
        "resume": cmd_resume,
        "export-grid": cmd_export_grid,
        "status": cmd_status,
        "retry": cmd_retry,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
