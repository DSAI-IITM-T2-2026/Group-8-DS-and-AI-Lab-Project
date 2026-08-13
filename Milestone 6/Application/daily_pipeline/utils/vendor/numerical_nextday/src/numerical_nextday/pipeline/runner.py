"""CLI pipeline runner for Milestone 4 numerical_nextday."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from numerical_nextday.config import load_config, shared_cache
from numerical_nextday.data.era5_firms import (
    assemble_stage_a_year,
    build_era5_firms_month,
    parse_year_month_ranges,
)
from numerical_nextday.data.s2_s5p import (
    attach_s2_stage_b,
    attach_s5p_stage_c,
    build_eo_cell_cache,
    merge_stage_a,
    write_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

STAGES_HELP = """
Stages:
  verify_gcs           List GCS prefixes
  era5_firms           Monthly ERA5 + FIRMS caches
  stage_a_year         Lag-aware Stage A year parquet (use --era5-lag-days)
  merge_stage_a        Concat Stage A + train/val/test
  s2_cell_cache        S2 CSV → ERA5 cell means
  s5p_cell_cache       S5P CSV → ERA5 cell means (skips 2021 if placeholder)
  s2_attach / s5p_attach   Stage B / C causal joins
  write_splits         Refresh splits + metadata
  build_data           Solo full data build (lag-5 primary)
  build_lag0_data      Oracle lag-0 Stage A→C under m4_shared_cache/lag0/
  synthetic_smoke      Tiny synthetic tables (dry-run only)
  train_fire_season    A/B/C defaults + LGBM HP + MLP
  train_month_models   jan/feb/mar/dec
  train_lag0_ablation  fire_season LGBM on lag0 Stage C
  train_all            train_fire_season → month models → eval
  eval_figures         Metrics + figures
  run_all              build_data → train_all
"""


def _parse_years(spec: str | None) -> list[int]:
    if not spec:
        return list(range(2019, 2026))
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return years


def _parse_months(spec: str | None) -> list[int]:
    if not spec:
        return list(range(1, 13))
    months: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            months.extend(range(int(a), int(b) + 1))
        else:
            months.append(int(part))
    return months


def stage_verify_gcs(cfg: dict) -> int:
    repo_docs = Path(__file__).resolve().parents[5] / "docs" / "validate_m4_gcs.sh"
    path = repo_docs
    if not path.exists():
        logger.warning("validate_m4_gcs.sh not found — checking ERA5 prefix only")
        uri = cfg["gcs"]["era5_prefix"].rstrip("/") + "/"
        r = subprocess.run(["gsutil", "ls", uri], capture_output=True, text=True)
        logger.info("gsutil ls %s → rc=%s", uri, r.returncode)
        return r.returncode
    r = subprocess.run(["bash", str(path)], cwd=str(path.parent.parent))
    return r.returncode


def _s5p_years(cfg: dict, years: list[int]) -> list[int]:
    if cfg.get("s5p_2021_mode", "placeholder") == "placeholder":
        return [y for y in years if y != 2021]
    return years


def run_build_data(
    cfg: dict,
    years: list[int],
    months: list[int],
    worker: str,
    force: bool,
    limit_windows: int | None,
    lag: int | None = None,
) -> int:
    """Full solo data build. lag=None uses config (default 5); lag=0 → lag0/ tree."""
    if lag is not None:
        cfg = dict(cfg)
        cfg["task"] = dict(cfg["task"])
        cfg["task"]["era5_lag_days"] = lag

    effective_lag = int(cfg["task"]["era5_lag_days"])
    logger.info("=== build_data years=%s months=%s era5_lag_days=%s ===", years, months, effective_lag)

    for y, m in parse_year_month_ranges(years, months):
        build_era5_firms_month(cfg, y, m, worker=worker, force=force)

    for y in years:
        assemble_stage_a_year(
            cfg, y, months=months, worker=worker, force=force, era5_lag_days=effective_lag
        )

    path = merge_stage_a(cfg, years)
    import pandas as pd

    write_splits(cfg, pd.read_parquet(path), "stage_a")

    build_eo_cell_cache(
        cfg, "s2", years, months=months, worker=worker, force=force, limit_windows=limit_windows
    )
    attach_s2_stage_b(cfg, years)

    s5_years = _s5p_years(cfg, years)
    build_eo_cell_cache(
        cfg, "s5p", s5_years, months=months, worker=worker, force=force, limit_windows=limit_windows
    )
    attach_s5p_stage_c(cfg, years)
    logger.info("=== build_data done (lag=%s) ===", effective_lag)
    return 0


def run_train_all(cfg: dict) -> int:
    from numerical_nextday.eval.figures import run_eval_figures
    from numerical_nextday.train.lgbm import run_fire_season_schedule, run_month_models

    logger.info("=== train_all ===")
    rc = run_fire_season_schedule(cfg, model_bucket="fire_season")
    if rc:
        return rc
    rc = run_month_models(cfg)
    if rc:
        return rc
    return run_eval_figures(cfg)


def run_stage(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.era5_lag_days is not None:
        cfg["task"]["era5_lag_days"] = args.era5_lag_days
    years = _parse_years(args.years)
    months = _parse_months(args.months)
    worker = args.worker
    force = args.force
    stage = args.stage

    if stage == "verify_gcs":
        return stage_verify_gcs(cfg)

    if stage == "era5_firms":
        for y, m in parse_year_month_ranges(years, months):
            build_era5_firms_month(cfg, y, m, worker=worker, force=force)
        return 0

    if stage == "stage_a_year":
        for y in years:
            assemble_stage_a_year(
                cfg,
                y,
                months=months,
                worker=worker,
                force=force,
                era5_lag_days=args.era5_lag_days,
            )
        return 0

    if stage == "merge_stage_a":
        path = merge_stage_a(cfg, years)
        import pandas as pd

        df = pd.read_parquet(path)
        write_splits(cfg, df, "stage_a")
        logger.info("Merged Stage A → %s", path)
        return 0

    if stage == "s2_cell_cache":
        build_eo_cell_cache(
            cfg, "s2", years, months=months, worker=worker, force=force, limit_windows=args.limit_windows
        )
        return 0

    if stage == "s5p_cell_cache":
        build_eo_cell_cache(
            cfg,
            "s5p",
            _s5p_years(cfg, years),
            months=months,
            worker=worker,
            force=force,
            limit_windows=args.limit_windows,
        )
        return 0

    if stage == "s2_attach":
        attach_s2_stage_b(cfg, years)
        return 0

    if stage == "s5p_attach":
        if args.years and "2021" in str(args.years):
            cfg["s5p_2021_mode"] = "ready"
        attach_s5p_stage_c(cfg, years)
        return 0

    if stage == "write_splits":
        cache = shared_cache(cfg)
        import pandas as pd
        from numerical_nextday.data.s2_s5p import _lag_prefix

        root = _lag_prefix(cfg)
        for name in ("stage_a", "stage_b", "stage_c"):
            all_p = root / name / "all.parquet"
            if all_p.exists():
                write_splits(cfg, pd.read_parquet(all_p), name)
        return 0

    if stage == "build_data":
        return run_build_data(
            cfg, years, months, worker, force, args.limit_windows, lag=args.era5_lag_days
        )

    if stage == "build_lag0_data":
        return run_build_data(cfg, years, months, worker, force, args.limit_windows, lag=0)

    if stage == "synthetic_smoke":
        from numerical_nextday.data.synthetic import build_synthetic_tables

        build_synthetic_tables(cfg)
        return 0

    if stage == "train_fire_season":
        from numerical_nextday.train.lgbm import run_fire_season_schedule

        return run_fire_season_schedule(cfg, model_bucket=args.model_bucket or "fire_season")

    if stage == "train_month_models":
        from numerical_nextday.train.lgbm import run_month_models

        return run_month_models(cfg)

    if stage == "train_lag0_ablation":
        from numerical_nextday.train.lgbm import run_lag0_ablation

        return run_lag0_ablation(cfg)

    if stage == "train_all":
        return run_train_all(cfg)

    if stage == "eval_figures":
        from numerical_nextday.eval.figures import run_eval_figures

        return run_eval_figures(cfg)

    if stage == "run_all":
        rc = run_build_data(
            cfg, years, months, worker, force, args.limit_windows, lag=args.era5_lag_days
        )
        if rc:
            return rc
        return run_train_all(cfg)

    raise SystemExit(f"Unknown stage: {stage}\n{STAGES_HELP}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M4 numerical_nextday pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=STAGES_HELP,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run a pipeline stage", epilog=STAGES_HELP,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    run.add_argument(
        "--stage",
        required=True,
        help="Stage name (see --help epilog). Orchestrators: build_data, train_all, run_all",
    )
    run.add_argument("--config", default=None)
    run.add_argument("--years", default=None, help="e.g. 2019-2025 or 2024 (default 2019-2025)")
    run.add_argument("--months", default=None, help="e.g. 1-12 or 8 (default 1-12)")
    run.add_argument("--worker", default="local")
    run.add_argument("--era5-lag-days", type=int, default=None, help="Override config lag (5 primary, 0 ablation)")
    run.add_argument("--model-bucket", default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--limit-windows", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return run_stage(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
