from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import load_config
from .inference import score_parquet
from .pipeline.runner import STAGES, run_pipeline


def parse_int_set(value: str) -> list[int]:
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one or all resumable pipeline stages")
    run.add_argument("--config", default=None)
    run.add_argument("--stage", choices=[*STAGES, "all", "demo"], default="all")
    run.add_argument("--years", default="2019-2025")
    run.add_argument("--months", default="1-12")
    run.add_argument("--worker", default=os.environ.get("USER", "local"))
    run.add_argument("--era5-lag-days", type=int, default=None)
    run.add_argument("--force", action="store_true")

    predict = subparsers.add_parser("predict", help="Score a prepared point-in-time parquet")
    predict.add_argument("--routing", required=True, type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.command == "predict":
        score_parquet(args.routing, args.input, args.output)
        return 0
    cfg = load_config(args.config)
    lag = int(cfg["task"]["era5_lag_days"]) if args.era5_lag_days is None else args.era5_lag_days
    run_pipeline(
        cfg,
        args.stage,
        parse_int_set(args.years),
        parse_int_set(args.months),
        args.worker,
        lag,
        args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
