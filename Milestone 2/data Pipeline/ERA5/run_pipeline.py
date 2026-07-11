#!/usr/bin/env python3
"""ERA5 Wildfire Data Pipeline — main orchestrator."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.download_manager import run_downloads
from pipeline.features import process_all_years
from pipeline.merger import merge_all_years
from pipeline.metadata import generate_metadata, generate_summary_csv
from pipeline.reports import generate_quality_report
from pipeline.validator import validate_all_raw
from pipeline.visualization import generate_plots


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERA5 Wildfire Data Pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "download", "validate", "merge", "process", "report"],
        default="all",
        help="Pipeline phase to run",
    )
    parser.add_argument(
        "--year",
        help="Limit processing to a single year (e.g. 2020)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    config.ensure_directories()
    setup_logging(config.paths["logs"])

    logger = logging.getLogger("pipeline")
    logger.info("Starting ERA5 Wildfire Pipeline (phase=%s)", args.phase)

    if args.year:
        config.years = [args.year]

    download_stats = {"total": 0, "success": 0, "failed": 0}
    validation_results = []

    if args.phase in ("all", "download"):
        logger.info("=== Phase 1: Download ===")
        download_stats = run_downloads(config)

    if args.phase in ("all", "validate", "merge", "process", "report"):
        logger.info("=== Phase 2: Validation ===")
        validation_results = validate_all_raw(config)

    if args.phase in ("all", "merge", "process", "report"):
        logger.info("=== Phase 3: Monthly Merge ===")
        merge_results = merge_all_years(config)
        logger.info("Merge results: %s", merge_results)

    if args.phase in ("all", "process", "report"):
        logger.info("=== Phase 4: Feature Engineering ===")
        process_results = process_all_years(config)
        logger.info("Process results: %s", process_results)

    if args.phase in ("all", "report"):
        logger.info("=== Phase 5: Metadata & Reports ===")
        metadata = generate_metadata(config, validation_results, download_stats)
        generate_summary_csv(config, validation_results)
        plot_paths = generate_plots(config)
        generate_quality_report(config, validation_results, metadata, plot_paths)

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
