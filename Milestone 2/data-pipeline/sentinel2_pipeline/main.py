#!/usr/bin/env python3
"""
Sentinel-2 Standalone Fetch Pipeline — CLI entrypoint.

Run `python main.py --help` for options. See README.md for setup and for
the "confirmed" vs "alt_design" logic mode explanation — read that before
trusting any output from this script.
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")  # allow `src` to be imported when run as a script

from src.config import load_config
from src.ee_client import ee_init
from src.export import check_tasks, reconcile_progress
from src.pipeline import run
from src.progress import load_progress, save_progress, upload_progress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    parser = argparse.ArgumentParser(description="Standalone Sentinel-2 GEE -> GCS export pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--check-tasks", action="store_true", help="List active GEE export tasks and exit")
    parser.add_argument(
        "--logic",
        choices=["confirmed", "alt_design"],
        default=None,
        help="Override logic_mode from config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.logic and args.logic != cfg.get("logic_mode"):
        cfg["logic_mode"] = args.logic
        if args.logic == "alt_design":
            logging.getLogger("sentinel2").warning(
                "CLI --logic alt_design override active — outputs tagged ALT_DESIGN_UNVERIFIED."
            )
        else:
            logging.getLogger("sentinel2").info("CLI --logic confirmed override active.")

    ee_init(cfg)

    if args.check_tasks:
        progress_path = cfg["progress"]["local_file"]
        progress = reconcile_progress(load_progress(progress_path))
        save_progress(progress_path, progress)
        upload_progress(progress_path, cfg)
        check_tasks(progress)
        return

    run(cfg)


if __name__ == "__main__":
    main()
