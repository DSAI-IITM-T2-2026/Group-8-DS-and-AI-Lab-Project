#!/usr/bin/env python3
"""Run full EDA pipeline: clip → extract → ERA5 fusion analysis."""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    for name in (
        "01_clip_to_california.py",
        "02_extract_numerical_values.py",
        "03_era5_fusion_analysis.py",
    ):
        print(f"\n=== Running {name} ===")
        runpy.run_path(str(SCRIPTS / name), run_name="__main__")


if __name__ == "__main__":
    main()
