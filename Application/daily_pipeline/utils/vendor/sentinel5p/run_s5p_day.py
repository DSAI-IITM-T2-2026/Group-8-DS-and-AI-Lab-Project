#!/usr/bin/env python3
"""DEPRECATED for imports. Use s5p_lib.py for daily pipeline.

Year-wide EE submit only when run as: python run_s5p_day.py
"""
from __future__ import annotations
import runpy
from pathlib import Path

if __name__ == "__main__":
    # Preserve original year-runner by executing sibling backup if present
    backup = Path(__file__).with_name("run_s5p_day_year_cli.py")
    if backup.exists():
        runpy.run_path(str(backup), run_name="__main__")
    else:
        raise SystemExit(
            "Year CLI moved. Daily pipeline uses s5p_lib.py. "
            "Restore run_s5p_day_year_cli.py if you need full-year submit."
        )
else:
    raise ImportError(
        "Do not import run_s5p_day (it used to submit YEAR=2019 on import). "
        "Import from s5p_lib instead."
    )
