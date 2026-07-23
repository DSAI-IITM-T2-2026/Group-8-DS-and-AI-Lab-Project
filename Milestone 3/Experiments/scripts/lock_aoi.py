#!/usr/bin/env python3
"""Compute and lock 5-source AOI intersection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GS_NO_SIGN_REQUEST"] = "YES"

from src.data.aoi import lock_intersection_aoi


def main():
    result = lock_intersection_aoi()
    print("Done.")
    print(result["aoi_bounds"])


if __name__ == "__main__":
    main()
