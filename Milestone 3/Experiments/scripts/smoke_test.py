#!/usr/bin/env python3
"""Smoke test: MPS device + anonymous GCS listing + one FIRMS open."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GS_NO_SIGN_REQUEST"] = "YES"

from src import config
from src.data.gcs import get_fs, list_bucket_files
from src.data.loaders import load_firms_raster


def main():
    print("=== Milestone 3 smoke test ===")
    print(f"DEVICE = {config.DEVICE}")
    print(f"MPS available = {__import__('torch').backends.mps.is_available()}")
    print(f"M3_ROOT = {config.M3_ROOT}")
    print(f"Channels = {len(config.FEATURE_CHANNEL_NAMES)}")

    fs = get_fs()
    firms = list_bucket_files(config.FIRMS_BUCKET, config.FIRMS_PREFIX, suffix=".tif")
    print(f"FIRMS files listed: {len(firms)}")
    others = list_bucket_files(config.GCS_BUCKET, config.PREFIX_S2, suffix=".tif")
    print(f"Sentinel-2 tiles listed: {len(others)}")

    da = load_firms_raster("2024-08-15")
    print(f"FIRMS 2024-08-15 shape: {da.shape}, CRS: {da.rio.crs}")
    print("Smoke test OK")


if __name__ == "__main__":
    main()
