from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1] / "utils" / "vendor" / "mvp_era5_dem"
NUMERICAL_SRC = Path(__file__).resolve().parents[1] / "utils" / "vendor" / "numerical_nextday" / "src"
sys.path.insert(0, str(NUMERICAL_SRC))

from numerical_nextday.data.m3_imports import _load_pkg_module  # noqa: E402

firms_labels = _load_pkg_module("firms_truth_test", ROOT / "src" / "firms_labels.py", ROOT)


def write_firms(path: Path) -> None:
    confidence = np.array([[50.0, 20.0], [np.nan, np.nan]], dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-120.125, 38.125, 0.25, 0.25),
    ) as target:
        target.write(confidence, 1)
        target.write(np.zeros((2, 2), dtype="float32"), 2)
        target.write((confidence >= 30).astype("float32"), 3)
        target.set_band_description(1, "firms_confidence")
        target.set_band_description(2, "firms_t21")
        target.set_band_description(3, "label")


def test_completed_firms_raster_maps_qualified_pixels_to_grid_cells(tmp_path):
    target_day = pd.Timestamp("2026-08-14")
    write_firms(tmp_path / "2026-08-14.tif")

    labels = firms_labels.label_day_to_cells(target_day, str(tmp_path), 30, strict=True)

    assert labels[["cell_id", "firms_n_pixels", "firms_max_confidence", "y_fire"]].to_dict("records") == [
        {"cell_id": "38.00_-120.00", "firms_n_pixels": 1, "firms_max_confidence": 50.0, "y_fire": 1}
    ]


def test_strict_firms_reader_raises_instead_of_returning_false_zero_day(tmp_path):
    (tmp_path / "2026-08-14.tif").write_bytes(b"not-a-geotiff")
    with pytest.raises(RuntimeError, match="FIRMS read failed"):
        firms_labels.label_day_to_cells(pd.Timestamp("2026-08-14"), str(tmp_path), 30, strict=True)
