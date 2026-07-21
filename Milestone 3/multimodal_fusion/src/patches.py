from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def extract_patch_from_dataset(
    src: rasterio.DatasetReader,
    lon: float,
    lat: float,
    size: int = 64,
    bands: int = 6,
) -> np.ndarray | None:
    """Windowed read from an already-open dataset."""
    try:
        row, col = src.index(lon, lat)
        half = size // 2
        r0 = int(row) - half
        c0 = int(col) - half
        r0 = max(0, min(r0, src.height - size))
        c0 = max(0, min(c0, src.width - size))
        if src.height < size or src.width < size:
            return None
        window = Window(c0, r0, size, size)
        data = src.read(indexes=list(range(1, bands + 1)), window=window)
        data = np.asarray(data, dtype=np.float32)
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        logger.warning("Patch extract failed @ (%.3f,%.3f): %s", lon, lat, exc)
        return None


def sample_point_from_dataset(
    src: rasterio.DatasetReader,
    lon: float,
    lat: float,
    band: int = 1,
) -> float | None:
    """Read a single-pixel value from an already-open raster."""
    try:
        row, col = src.index(lon, lat)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return None
        val = src.read(band, window=Window(int(col), int(row), 1, 1))
        v = float(np.asarray(val).ravel()[0])
        if not np.isfinite(v):
            return None
        return v
    except Exception as exc:
        logger.warning("Point sample failed @ (%.3f,%.3f): %s", lon, lat, exc)
        return None


class TileHandleCache:
    """Keep one rasterio dataset open at a time; reopen only when path changes."""

    def __init__(self):
        self._path: str | None = None
        self._src: rasterio.DatasetReader | None = None

    def get(self, path: str) -> rasterio.DatasetReader:
        os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
        if self._src is not None and self._path == path:
            return self._src
        self.close()
        self._src = rasterio.open(path)
        self._path = path
        return self._src

    def close(self) -> None:
        if self._src is not None:
            self._src.close()
            self._src = None
            self._path = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def save_patch(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
