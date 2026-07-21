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


def extract_patch(
    path: str,
    lon: float,
    lat: float,
    size: int = 64,
    bands: int = 6,
) -> np.ndarray | None:
    """Open path (local or /vsigs/), extract one patch, close."""
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    try:
        with rasterio.open(path) as src:
            return extract_patch_from_dataset(src, lon, lat, size=size, bands=bands)
    except Exception as exc:
        logger.warning("Patch open failed %s: %s", path, exc)
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
