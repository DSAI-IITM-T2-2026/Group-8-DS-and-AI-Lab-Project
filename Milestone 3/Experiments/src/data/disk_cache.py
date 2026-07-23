"""Durable on-disk caches so rebuilds skip GCS when files already exist."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_arrays(path: Path) -> Optional[dict[str, np.ndarray]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr))


def load_array(path: Path) -> Optional[np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return np.load(path, allow_pickle=False)
