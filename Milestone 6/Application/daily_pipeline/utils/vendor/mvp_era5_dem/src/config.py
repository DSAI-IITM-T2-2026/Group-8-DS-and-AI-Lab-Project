from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    for key in ("dem_cells", "output_dir", "cache_dir"):
        p = Path(paths[key])
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        paths[key] = p

    return cfg
