from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    for key, val in list(paths.items()):
        if not isinstance(val, str):
            continue
        p = Path(val)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        paths[key] = p

    return cfg


def shared_cache(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["paths"]["shared_cache"])
    p.mkdir(parents=True, exist_ok=True)
    return p
