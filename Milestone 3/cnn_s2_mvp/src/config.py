from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    for key in (
        "mvp_output_dir",
        "california_geojson",
        "output_dir",
        "patches_dir",
        "tiles_dir",
        "manifest",
    ):
        if key not in cfg["paths"]:
            continue
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        cfg["paths"][key] = p
    return cfg
