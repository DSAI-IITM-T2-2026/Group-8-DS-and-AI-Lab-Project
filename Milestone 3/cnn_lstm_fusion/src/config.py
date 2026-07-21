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
        "sequences_dir",
        "tiles_dir",
        "s5p_tiles_dir",
        "manifest",
    ):
        if key not in cfg.get("paths", {}):
            continue
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        cfg["paths"][key] = p
    return cfg


def use_sentinel5p(cfg: dict[str, Any], cli_force: bool | None = None) -> bool:
    """Resolve S5P toggle: CLI overrides config."""
    if cli_force is not None:
        return bool(cli_force)
    return bool(cfg.get("sources", {}).get("sentinel5p", {}).get("enabled", False))
