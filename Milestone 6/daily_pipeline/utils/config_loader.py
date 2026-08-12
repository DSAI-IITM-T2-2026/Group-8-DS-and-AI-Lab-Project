"""Load daily pipeline + M4 merged config."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from paths import pipeline_root, resolve_path, utils_root


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_daily_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (utils_root() / "config.yaml")
    cfg = load_yaml(path)
    cfg["_pipeline_root"] = str(pipeline_root())
    cfg["_utils_root"] = str(utils_root())
    return cfg


def load_feature_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(cfg, "contracts")
    return json.loads(path.read_text(encoding="utf-8"))


def load_m4_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge vendored M4 config with daily GCS overrides."""
    m4_path = resolve_path(cfg, "m4_config")
    m4 = load_yaml(m4_path)
    m4 = copy.deepcopy(m4)

    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    era5_primary = f"gs://{bucket}/{prefixes['era5']}"
    era5_legacy = (
        f"gs://{cfg['gcs']['era5_legacy_bucket']}/{cfg['gcs']['era5_legacy_prefix']}"
    )

    m4.setdefault("gcs", {})
    m4["gcs"]["era5_prefix"] = era5_primary
    m4["gcs"]["era5_legacy_prefix"] = era5_legacy
    m4["gcs"]["firms_prefix"] = f"gs://{bucket}/{prefixes['firms']}"
    m4["gcs"]["firms_vsigs_prefix"] = cfg["gcs"]["firms_vsigs_prefix"]
    m4["gcs"]["daily_bucket"] = bucket
    m4["gcs"]["flat_s2_prefix"] = prefixes["sentinel2"]
    m4["gcs"]["flat_s5p_prefix"] = prefixes["sentinel5p"]

    flat_s2 = {"bucket": bucket, "prefix": prefixes["sentinel2"], "layout": "flat_parquet"}
    flat_s5p = {"bucket": bucket, "prefix": prefixes["sentinel5p"], "layout": "flat_parquet"}
    for year in range(2018, 2031):
        m4["gcs"]["s2_by_year"][str(year)] = flat_s2
        if str(year) in m4["gcs"].get("s5p_by_year", {}):
            m4["gcs"]["s5p_by_year"][str(year)] = flat_s5p

    cache = resolve_path(cfg, "local_cache") / "m4_shared_cache"
    m4["paths"]["shared_cache"] = str(cache)
    m4["paths"]["output_dir"] = str(cache.parent)
    m4["paths"]["grid_map"] = str(cache / "era5_to_feature_grid.parquet")
    m4["paths"]["dem_cells"] = str(resolve_path(cfg, "dem_local"))
    m4["paths"]["mvp_era5_dem_root"] = str(resolve_path(cfg, "mvp_era5_dem_root"))
    m4["paths"]["multimodal_fusion_root"] = str(resolve_path(cfg, "multimodal_fusion_root"))
    m4["task"]["era5_lag_days"] = cfg["task"]["era5_lag_days"]
    m4["task"]["lookback_days"] = cfg["task"].get("lookback_days", 45)
    return m4


def setup_m4_imports(cfg: dict[str, Any]) -> Path:
    """Return vendored M4 src path (bootstrap already added it to sys.path)."""
    m4_root = resolve_path(cfg, "m4_root")
    src = m4_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(
            f"Vendored M4 src not found: {src}. Expected utils/vendor/numerical_nextday/src"
        )
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src
