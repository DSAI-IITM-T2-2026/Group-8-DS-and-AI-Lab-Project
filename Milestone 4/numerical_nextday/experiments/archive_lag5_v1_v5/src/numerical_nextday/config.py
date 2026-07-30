from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config.yaml"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)

    root = config_path.parent
    paths = cfg["paths"]
    workspace = Path(paths.get("workspace", "."))
    if not workspace.is_absolute():
        workspace = (root / workspace).resolve()
    paths["workspace"] = workspace
    for key in (
        "dem_cells",
        "cache_dir",
        "dataset_dir",
        "artifact_dir",
        "manifest_dir",
        "report_dir",
    ):
        value = Path(paths[key])
        paths[key] = value if value.is_absolute() else (workspace / value).resolve()
    for key in ("era5_daily_seed_dirs", "firms_seed_dirs"):
        paths[key] = [
            Path(p) if Path(p).is_absolute() else (workspace / p).resolve()
            for p in paths.get(key, [])
        ]

    validate_config(cfg)
    cfg["_config_path"] = config_path
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    start = str(cfg["temporal"]["start_date"])
    end = str(cfg["temporal"]["end_date"])
    if start > end:
        raise ValueError("temporal.start_date must not be after temporal.end_date")
    split_sets = [
        set(cfg["temporal"][name])
        for name in ("train_years", "tune_years", "calibration_years", "test_years")
    ]
    for i, left in enumerate(split_sets):
        for right in split_sets[i + 1 :]:
            if left & right:
                raise ValueError(f"Temporal split years overlap: {sorted(left & right)}")
    month_values = []
    for values in cfg["model_buckets"].values():
        if isinstance(values, list):
            month_values.extend(values)
    if sorted(month_values) != list(range(1, 13)):
        raise ValueError("model_buckets must assign every calendar month exactly once")
    if int(cfg["task"]["lead_days"]) < 1:
        raise ValueError("task.lead_days must be positive")
    if int(cfg["task"]["era5_lag_days"]) < 0:
        raise ValueError("task.era5_lag_days cannot be negative")
    if cfg["execution"].get("mode", "single_machine") != "single_machine":
        raise ValueError("Only execution.mode=single_machine is currently supported")
    if int(cfg["execution"]["max_workers"]) < 1:
        raise ValueError("execution.max_workers must be positive")
    if int(cfg["execution"].get("eo_parallel_streams", 1)) < 1:
        raise ValueError("execution.eo_parallel_streams must be positive")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def config_hash(cfg: dict) -> str:
    payload = json.dumps(_jsonable(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def bucket_for_month(cfg: dict, month: int) -> str:
    for bucket, months in cfg["model_buckets"].items():
        if isinstance(months, list) and month in months:
            return bucket
    raise ValueError(f"No model bucket configured for month={month}")


def years_for_split(cfg: dict, split: str) -> list[int]:
    mapping = {
        "train": "train_years",
        "tune": "tune_years",
        "calibration": "calibration_years",
        "test": "test_years",
    }
    return [int(y) for y in cfg["temporal"][mapping[split]]]
