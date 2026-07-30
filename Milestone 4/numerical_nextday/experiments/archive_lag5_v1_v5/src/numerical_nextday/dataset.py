from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .contracts import feature_columns, validate_samples
from .data.assemble import stage_path
from .io import atomic_json, atomic_parquet


def split_dir(cfg: dict, stage: str, lag: int) -> Path:
    return cfg["paths"]["dataset_dir"] / "splits" / f"stage_{stage.lower()}_lag{lag}"


def split_path(cfg: dict, stage: str, lag: int, split: str) -> Path:
    return split_dir(cfg, stage, lag) / f"{split}.parquet"


def stage_key(stage: str, lag: int) -> str:
    return f"{stage.lower()}_lag{lag}"


def write_splits(cfg: dict, stage: str, lag: int, force: bool = False) -> dict[str, Path]:
    destination_dir = split_dir(cfg, stage, lag)
    outputs = {
        name: split_path(cfg, stage, lag, name) for name in ("train", "tune", "calibration", "test")
    }
    if all(path.exists() for path in outputs.values()) and not force:
        return outputs
    years = sorted(
        {
            int(year)
            for key in ("train_years", "tune_years", "calibration_years", "test_years")
            for year in cfg["temporal"][key]
        }
    )
    paths = [stage_path(cfg, stage_key(stage, lag), year) for year in years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing stage-year tables:\n" + "\n".join(missing))
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    for column in ("feature_end_date", "era5_source_date", "label_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    validate_samples(frame, int(cfg["task"]["lead_days"]), lag)
    split_year_mapping = {
        "train": {int(year) for year in cfg["temporal"]["train_years"]},
        "tune": {int(year) for year in cfg["temporal"]["tune_years"]},
        "calibration": {int(year) for year in cfg["temporal"]["calibration_years"]},
        "test": {int(year) for year in cfg["temporal"]["test_years"]},
    }
    destination_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, years_for_split in split_year_mapping.items():
        part = frame.loc[frame["label_date"].dt.year.isin(years_for_split)].copy()
        atomic_parquet(part, outputs[name])
        summaries[name] = {
            "rows": len(part),
            "positives": int(part["y_fire"].sum()),
            "years": sorted(years_for_split),
        }
    features = feature_columns(frame, stage)
    atomic_json(features, destination_dir / "feature_columns.json")
    atomic_json(
        {
            "stage": stage.upper(),
            "era5_lag_days": lag,
            "target": cfg["task"]["target"],
            "label_definition": cfg["task"]["label_definition"],
            "unit": "era5_0.25_degree_cell_x_day",
            "features": features,
            "splits": summaries,
            "config_hash": cfg["_config_hash"],
        },
        destination_dir / "dataset_metadata.json",
    )
    return outputs


def load_splits(cfg: dict, stage: str, lag: int) -> dict[str, pd.DataFrame]:
    result = {}
    for name in ("train", "tune", "calibration", "test"):
        path = split_path(cfg, stage, lag, name)
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = pd.read_parquet(path)
    return result


def load_feature_columns(cfg: dict, stage: str, lag: int) -> list[str]:
    path = split_dir(cfg, stage, lag) / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
