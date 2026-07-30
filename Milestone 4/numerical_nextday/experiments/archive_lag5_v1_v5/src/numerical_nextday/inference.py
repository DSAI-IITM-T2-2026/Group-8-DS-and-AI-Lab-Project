from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .train.lightgbm_model import LightGBMBundle
from .train.router import MonthRouter


def score_parquet(routing_path: Path, input_path: Path, output_path: Path) -> Path:
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    models = {
        bucket: LightGBMBundle.load(Path(entry["model_path"]))
        for bucket, entry in routing["models"].items()
    }
    frame = pd.read_parquet(input_path).reset_index(drop=True)
    frame["p_fire"] = MonthRouter(
        models, routing.get("fallback_bucket", "fire_season")
    ).predict_proba(frame)
    frame["confidence_pct"] = 100 * frame["p_fire"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path
