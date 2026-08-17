#!/usr/bin/env python3
"""Score a verified 06:30 California-time replay panel with the unchanged model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--provenance-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    application_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(application_root / "api"))
    from app.model_registry import ModelRegistry

    panel = pd.read_parquet(args.panel)
    panel["label_date"] = pd.to_datetime(panel["label_date"]).dt.normalize()
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")
    panel = panel.loc[panel["label_date"].between(start, end)].copy()
    if panel.empty or "y_fire" not in panel:
        raise ValueError("Replay panel must contain 2023–2025 rows and y_fire.")

    records = [json.loads(line) for line in args.provenance_jsonl.read_text().splitlines() if line.strip()]
    provenance = {record["labelDate"]: record for record in records}
    replay_days = set(panel["label_date"].dt.date)
    expected_days = set(pd.date_range(start, end, freq="D").date)
    if replay_days != expected_days:
        missing = sorted(expected_days - replay_days)
        raise ValueError(f"Replay panel must cover every day in 2023–2025; missing {missing[:5]}.")
    for day in sorted(replay_days):
        record = provenance.get(day.isoformat())
        if not record:
            raise ValueError(f"Missing cutoff provenance for {day}.")
        if record.get("timezone") != "America/Los_Angeles" or not str(record.get("cutoffAt", "")).startswith(
            (day - pd.Timedelta(days=1)).isoformat()
        ):
            raise ValueError(f"Invalid cutoff provenance for {day}.")
        snapshots = record.get("sourceSnapshots") or {}
        if not snapshots or any(not source.get("ready") for source in snapshots.values()):
            raise ValueError(f"Incomplete source snapshot for {day}.")

    registry = ModelRegistry(args.model)
    scored = registry.score(panel)
    truth = pd.to_numeric(panel["y_fire"], errors="raise").astype(int).to_numpy()
    positives = int(truth.sum())
    captured = int(truth[scored.alert_top_25].sum())
    result = {
        "policy": "06:30 California time causal replay",
        "timezone": "America/Los_Angeles",
        "period": {"start": str(date(2023, 1, 1)), "end": str(date(2025, 12, 31))},
        "modelRetrained": False,
        "rowCount": int(len(panel)),
        "positiveCellDays": positives,
        "prAuc": float(average_precision_score(truth, scored.probability)),
        "recallAt25": float(captured / positives) if positives else None,
        "existingHeldOutReference": {"prAuc": 0.1451, "recallAt25": 0.3638},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
