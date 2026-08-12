"""Export champion 86-feature inference-ready parquet for one label date."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import load_feature_contract
from paths import resolve_path
from preprocess.build_champion_features import (
    apply_champion_prune,
    build_features,
    infer_base_features,
    load_cell_subset,
)
from preprocess.adapters_gcs import upload_parquet

logger = logging.getLogger("preprocess.export_inference_day")

ID_COLUMNS = [
    "cell_id", "label_date", "eo_asof_date", "feature_end_date",
    "latitude", "longitude", "y_fire",
]


def _assert_inference_contract(
    pruned: pd.DataFrame,
    label_date: date,
    feature_cols: list[str],
    *,
    era5_lag: int = 5,
    lead: int = 1,
) -> None:
    """Cheap checks that match Milestone 5 prepared_champion_day expectations."""
    if pruned.empty:
        raise ValueError(f"Empty export for label_date={label_date}")
    if len(feature_cols) != 86:
        raise ValueError(f"Expected 86 champion features, got {len(feature_cols)}")
    missing = [c for c in feature_cols if c not in pruned.columns]
    if missing:
        raise ValueError(f"Pruned frame missing features: {missing[:5]}")

    labels = pd.to_datetime(pruned["label_date"]).dt.normalize().unique()
    if len(labels) != 1 or pd.Timestamp(labels[0]).date() != label_date:
        raise ValueError(f"Expected single label_date={label_date}, got {labels}")

    eo = pd.to_datetime(pruned["eo_asof_date"]).dt.normalize()
    feat = pd.to_datetime(pruned["feature_end_date"]).dt.normalize()
    expected_eo = pd.Timestamp(label_date) - pd.Timedelta(days=1)
    expected_feat = pd.Timestamp(label_date) - pd.Timedelta(days=era5_lag + lead)
    if not eo.eq(expected_eo).all():
        raise ValueError(f"eo_asof_date must be {expected_eo.date()} (D−1)")
    if not feat.eq(expected_feat).all():
        raise ValueError(f"feature_end_date must be {expected_feat.date()} (D−6)")
    if not (eo - feat).dt.days.eq(era5_lag).all():
        raise ValueError(f"eo_asof − feature_end must equal era5_lag={era5_lag}")

    n_cells = pruned["cell_id"].nunique()
    if n_cells < 100:
        logger.warning("Only %d cells in export (expected ~437 high/medium)", n_cells)
    else:
        logger.info("Export contract OK: %d cells × 86 features for %s", n_cells, label_date)


def export_champion_day(
    label_date: date,
    history: pd.DataFrame,
    daily_cfg: dict,
    *,
    upload: bool = True,
) -> tuple[pd.DataFrame, str]:
    contract = load_feature_contract(daily_cfg)
    kept = contract["feature_prune"]["kept_features"]
    use_neighbor = contract.get("use_neighbor_fire_features", True)
    lag = int(daily_cfg["task"].get("era5_lag_days", 5))
    lead = int(daily_cfg["task"].get("lead_days", 1))

    hist = history.copy()
    hist["label_date"] = pd.to_datetime(hist["label_date"]).dt.normalize()
    day_ts = pd.Timestamp(label_date)

    # Live prediction: y_fire on D is not a model input. Keep column for schema;
    # use 0 when missing / for forecast (M5 next_day does the same).
    if "y_fire" not in hist.columns:
        hist["y_fire"] = np.int8(0)
    day_mask = hist["label_date"].eq(day_ts)
    if day_mask.any():
        # Preserve real historical labels if present and non-null for demo metrics;
        # still coerce NaN → 0 so scoring never blocks on FIRMS for D.
        hist.loc[day_mask, "y_fire"] = (
            pd.to_numeric(hist.loc[day_mask, "y_fire"], errors="coerce").fillna(0).astype("int8")
        )
        if hist.loc[day_mask, "y_fire"].sum() == 0:
            logger.info(
                "y_fire on %s is all-zero (live forecast placeholder or no FIRMS that day)",
                label_date,
            )

    base = infer_base_features(hist.columns.tolist())
    engineered, all_features, groups = build_features(
        hist,
        base,
        use_neighbor_fire_features=use_neighbor,
    )
    logger.info("Engineered %d features (groups=%s)", len(all_features), groups)

    if len(all_features) != contract["feature_prune"]["n_before"]:
        logger.warning(
            "Feature count %d != expected %d (continuing with locked prune list)",
            len(all_features),
            contract["feature_prune"]["n_before"],
        )

    day_frame = engineered.loc[engineered["label_date"].eq(day_ts)].copy()
    if day_frame.empty:
        raise ValueError(f"No rows for label_date={label_date} in history panel")

    # Ensure date IDs match M4 (in case Stage C omitted them).
    day_frame["label_date"] = day_ts
    day_frame["eo_asof_date"] = day_ts - pd.Timedelta(days=1)
    day_frame["feature_end_date"] = day_ts - pd.Timedelta(days=lag + lead)
    day_frame["y_fire"] = (
        pd.to_numeric(day_frame.get("y_fire", 0), errors="coerce").fillna(0).astype("int8")
    )

    subset_path = resolve_path(daily_cfg, "fire_region_csv")
    subset_mode = daily_cfg.get("preprocess", {}).get("cell_subset", "high_medium_fire")
    cell_ids = load_cell_subset(str(subset_path), mode=subset_mode)
    if cell_ids is not None:
        before = len(day_frame)
        day_frame = day_frame.loc[day_frame["cell_id"].astype(str).isin(cell_ids)].copy()
        logger.info(
            "Cell subset mode=%s → %d / %d rows for %s",
            subset_mode,
            len(day_frame),
            before,
            label_date,
        )
    else:
        logger.info("Cell subset mode=%s → all cells (%d rows)", subset_mode, len(day_frame))

    pruned, feature_cols = apply_champion_prune(day_frame, all_features, kept)
    _assert_inference_contract(
        pruned, label_date, feature_cols, era5_lag=lag, lead=lead
    )

    cache = resolve_path(daily_cfg, "local_cache")
    out_name = f"{label_date.isoformat()}_test.parquet"
    local_dir = cache / "final_processed"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / out_name
    pruned.to_parquet(local_path, index=False)

    contract_path = local_dir / "feature_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    manifest = local_dir / "manifest.jsonl"
    entry = {
        "label_date": label_date.isoformat(),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(pruned),
        "n_features": len(feature_cols),
        "n_cells": int(pruned["cell_id"].nunique()),
        "eo_asof_date": (label_date - timedelta(days=1)).isoformat(),
        "feature_end_date": (label_date - timedelta(days=lag + lead)).isoformat(),
        "local_path": str(local_path),
        "gcs_object": f"final_processed/{out_name}",
    }
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    gcs_uri = ""
    if upload:
        bucket = daily_cfg["gcs"]["bucket"]
        prefix = daily_cfg["gcs"]["prefixes"]["final_processed"]
        gcs_uri = f"gs://{bucket}/{prefix}/{out_name}"
        upload_parquet(pruned, gcs_uri)
        from google.cloud import storage

        client = storage.Client()
        client.bucket(bucket).blob(f"{prefix}/feature_contract.json").upload_from_string(
            json.dumps(contract, indent=2), content_type="application/json"
        )
        logger.info("Uploaded final processed → %s", gcs_uri)

    return pruned, gcs_uri or str(local_path)
