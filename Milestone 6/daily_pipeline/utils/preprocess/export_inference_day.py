"""Export champion 86-feature inference-ready parquet for one label date."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

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

    base = infer_base_features(history.columns.tolist())
    engineered, all_features, groups = build_features(
        history,
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

    day_ts = pd.Timestamp(label_date)
    day_frame = engineered.loc[engineered["label_date"].eq(day_ts)].copy()
    if day_frame.empty:
        raise ValueError(f"No rows for label_date={label_date} in history panel")

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
