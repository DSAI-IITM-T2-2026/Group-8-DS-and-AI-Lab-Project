"""Dataset metadata and summary generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

from pipeline.config import PipelineConfig
from pipeline.validator import ValidationResult

logger = logging.getLogger(__name__)


def generate_metadata(
    config: PipelineConfig,
    validation_results: list[ValidationResult],
    download_stats: dict[str, int],
) -> dict:
    processed_files = sorted(config.paths["processed"].glob("*.nc"))
    merged_files = sorted(config.paths["merged"].glob("*.nc"))
    raw_files = sorted(config.paths["raw"].glob("*.nc"))

    spatial = {}
    temporal = {}

    if processed_files:
        with xr.open_dataset(processed_files[0]) as ds:
            lat_dim = "latitude" if "latitude" in ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds.dims else "lon"
            time_dim = "valid_time" if "valid_time" in ds.dims else "time"

            spatial = {
                "lat_min": float(ds[lat_dim].min()),
                "lat_max": float(ds[lat_dim].max()),
                "lon_min": float(ds[lon_dim].min()),
                "lon_max": float(ds[lon_dim].max()),
                "lat_points": int(ds.sizes[lat_dim]),
                "lon_points": int(ds.sizes[lon_dim]),
            }

            all_times = []
            for pf in processed_files:
                with xr.open_dataset(pf) as pds:
                    tdim = "valid_time" if "valid_time" in pds.dims else "time"
                    all_times.extend(pd.to_datetime(pds[tdim].values).tolist())

            if all_times:
                temporal = {
                    "start": min(all_times).isoformat(),
                    "end": max(all_times).isoformat(),
                    "total_hours": len(all_times),
                }

    variable_inventory = list(config.variables) + [
        "relative_humidity",
        "wind_speed",
        "wind_direction",
        f"rainfall_{config.rainfall_aggregation_hours}h",
        "soil_moisture_index",
    ]

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset,
        "study_area": {
            "north": config.area[0],
            "west": config.area[1],
            "south": config.area[2],
            "east": config.area[3],
        },
        "years": config.years,
        "variables": variable_inventory,
        "spatial_coverage": spatial,
        "temporal_coverage": temporal,
        "file_counts": {
            "raw_monthly": len(raw_files),
            "merged_yearly": len(merged_files),
            "processed_yearly": len(processed_files),
        },
        "download_stats": download_stats,
        "validation": {
            "files_checked": len(validation_results),
            "valid": sum(1 for r in validation_results if r.valid),
            "invalid": sum(1 for r in validation_results if not r.valid),
        },
    }

    config.metadata_file().parent.mkdir(parents=True, exist_ok=True)
    with open(config.metadata_file(), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Wrote metadata: %s", config.metadata_file())
    return metadata


def generate_summary_csv(config: PipelineConfig, validation_results: list[ValidationResult]) -> Path:
    rows = []

    for result in validation_results:
        rows.append({
            "file": Path(result.file).name,
            "valid": result.valid,
            "time_steps": result.time_steps,
            "variables": len(result.variables_found),
            "missing_fraction": round(result.missing_fraction, 6),
            "errors": "; ".join(result.errors),
            "warnings": "; ".join(result.warnings),
        })

    for year in config.years:
        merged = config.merged_file(year)
        processed = config.processed_file(year)
        if merged.exists():
            rows.append({
                "file": merged.name,
                "valid": True,
                "time_steps": "",
                "variables": "",
                "missing_fraction": "",
                "errors": "",
                "warnings": "merged yearly file",
            })
        if processed.exists():
            rows.append({
                "file": processed.name,
                "valid": True,
                "time_steps": "",
                "variables": "",
                "missing_fraction": "",
                "errors": "",
                "warnings": "processed yearly file with engineered features",
            })

    df = pd.DataFrame(rows)
    config.summary_csv().parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.summary_csv(), index=False)
    logger.info("Wrote summary CSV: %s", config.summary_csv())
    return config.summary_csv()
