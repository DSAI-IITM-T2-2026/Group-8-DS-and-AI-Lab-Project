from __future__ import annotations

import logging

import pandas as pd

from ..config import config_hash
from ..data.assemble import attach_eo_year, build_stage_a_year
from ..data.eo import build_eo_month, eo_cache_path
from ..data.era5 import build_era5_month, daily_cache_path
from ..data.firms import build_firms_month, firms_cache_path
from ..data.synthetic import build_synthetic_inputs
from ..dataset import write_splits
from ..io import claim_shard, write_shard_manifest
from .training import evaluate_router, train_fire_season, train_month_models
from .verify import verify_gcs

logger = logging.getLogger(__name__)

STAGES = [
    "verify_gcs",
    "era5_firms",
    "stage_a",
    "s2_cell",
    "s2_attach",
    "s5p_cell",
    "s5p_attach",
    "write_splits",
    "train_fire_season",
    "train_month_models",
    "eval_figures",
]


def demo_config(cfg: dict) -> dict:
    workspace = cfg["paths"]["workspace"]
    root = workspace / "outputs" / "demo"
    cfg["paths"]["dem_cells"] = root / "data" / "era5_grid_dem_features.parquet"
    cfg["paths"]["cache_dir"] = root / "cache"
    cfg["paths"]["dataset_dir"] = root / "datasets"
    cfg["paths"]["manifest_dir"] = root / "manifests"
    cfg["paths"]["report_dir"] = root / "reports"
    cfg["paths"]["artifact_dir"] = workspace / "artifacts" / "demo"
    cfg["training"]["num_boost_round"] = min(int(cfg["training"]["num_boost_round"]), 120)
    cfg["training"]["early_stopping_rounds"] = min(
        int(cfg["training"]["early_stopping_rounds"]), 15
    )
    cfg["training"]["run_hyperparameter_sweep"] = False
    cfg["training"]["run_mlp"] = True
    cfg["model_buckets"]["min_train_positives"] = 2
    cfg["model_buckets"]["min_calibration_positives"] = 1
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def run_pipeline(
    cfg: dict,
    stage: str,
    years: list[int],
    months: list[int],
    worker: str,
    lag: int,
    force: bool = False,
) -> None:
    if stage == "all":
        for current in STAGES:
            run_pipeline(cfg, current, years, months, worker, lag, force)
        return
    if stage == "demo":
        cfg = demo_config(cfg)
        build_synthetic_inputs(cfg, seed=int(cfg["project"]["random_seed"]))
        for current in (
            "stage_a",
            "s2_attach",
            "s5p_attach",
            "write_splits",
            "train_fire_season",
            "train_month_models",
            "eval_figures",
        ):
            run_pipeline(cfg, current, years, months, worker, lag, True)
        return
    if stage == "verify_gcs":
        verify_gcs(cfg, years, months)
        return
    if stage == "era5_firms":
        for year in years:
            # The January rolling window needs December from the preceding year.
            _build_source_shard(cfg, "era5", year - 1, 12, worker, False, build_era5_month)
            for month in months:
                _build_source_shard(cfg, "era5", year, month, worker, force, build_era5_month)
                _build_source_shard(cfg, "firms", year, month, worker, force, build_firms_month)
        return
    if stage in {"s2_cell", "s5p_cell"}:
        source = stage.split("_", 1)[0]
        for year in years:
            if source == "s2" and str(year - 1) in cfg["gcs"]["s2_prefix_by_year"]:
                _build_eo_shard(cfg, source, year - 1, 12, worker, False)
            for month in months:
                _build_eo_shard(cfg, source, year, month, worker, force)
        return
    if stage == "stage_a":
        for year in years:
            build_stage_a_year(cfg, year, lag, force)
        return
    if stage == "s2_attach":
        for year in years:
            attach_eo_year(cfg, "s2", year, lag, force)
        return
    if stage == "s5p_attach":
        for year in years:
            attach_eo_year(cfg, "s5p", year, lag, force)
        return
    if stage == "write_splits":
        for model_stage in ("A", "B", "C"):
            write_splits(cfg, model_stage, lag, force)
        return
    if stage == "train_fire_season":
        train_fire_season(cfg, lag, force)
        return
    if stage == "train_month_models":
        train_month_models(cfg, lag, force)
        return
    if stage == "eval_figures":
        evaluate_router(cfg, lag)
        return
    raise ValueError(f"Unknown stage {stage!r}; choose one of {STAGES + ['all', 'demo']}")


def _build_source_shard(
    cfg: dict,
    source: str,
    year: int,
    month: int,
    worker: str,
    force: bool,
    builder,
) -> None:
    output = (
        daily_cache_path(cfg, year, month)
        if source == "era5"
        else firms_cache_path(cfg, year, month)
    )
    claim = cfg["paths"]["manifest_dir"] / "claims" / f"{source}_{year}_{month:02d}.json"
    if output.exists() and not force:
        return
    with claim_shard(
        claim,
        worker,
        ttl_hours=float(cfg["execution"]["claim_ttl_hours"]),
        force=force,
    ):
        builder(cfg, year, month, force)
        frame = pd.read_parquet(output)
        write_shard_manifest(
            output,
            {
                "source": source,
                "year": year,
                "month": month,
                "worker": worker,
                "rows": len(frame),
            },
            cfg["_config_hash"],
            cfg["paths"]["manifest_dir"] / source,
        )


def _build_eo_shard(
    cfg: dict, source: str, year: int, month: int, worker: str, force: bool
) -> None:
    output = eo_cache_path(cfg, source, year, month)
    if output.exists() and not force:
        return
    claim = cfg["paths"]["manifest_dir"] / "claims" / f"{source}_{year}_{month:02d}.json"
    with claim_shard(
        claim,
        worker,
        ttl_hours=float(cfg["execution"]["claim_ttl_hours"]),
        force=force,
    ):
        build_eo_month(cfg, source, year, month, force)
        frame = pd.read_parquet(output)
        write_shard_manifest(
            output,
            {
                "source": source,
                "year": year,
                "month": month,
                "worker": worker,
                "rows": len(frame),
            },
            cfg["_config_hash"],
            cfg["paths"]["manifest_dir"] / source,
        )
