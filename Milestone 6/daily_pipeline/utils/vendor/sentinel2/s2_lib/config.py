"""Load and validate YAML configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_COMPOSITES = frozenset({"median", "mean", "mosaic", "first"})
REQUIRED_SECTIONS = ("aoi", "temporal", "grid", "sentinel2", "export", "scheduler")
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


@dataclass(frozen=True)
class AOIConfig:
    geojson_path: Path
    north: float
    south: float
    west: float
    east: float
    error_margin_m: float


@dataclass(frozen=True)
class TemporalConfig:
    start_year: int
    end_year: int
    window_days: int


@dataclass(frozen=True)
class GridConfig:
    resolution_m: float
    crs: str
    asset_id: str
    # Spatial chunk size for reduceRegions (avoids "Computed value is too large")
    reduce_tile_m: float


@dataclass(frozen=True)
class Sentinel2Config:
    collection: str
    cloud_probability_collection: str
    bands: list[str]
    cloud_filter_pct: float
    cloud_probability_threshold: float
    composite: str
    evi_min_abs_denominator: float
    evi_max_abs: float
    reduce_scale_m: float
    tile_scale: int


@dataclass(frozen=True)
class ExportConfig:
    bucket: str
    prefix: str
    gee_format: str
    format: str
    drop_empty_cells: bool


@dataclass(frozen=True)
class SchedulerConfig:
    max_running_tasks: int
    poll_interval_seconds: int
    retry_attempts: int
    state_db: str
    log_dir: str


@dataclass(frozen=True)
class AppConfig:
    project_id: str
    aoi: AOIConfig
    temporal: TemporalConfig
    grid: GridConfig
    sentinel2: Sentinel2Config
    export: ExportConfig
    scheduler: SchedulerConfig
    config_path: Path
    project_root: Path


def _require(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key '{section_name}.{key}' in config")
    return section[key]


def _geojson_coordinate_pairs(value: Any):
    """Yield coordinate pairs from arbitrarily nested GeoJSON coordinates."""
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, list):
        for child in value:
            yield from _geojson_coordinate_pairs(child)


def _load_geojson_bounds(path: Path) -> tuple[float, float, float, float]:
    try:
        with path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid AOI GeoJSON '{path}': {exc}") from exc

    if payload.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") for f in payload.get("features", [])]
    elif payload.get("type") == "Feature":
        geometries = [payload.get("geometry")]
    else:
        geometries = [payload]

    pairs: list[tuple[float, float]] = []
    for geometry in geometries:
        if isinstance(geometry, dict):
            pairs.extend(_geojson_coordinate_pairs(geometry.get("coordinates")))
    if not pairs:
        raise ValueError(f"AOI GeoJSON contains no polygon coordinates: {path}")

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return min(xs), min(ys), max(xs), max(ys)


def load_config(path: str | Path | None = None) -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(path) if path else project_root / "config" / "config.yaml"
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(f"Missing required config section: {section}")

    gee = raw.get("gee") or {}
    project_id = gee.get("project_id") or raw.get("project_id")
    if not project_id:
        raise ValueError("Missing required key 'gee.project_id'")

    aoi_raw = raw["aoi"]
    geojson_value = str(_require(aoi_raw, "geojson_path", "aoi"))
    geojson_path = Path(geojson_value)
    if not geojson_path.is_absolute():
        geojson_path = (project_root / geojson_path).resolve()
    if not geojson_path.is_file():
        raise FileNotFoundError(f"AOI GeoJSON not found: {geojson_path}")
    west, south, east, north = _load_geojson_bounds(geojson_path)
    aoi = AOIConfig(
        geojson_path=geojson_path,
        north=north,
        south=south,
        west=west,
        east=east,
        error_margin_m=float(aoi_raw.get("error_margin_m", 100)),
    )
    if aoi.error_margin_m <= 0:
        raise ValueError("aoi.error_margin_m must be > 0")

    temporal_raw = raw["temporal"]
    temporal = TemporalConfig(
        start_year=int(_require(temporal_raw, "start_year", "temporal")),
        end_year=int(_require(temporal_raw, "end_year", "temporal")),
        window_days=int(_require(temporal_raw, "window_days", "temporal")),
    )
    if temporal.start_year > temporal.end_year or temporal.window_days < 1:
        raise ValueError("Invalid temporal configuration")
    if temporal.start_year < 2018:
        raise ValueError(
            "COPERNICUS/S2_SR_HARMONIZED has incomplete/no Level-2A coverage "
            "before 2018; temporal.start_year must be >= 2018 for a "
            "quality-consistent training dataset"
        )

    grid_raw = raw["grid"]
    grid = GridConfig(
        resolution_m=float(_require(grid_raw, "resolution_m", "grid")),
        crs=str(grid_raw.get("crs", "EPSG:3310")),
        asset_id=str(grid_raw.get("asset_id") or ""),
        reduce_tile_m=float(grid_raw.get("reduce_tile_m", 50_000)),
    )
    if grid.resolution_m <= 0:
        raise ValueError("grid.resolution_m must be > 0")
    if grid.reduce_tile_m < grid.resolution_m:
        raise ValueError("grid.reduce_tile_m must be >= grid.resolution_m")

    s2_raw = raw["sentinel2"]
    composite = str(_require(s2_raw, "composite", "sentinel2")).lower()
    if composite not in SUPPORTED_COMPOSITES:
        raise ValueError(f"Unsupported composite '{composite}'")
    bands = list(_require(s2_raw, "bands", "sentinel2"))
    if not bands:
        raise ValueError("sentinel2.bands must be non-empty")
    collection = str(_require(s2_raw, "collection", "sentinel2"))
    if collection != S2_COLLECTION:
        raise ValueError(f"Only {S2_COLLECTION} is supported")

    sentinel2 = Sentinel2Config(
        collection=collection,
        cloud_probability_collection=str(
            s2_raw.get(
                "cloud_probability_collection",
                "COPERNICUS/S2_CLOUD_PROBABILITY",
            )
        ),
        bands=bands,
        cloud_filter_pct=float(_require(s2_raw, "cloud_filter_pct", "sentinel2")),
        cloud_probability_threshold=float(
            s2_raw.get("cloud_probability_threshold", 65)
        ),
        composite=composite,
        evi_min_abs_denominator=float(
            s2_raw.get("evi_min_abs_denominator", 0.05)
        ),
        evi_max_abs=float(s2_raw.get("evi_max_abs", 2.0)),
        reduce_scale_m=float(s2_raw.get("reduce_scale_m", 100)),
        tile_scale=int(s2_raw.get("tile_scale", 16)),
    )
    if not 0 < sentinel2.cloud_filter_pct <= 100:
        raise ValueError("sentinel2.cloud_filter_pct must be in (0, 100]")
    if not 0 < sentinel2.cloud_probability_threshold <= 100:
        raise ValueError(
            "sentinel2.cloud_probability_threshold must be in (0, 100]"
        )
    if sentinel2.evi_min_abs_denominator <= 0 or sentinel2.evi_max_abs <= 0:
        raise ValueError("Sentinel-2 EVI stability thresholds must be > 0")
    if sentinel2.reduce_scale_m <= 0:
        raise ValueError("sentinel2.reduce_scale_m must be > 0")
    if not 1 <= sentinel2.tile_scale <= 16:
        raise ValueError("sentinel2.tile_scale must be between 1 and 16")

    export_raw = raw["export"]
    export = ExportConfig(
        bucket=str(_require(export_raw, "bucket", "export")),
        prefix=str(_require(export_raw, "prefix", "export")).rstrip("/"),
        gee_format=str(export_raw.get("gee_format", "CSV")).upper(),
        format=str(export_raw.get("format", "parquet")).lower(),
        drop_empty_cells=bool(export_raw.get("drop_empty_cells", True)),
    )
    if export.format not in {"parquet", "csv"}:
        raise ValueError("export.format must be 'parquet' or 'csv'")
    if export.drop_empty_cells:
        raise ValueError(
            "export.drop_empty_cells must be false so every window retains "
            "the complete California grid"
        )

    sched_raw = raw["scheduler"]
    scheduler = SchedulerConfig(
        max_running_tasks=int(_require(sched_raw, "max_running_tasks", "scheduler")),
        poll_interval_seconds=int(
            _require(sched_raw, "poll_interval_seconds", "scheduler")
        ),
        retry_attempts=int(_require(sched_raw, "retry_attempts", "scheduler")),
        state_db=str(sched_raw.get("state_db", "state/features.db")),
        log_dir=str(sched_raw.get("log_dir", "logs")),
    )
    if scheduler.max_running_tasks < 1:
        raise ValueError("scheduler.max_running_tasks must be >= 1")

    return AppConfig(
        project_id=str(project_id),
        aoi=aoi,
        temporal=temporal,
        grid=grid,
        sentinel2=sentinel2,
        export=export,
        scheduler=scheduler,
        config_path=config_path,
        project_root=project_root,
    )


def resolve_path(cfg: AppConfig, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return (cfg.project_root / path).resolve()
