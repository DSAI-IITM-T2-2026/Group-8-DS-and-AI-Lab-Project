"""Pipeline configuration loading and directory setup."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StudyArea:
    north: float
    south: float
    west: float
    east: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north


@dataclass
class PipelineConfig:
    study_area: StudyArea
    base_dir: Path
    raw_dir: Path
    merged_dir: Path
    clipped_dir: Path
    terrain_dir: Path
    metadata_dir: Path
    logs_dir: Path
    workers: int
    download_retries: int
    retry_delay_seconds: float
    expected_crs: str
    nodata_value: float
    base_url: str
    arc_seconds: int
    hillshade_azimuth: float
    hillshade_altitude: float
    tpi_radius_pixels: int
    log_level: str
    dataset_name: str
    resolution_m: int

    @classmethod
    def from_yaml(cls, config_path: Path, project_root: Path | None = None) -> PipelineConfig:
        with config_path.open() as f:
            data = yaml.safe_load(f)

        root = project_root or config_path.parent
        base_dir = root / data["output"]["base_dir"]
        terrain = data.get("terrain", {})

        return cls(
            study_area=StudyArea(**data["study_area"]),
            base_dir=base_dir,
            raw_dir=base_dir / "raw",
            merged_dir=base_dir / "merged",
            clipped_dir=base_dir / "clipped",
            terrain_dir=base_dir / "terrain",
            metadata_dir=base_dir / "metadata",
            logs_dir=base_dir / "logs",
            workers=data["processing"]["workers"],
            download_retries=data["processing"]["download_retries"],
            retry_delay_seconds=data["processing"]["retry_delay_seconds"],
            expected_crs=data["processing"]["expected_crs"],
            nodata_value=data["processing"]["nodata_value"],
            base_url=data["dataset"]["base_url"],
            arc_seconds=data["dataset"]["arc_seconds"],
            hillshade_azimuth=terrain.get("hillshade", {}).get("azimuth", 315),
            hillshade_altitude=terrain.get("hillshade", {}).get("altitude", 45),
            tpi_radius_pixels=terrain.get("tpi", {}).get("radius_pixels", 5),
            log_level=data["logging"]["level"],
            dataset_name=data["dataset"]["name"],
            resolution_m=data["dataset"]["resolution_m"],
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_dir,
            self.merged_dir,
            self.clipped_dir,
            self.terrain_dir,
            self.metadata_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def setup_logging(config: PipelineConfig) -> logging.Logger:
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.logs_dir / "pipeline.log"

    logger = logging.getLogger("dem_pipeline")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
