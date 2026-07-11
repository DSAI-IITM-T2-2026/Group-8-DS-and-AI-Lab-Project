"""GCS upload configuration and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dem_pipeline.config import PipelineConfig


@dataclass
class GcsUploadConfig:
    destination: str
    gcs_project_id: str
    gcs_bucket: str
    gcs_prefix: str
    year_range: str
    region: str
    include_raw: bool
    include_merged: bool
    include_clipped: bool
    include_terrain: bool
    include_metadata: bool
    include_logs: bool = False
    merged_final_only: bool = True

    @property
    def gcs_base_uri(self) -> str:
        return (
            f"gs://{self.gcs_bucket}/{self.gcs_prefix}/dem/"
            f"{self.year_range}/{self.region}"
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> GcsUploadConfig:
        with config_path.open() as f:
            data = yaml.safe_load(f)

        upload = data.get("upload", {})
        return cls(
            destination=upload.get("destination", "gcs"),
            gcs_project_id=upload["gcs_project_id"],
            gcs_bucket=upload["gcs_bucket"],
            gcs_prefix=upload["gcs_prefix"],
            year_range=upload.get("year_range", "2021-2025"),
            region=upload.get("region", "california"),
            include_raw=upload.get("include_raw", False),
            include_merged=upload.get("include_merged", False),
            include_clipped=upload.get("include_clipped", True),
            include_terrain=upload.get("include_terrain", True),
            include_metadata=upload.get("include_metadata", True),
            include_logs=upload.get("include_logs", False),
            merged_final_only=upload.get("merged_final_only", True),
        )


def collect_upload_files(
    pipeline_config: PipelineConfig,
    upload_config: GcsUploadConfig,
) -> list[tuple[Path, str]]:
    """Return local paths and their GCS object keys relative to the base prefix."""
    base_key = f"{upload_config.gcs_prefix}/dem/{upload_config.year_range}/{upload_config.region}"
    files: list[tuple[Path, str]] = []

    def add_dir(local_dir: Path, subdir: str, *, name_filter: str | None = None) -> None:
        if not local_dir.exists():
            return
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            if name_filter and path.name != name_filter:
                continue
            rel = path.relative_to(local_dir)
            files.append((path, f"{base_key}/{subdir}/{rel.as_posix()}"))

    if upload_config.include_raw:
        add_dir(pipeline_config.raw_dir, "raw")
    if upload_config.include_merged:
        merged_filter = "dem_merged.tif" if upload_config.merged_final_only else None
        add_dir(pipeline_config.merged_dir, "merged", name_filter=merged_filter)
    if upload_config.include_clipped:
        add_dir(pipeline_config.clipped_dir, "clipped")
    if upload_config.include_terrain:
        add_dir(pipeline_config.terrain_dir, "terrain")
    if upload_config.include_metadata:
        add_dir(pipeline_config.metadata_dir, "metadata")
    if upload_config.include_logs:
        add_dir(pipeline_config.logs_dir, "logs")

    return files
