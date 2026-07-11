"""Configuration loading and path management."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    dataset: str
    product_type: str
    years: list[str]
    area: list[float]
    variables: list[str]
    times: list[str]
    cds_accounts: list[dict]
    max_workers_per_account: int
    partition_strategy: str
    account_stagger_seconds: int
    max_retries: int
    queue_limit_max_retries: int
    retry_base_delay: int
    queue_limit_delay: int
    startup_delay_seconds: int
    request_delay_seconds: int
    data_format: str
    download_format: str
    paths: dict[str, Path]
    rainfall_aggregation_hours: int
    compression: str
    complevel: int

    @property
    def max_workers(self) -> int:
        account_count = max(len(self.cds_accounts), 1)
        return account_count * self.max_workers_per_account

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> PipelineConfig:
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        years = [str(y) for y in range(raw["years"]["start"], raw["years"]["end"] + 1)]
        if raw["download"].get("year_order", "ascending") == "descending":
            years.reverse()
        area_cfg = raw["area"]
        area = [area_cfg["north"], area_cfg["west"], area_cfg["south"], area_cfg["east"]]

        paths = {key: Path(value) for key, value in raw["paths"].items()}
        download = raw["download"]

        cds_accounts = download.get("cds_accounts", [])
        if not cds_accounts and download.get("max_workers", 1) == 1:
            cds_accounts = [{"name": "primary", "rc_file": "~/.cdsapirc"}]

        return cls(
            dataset=raw["dataset"],
            product_type=raw["product_type"],
            years=years,
            area=area,
            variables=raw["variables"],
            times=raw["times"],
            cds_accounts=cds_accounts,
            max_workers_per_account=download.get("max_workers_per_account", 1),
            partition_strategy=download.get("partition_strategy", "round_robin"),
            account_stagger_seconds=download.get("account_stagger_seconds", 30),
            max_retries=download["max_retries"],
            queue_limit_max_retries=download.get("queue_limit_max_retries", 30),
            retry_base_delay=download["retry_base_delay"],
            queue_limit_delay=download.get("queue_limit_delay", 600),
            startup_delay_seconds=download.get("startup_delay_seconds", 60),
            request_delay_seconds=download.get("request_delay_seconds", 15),
            data_format=download["data_format"],
            download_format=download["download_format"],
            paths=paths,
            rainfall_aggregation_hours=raw["processing"]["rainfall_aggregation_hours"],
            compression=raw["processing"]["compression"],
            complevel=raw["processing"]["complevel"],
        )

    def ensure_directories(self) -> None:
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def raw_file(self, year: str, month: str) -> Path:
        return self.paths["raw"] / f"era5_{year}_{month}.nc"

    def merged_file(self, year: str) -> Path:
        return self.paths["merged"] / f"era5_{year}.nc"

    def processed_file(self, year: str) -> Path:
        return self.paths["processed"] / f"era5_{year}_processed.nc"

    def checkpoint_file(self) -> Path:
        return self.paths["logs"] / "checkpoint.json"

    def download_log(self) -> Path:
        return self.paths["logs"] / "download.log"

    def metadata_file(self) -> Path:
        return self.paths["metadata"] / "dataset_metadata.json"

    def summary_csv(self) -> Path:
        return self.paths["metadata"] / "dataset_summary.csv"

    def quality_report(self) -> Path:
        return self.paths["reports"] / "quality_report.html"


def days_for_month(year: str, month: str) -> list[str]:
    count = calendar.monthrange(int(year), int(month))[1]
    return [f"{day:02d}" for day in range(1, count + 1)]
