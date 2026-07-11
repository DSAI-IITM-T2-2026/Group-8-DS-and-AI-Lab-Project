"""NetCDF validation for downloaded ERA5 files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import xarray as xr

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    file: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    variables_found: list[str] = field(default_factory=list)
    time_steps: int = 0
    missing_fraction: float = 0.0


def validate_netcdf(path: Path, expected_variables: list[str]) -> ValidationResult:
    result = ValidationResult(file=str(path), valid=True)

    if not path.exists():
        result.valid = False
        result.errors.append("File does not exist")
        return result

    if path.stat().st_size == 0:
        result.valid = False
        result.errors.append("File is empty")
        return result

    try:
        with xr.open_dataset(path) as ds:
            result.variables_found = sorted(ds.data_vars)
            result.time_steps = int(ds.sizes.get("valid_time", ds.sizes.get("time", 0)))

            if result.time_steps == 0:
                result.valid = False
                result.errors.append("No time dimension found")

            for var in expected_variables:
                if var not in ds.data_vars:
                    result.warnings.append(f"Expected variable missing: {var}")

            total = 0
            missing = 0
            for var in ds.data_vars:
                arr = ds[var].values
                total += arr.size
                missing += int((arr != arr).sum())  # NaN count

            result.missing_fraction = missing / total if total else 0.0
            if result.missing_fraction > 0.01:
                result.warnings.append(
                    f"High missing value fraction: {result.missing_fraction:.2%}"
                )

    except Exception as exc:
        result.valid = False
        result.errors.append(f"Failed to open NetCDF: {exc}")

    if result.errors:
        result.valid = False

    return result


def validate_all_raw(config: PipelineConfig) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    for year in config.years:
        for month in [f"{m:02d}" for m in range(1, 13)]:
            path = config.raw_file(year, month)
            if path.exists():
                results.append(validate_netcdf(path, config.variables))

    valid_count = sum(1 for r in results if r.valid)
    logger.info("Validated %d files: %d valid, %d invalid", len(results), valid_count, len(results) - valid_count)
    return results
