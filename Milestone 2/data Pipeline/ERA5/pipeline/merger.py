"""Merge monthly NetCDF files into yearly datasets."""

from __future__ import annotations

import logging

import xarray as xr
from tqdm import tqdm

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def merge_year(config: PipelineConfig, year: str) -> bool:
    monthly_files = [config.raw_file(year, f"{m:02d}") for m in range(1, 13)]
    existing = [f for f in monthly_files if f.exists()]

    if len(existing) < 12:
        logger.warning("Year %s: only %d/12 monthly files available, skipping merge", year, len(existing))
        return False

    outfile = config.merged_file(year)
    if outfile.exists() and outfile.stat().st_size > 0:
        logger.info("Merged file already exists for %s, skipping", year)
        return True

    logger.info("Merging %s (%d monthly files)", year, len(existing))

    datasets = []
    for path in existing:
        ds = xr.open_dataset(path)
        datasets.append(ds)

    merged = xr.concat(datasets, dim="valid_time")
    if "valid_time" in merged.dims:
        merged = merged.sortby("valid_time")

    encoding = {
        var: {"zlib": True, "complevel": config.complevel}
        for var in merged.data_vars
    }

    merged.to_netcdf(outfile, encoding=encoding)
    merged.close()
    for ds in datasets:
        ds.close()

    logger.info("Wrote merged file: %s", outfile)
    return True


def merge_all_years(config: PipelineConfig) -> dict[str, bool]:
    results = {}
    for year in tqdm(config.years, desc="Merging yearly files"):
        results[year] = merge_year(config, year)
    return results
