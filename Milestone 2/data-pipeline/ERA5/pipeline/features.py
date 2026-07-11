"""Feature engineering for wildfire ML datasets."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from tqdm import tqdm

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def _relative_humidity(t2m_k: xr.DataArray, d2m_k: xr.DataArray) -> xr.DataArray:
    """Magnus formula for relative humidity (%)."""
    t_c = t2m_k - 273.15
    d_c = d2m_k - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * d_c) / (d_c + 243.5))
    rh = (e / es) * 100.0
    return rh.clip(0, 100).rename("relative_humidity")


def _wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return np.sqrt(u**2 + v**2).rename("wind_speed")


def _wind_direction(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    direction = (180.0 / np.pi) * np.arctan2(u, v) + 180.0
    return direction.rename("wind_direction")


def _soil_moisture_index(swvl1: xr.DataArray, swvl2: xr.DataArray) -> xr.DataArray:
    return ((swvl1 + swvl2) / 2.0).rename("soil_moisture_index")


def process_year(config: PipelineConfig, year: str) -> bool:
    infile = config.merged_file(year)
    outfile = config.processed_file(year)

    if not infile.exists():
        logger.warning("No merged file for %s, skipping feature engineering", year)
        return False

    if outfile.exists() and outfile.stat().st_size > 0:
        logger.info("Processed file already exists for %s, skipping", year)
        return True

    logger.info("Feature engineering for %s", year)

    with xr.open_dataset(infile) as ds:
        features = ds.copy()

        if "t2m" in ds and "d2m" in ds:
            features["relative_humidity"] = _relative_humidity(ds["t2m"], ds["d2m"])
        elif "2m_temperature" in ds and "2m_dewpoint_temperature" in ds:
            features["relative_humidity"] = _relative_humidity(
                ds["2m_temperature"], ds["2m_dewpoint_temperature"]
            )

        u_var = "u10" if "u10" in ds else "10m_u_component_of_wind"
        v_var = "v10" if "v10" in ds else "10m_v_component_of_wind"

        if u_var in ds and v_var in ds:
            features["wind_speed"] = _wind_speed(ds[u_var], ds[v_var])
            features["wind_direction"] = _wind_direction(ds[u_var], ds[v_var])

        tp_var = "tp" if "tp" in ds else "total_precipitation"
        if tp_var in ds:
            hours = config.rainfall_aggregation_hours
            features[f"rainfall_{hours}h"] = (
                ds[tp_var].rolling(valid_time=hours, min_periods=1).sum()
            )

        sw1 = "swvl1" if "swvl1" in ds else "volumetric_soil_water_layer_1"
        sw2 = "swvl2" if "swvl2" in ds else "volumetric_soil_water_layer_2"
        if sw1 in ds and sw2 in ds:
            features["soil_moisture_index"] = _soil_moisture_index(ds[sw1], ds[sw2])

        encoding = {
            var: {"zlib": True, "complevel": config.complevel}
            for var in features.data_vars
        }
        features.to_netcdf(outfile, encoding=encoding)

    logger.info("Wrote processed file: %s", outfile)
    return True


def process_all_years(config: PipelineConfig) -> dict[str, bool]:
    results = {}
    for year in tqdm(config.years, desc="Feature engineering"):
        results[year] = process_year(config, year)
    return results
