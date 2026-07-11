"""Generate exploratory plots for ERA5 variables."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

PLOT_SPECS = [
    ("temperature", ["t2m", "2m_temperature"], "2m Temperature (K)"),
    ("wind", ["wind_speed"], "Wind Speed (m/s)"),
    ("rainfall", ["tp", "total_precipitation", "rainfall_24h"], "Precipitation"),
    ("soil_moisture", ["soil_moisture_index", "swvl1", "volumetric_soil_water_layer_1"], "Soil Moisture"),
    ("humidity", ["relative_humidity"], "Relative Humidity (%)"),
]


def _pick_variable(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def _time_dim(ds: xr.Dataset) -> str:
    if "valid_time" in ds.dims:
        return "valid_time"
    return "time"


def generate_plots(config: PipelineConfig) -> list[Path]:
    plot_dir = config.paths["plots"]
    plot_dir.mkdir(parents=True, exist_ok=True)

    processed_files = sorted(config.paths["processed"].glob("*.nc"))
    if not processed_files:
        merged_files = sorted(config.paths["merged"].glob("*.nc"))
        source_files = merged_files
    else:
        source_files = processed_files

    if not source_files:
        logger.warning("No data files available for visualization")
        return []

    # Use most recent available year for sample plots
    sample_file = source_files[-1]
    created: list[Path] = []

    with xr.open_dataset(sample_file) as ds:
        tdim = _time_dim(ds)
        mid_time = ds[tdim].values[len(ds[tdim]) // 2]

        for plot_name, candidates, title in PLOT_SPECS:
            var = _pick_variable(ds, candidates)
            if var is None:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            data = ds[var].sel({tdim: mid_time}, method="nearest")

            if data.ndim == 2:
                im = data.plot(ax=ax, cmap="viridis", add_colorbar=True)
                im.set_label(var)
            else:
                data.plot(ax=ax)

            ax.set_title(f"{title} — {sample_file.stem}")
            fig.tight_layout()

            outpath = plot_dir / f"{plot_name}.png"
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            created.append(outpath)
            logger.info("Wrote plot: %s", outpath)

    return created
