from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..grid import coordinates_to_cell_ids
from ..io import atomic_parquet

logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "date",
    "cell_id",
    "firms_n_pixels",
    "firms_max_confidence",
    "y_fire",
]


def firms_cache_path(cfg: dict, year: int, month: int) -> Path:
    return cfg["paths"]["cache_dir"] / "firms_cells" / f"year={year}" / f"month={month:02d}.parquet"


def _seed_file(cfg: dict, year: int, month: int) -> Path | None:
    names = [
        f"firms_cells_{year}_{month:02d}.parquet",
        f"year={year}/month={month:02d}.parquet",
    ]
    for directory in cfg["paths"].get("firms_seed_dirs", []):
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def label_day(
    date: pd.Timestamp,
    vsigs_prefix: str,
    confidence_min: float,
    resolution: float,
) -> pd.DataFrame:
    """Map qualifying FIRMS raster pixels to ERA5 cells for one UTC-named day."""
    try:
        import rasterio
        from rasterio.transform import xy
    except ImportError as exc:
        raise ImportError(
            "Building FIRMS labels requires rasterio. Install the project requirements "
            "before running the era5_firms stage."
        ) from exc
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    path = f"{vsigs_prefix.rstrip('/')}/{date:%Y-%m-%d}.tif"
    with rasterio.open(path) as source:
        descriptions = list(source.descriptions or [])
        band_map = (
            {name: index + 1 for index, name in enumerate(descriptions)}
            if descriptions and all(descriptions)
            else {"firms_confidence": 1}
        )
        if "firms_confidence" not in band_map:
            raise ValueError(f"{path} lacks a firms_confidence band")
        confidence = source.read(band_map["firms_confidence"]).astype("float32")
        fire = np.isfinite(confidence) & (confidence >= confidence_min)
        if not fire.any():
            return pd.DataFrame(columns=LABEL_COLUMNS)
        rows, columns = np.where(fire)
        longitudes, latitudes = xy(source.transform, rows, columns, offset="center")
        cell_ids = coordinates_to_cell_ids(
            np.asarray(longitudes), np.asarray(latitudes), resolution
        )
        values = confidence[rows, columns]

    pixels = pd.DataFrame({"cell_id": cell_ids, "confidence": values})
    result = pixels.groupby("cell_id", as_index=False).agg(
        firms_n_pixels=("confidence", "size"),
        firms_max_confidence=("confidence", "max"),
    )
    result["date"] = date.normalize()
    result["y_fire"] = 1
    return result[LABEL_COLUMNS]


def build_firms_month(cfg: dict, year: int, month: int, force: bool = False) -> Path:
    destination = firms_cache_path(cfg, year, month)
    if destination.exists() and not force:
        return destination
    seed = _seed_file(cfg, year, month)
    if seed is not None:
        atomic_parquet(pd.read_parquet(seed), destination)
        logger.info("Seeded FIRMS %04d-%02d from %s", year, month, seed)
        return destination

    period = pd.Period(f"{year}-{month:02d}", freq="M")
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    results: list[pd.DataFrame] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=int(cfg["execution"]["max_workers"])) as pool:
        futures = {
            pool.submit(
                label_day,
                day,
                cfg["gcs"]["firms_vsigs_prefix"],
                float(cfg["task"]["firms_confidence_min"]),
                float(cfg["aoi"]["resolution_deg"]),
            ): day
            for day in days
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            # Source libraries raise several exception families. Collect all day failures,
            # then fail the month so none are silently converted into negative labels.
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{futures[future].date()}: {exc}")
    if errors:
        examples = "\n".join(errors[:5])
        raise RuntimeError(
            f"FIRMS failed on {len(errors)} days in {year}-{month:02d}; "
            f"refusing to create false negatives.\n{examples}"
        )
    frame = (
        pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=LABEL_COLUMNS)
    )
    atomic_parquet(frame, destination)
    logger.info("Built FIRMS shard %s (%d positive cell-days)", destination, len(frame))
    return destination


def read_firms_year(cfg: dict, year: int) -> pd.DataFrame:
    frames = []
    for month in range(1, 13):
        path = firms_cache_path(cfg, year, month)
        if not path.exists():
            raise FileNotFoundError(f"Missing FIRMS shard: {path}")
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    return combined
