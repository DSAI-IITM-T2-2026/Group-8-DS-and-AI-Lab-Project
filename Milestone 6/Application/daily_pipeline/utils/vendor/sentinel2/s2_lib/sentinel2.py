"""Sentinel-2 collection filtering, cloud masking, and compositing."""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass

import ee

from .config import AppConfig
from .grid import aoi_geometry
from .indices import INDEX_NAMES, add_indices


@dataclass(frozen=True)
class TimeWindow:
    window_id: str
    start_date: date
    end_date: date
    window_index: int

    @property
    def start_str(self) -> str:
        return self.start_date.isoformat()

    @property
    def end_str(self) -> str:
        return self.end_date.isoformat()

    @property
    def ee_end_exclusive(self) -> str:
        return (self.end_date + timedelta(days=1)).isoformat()


def generate_windows(start_year: int, end_year: int, window_days: int) -> list[TimeWindow]:
    windows: list[TimeWindow] = []
    cursor = date(start_year, 1, 1)
    final = date(end_year, 12, 31)
    year_counters: dict[int, int] = {}

    while cursor <= final:
        end = min(cursor + timedelta(days=window_days - 1), final)
        year = cursor.year
        year_counters[year] = year_counters.get(year, 0) + 1
        idx = year_counters[year]
        windows.append(
            TimeWindow(
                window_id=f"{cursor.isoformat()}_{end.isoformat()}",
                start_date=cursor,
                end_date=end,
                window_index=idx,
            )
        )
        cursor = end + timedelta(days=1)
    return windows


def find_window_for_date(
    target: date,
    start_year: int,
    end_year: int,
    window_days: int,
) -> TimeWindow | None:
    """Return the temporal window that contains ``target``, if any."""
    for window in generate_windows(start_year, end_year, window_days):
        if window.start_date <= target <= window.end_date:
            return window
    return None


def windows_for_year(
    year: int,
    start_year: int,
    end_year: int,
    window_days: int,
) -> list[TimeWindow]:
    """Windows whose start date falls in ``year`` (newest first)."""
    if year < start_year or year > end_year:
        return []
    windows = [
        w
        for w in generate_windows(start_year, end_year, window_days)
        if w.start_date.year == year
    ]
    windows.reverse()  # Dec → Jan so submit-all / queue prefer year-end first
    return windows


def filtered_collection(cfg: AppConfig, window: TimeWindow) -> ee.ImageCollection:
    """Join SR granules to cloud probability; retain all AOI observations."""
    region = aoi_geometry(cfg)
    sr = (
        ee.ImageCollection(cfg.sentinel2.collection)
        .filterBounds(region)
        .filterDate(window.start_str, window.ee_end_exclusive)
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE", cfg.sentinel2.cloud_filter_pct
            )
        )
    )
    cloud_probability = (
        ee.ImageCollection(cfg.sentinel2.cloud_probability_collection)
        .filterBounds(region)
        .filterDate(window.start_str, window.ee_end_exclusive)
    )
    joined = ee.Join.saveFirst("cloud_probability").apply(
        primary=sr,
        secondary=cloud_probability,
        condition=ee.Filter.equals(
            leftField="system:index", rightField="system:index"
        ),
    )
    return ee.ImageCollection(joined)


def _prepare_scene(cfg: AppConfig, image: ee.Image) -> ee.Image:
    """Scale, quality-mask, and calculate indices before compositing."""
    image = ee.Image(image)
    scl = image.select("SCL")
    probability = ee.Image(image.get("cloud_probability")).select("probability")

    # The official cloud-probability example recommends the B8A/B9 masks to
    # remove invalid granule edges that can survive the 10 m band masks.
    edge_valid = image.select("B8A").mask().And(image.select("B9").mask())
    observation_valid = scl.mask().And(edge_valid)

    surface = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    probability_clear = probability.lt(
        cfg.sentinel2.cloud_probability_threshold
    )
    # Preserve SCL snow/ice (class 11): cloud probability can classify bright
    # snow as cloud, while snow is a meaningful wildfire-model covariate.
    clear = (
        surface.And(probability_clear)
        .Or(scl.eq(11))
        .And(observation_valid)
    )

    reflectance = (
        image.select(cfg.sentinel2.bands)
        .multiply(0.0001)
        .updateMask(clear)
    )
    features = add_indices(
        reflectance,
        evi_min_abs_denominator=cfg.sentinel2.evi_min_abs_denominator,
        evi_max_abs=cfg.sentinel2.evi_max_abs,
    )

    observation = (
        scl.multiply(0)
        .add(1)
        .rename("observation")
        .updateMask(observation_valid)
    )
    clear_observation = (
        scl.multiply(0)
        .add(1)
        .rename("clear_observation")
        .updateMask(clear)
    )
    evi_observation = (
        features.select("EVI")
        .multiply(0)
        .add(1)
        .rename("evi_observation")
    )
    return (
        features.addBands([observation, clear_observation, evi_observation])
        .copyProperties(image, ["system:time_start", "system:index"])
    )


def _apply_composite(collection: ee.ImageCollection, method: str) -> ee.Image:
    method = method.lower()
    if method == "median":
        return collection.median()
    if method == "mean":
        return collection.mean()
    if method == "mosaic":
        return collection.mosaic()
    if method == "first":
        return ee.Image(collection.first())
    raise ValueError(f"Unsupported composite: {method}")


def build_feature_image(
    cfg: AppConfig,
    window: TimeWindow,
    *,
    collection: ee.ImageCollection | None = None,
) -> tuple[ee.Image, ee.Number]:
    """
    Build the analysis-ready composite with reflectance bands + indices.

    Returns (image, observation_count).
    """
    if collection is None:
        collection = filtered_collection(cfg, window)
    observation_count = collection.size()

    prepared = collection.map(lambda image: _prepare_scene(cfg, ee.Image(image)))
    feature_bands = cfg.sentinel2.bands + list(INDEX_NAMES)
    composite = _apply_composite(
        prepared.select(feature_bands), cfg.sentinel2.composite
    )

    observation_count_image = prepared.select("observation").sum().rename(
        "observation_count"
    )
    # ImageCollection.sum() leaves pixels masked when every scene contributes
    # a masked value.  For support counts, an observed-but-never-clear pixel
    # must be zero, not null; otherwise cloud percentage is missing precisely
    # where it should be 100% and cell means use inconsistent pixel support.
    observed_mask = observation_count_image.gt(0)
    clear_observation_count = (
        prepared.select("clear_observation")
        .sum()
        .unmask(0)
        .updateMask(observed_mask)
        .rename("clear_observation_count")
    )
    evi_observation_count = (
        prepared.select("evi_observation")
        .sum()
        .unmask(0)
        .updateMask(observed_mask)
    )

    cloud_percentage = (
        ee.Image.constant(1)
        .subtract(
            clear_observation_count.divide(observation_count_image.max(1))
        )
        .multiply(100)
        .updateMask(observed_mask)
        .rename("cloud_percentage")
    )
    evi_valid_fraction = (
        evi_observation_count.divide(clear_observation_count.max(1))
        .updateMask(clear_observation_count.gt(0))
        .rename("evi_valid_fraction")
    )

    valid = (
        composite.select(cfg.sentinel2.bands)
        .mask()
        .reduce(ee.Reducer.min())
        .rename("valid")
    )
    image = composite.addBands(
        [
            valid,
            cloud_percentage,
            observation_count_image,
            clear_observation_count,
            evi_valid_fraction,
        ]
    ).clip(aoi_geometry(cfg))
    return image, observation_count
