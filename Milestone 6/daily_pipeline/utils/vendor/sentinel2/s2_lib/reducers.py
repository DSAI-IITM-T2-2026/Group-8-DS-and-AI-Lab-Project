"""Aggregate Sentinel-2 features onto the 1 km grid (vectorized EE ops)."""

from __future__ import annotations

import ee

from .config import AppConfig
from .grid import aoi_geometry
from .indices import INDEX_FULL_STATS, INDEX_MEAN_STD
from .sentinel2 import TimeWindow, build_feature_image, filtered_collection

SCHEMA_VERSION = "3.0"

OUTPUT_PROPERTIES = [
    "schema_version",
    "grid_id",
    "latitude",
    "longitude",
    "window_start",
    "window_end",
    "B2_mean",
    "B2_std",
    "B3_mean",
    "B3_std",
    "B4_mean",
    "B4_std",
    "B8_mean",
    "B8_std",
    "B11_mean",
    "B11_std",
    "B12_mean",
    "B12_std",
    "NDVI_mean",
    "NDVI_std",
    "NDVI_min",
    "NDVI_max",
    "NDMI_mean",
    "NDMI_std",
    "NDMI_min",
    "NDMI_max",
    "NBR_mean",
    "NBR_std",
    "NBR_min",
    "NBR_max",
    "NDWI_mean",
    "NDWI_std",
    "NDWI_min",
    "NDWI_max",
    "EVI_mean",
    "EVI_std",
    "EVI_min",
    "EVI_max",
    "SAVI_mean",
    "SAVI_std",
    "MSAVI_mean",
    "MSAVI_std",
    "valid_sample_count",
    "valid_fraction",
    "cloud_percentage",
    "observation_count_mean",
    "clear_observation_count_mean",
    "evi_valid_fraction",
    "s2_data_available",
]


def _mean_std_reducer(prefix: str) -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .setOutputs([f"{prefix}_mean", f"{prefix}_std"])
    )


def _full_stats_reducer(prefix: str) -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .setOutputs(
            [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_min",
                f"{prefix}_max",
            ]
        )
    )


def _selective_reduction(cfg: AppConfig) -> tuple[list[str], ee.Reducer]:
    """One raster pass, with only the statistics used by the output schema."""
    specs: list[tuple[str, ee.Reducer]] = []
    specs.extend((band, _mean_std_reducer(band)) for band in cfg.sentinel2.bands)
    specs.extend((name, _full_stats_reducer(name)) for name in INDEX_FULL_STATS)
    specs.extend((name, _mean_std_reducer(name)) for name in INDEX_MEAN_STD)

    valid_reducer = (
        # The validity image is an unmasked 0/1 support band. Summing it counts
        # only valid samples; Reducer.count() would incorrectly count zeros.
        # Unweighted reduction uses pixel centres and keeps the count integral.
        ee.Reducer.sum()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .setOutputs(["valid_sample_count", "valid_fraction"])
    )
    specs.extend(
        [
            ("valid", valid_reducer),
            (
                "cloud_percentage",
                ee.Reducer.mean().setOutputs(["cloud_percentage"]),
            ),
            (
                "observation_count",
                ee.Reducer.mean().setOutputs(["observation_count_mean"]),
            ),
            (
                "clear_observation_count",
                ee.Reducer.mean().setOutputs(
                    ["clear_observation_count_mean"]
                ),
            ),
            (
                "evi_valid_fraction",
                ee.Reducer.mean().setOutputs(["evi_valid_fraction"]),
            ),
        ]
    )

    reducer = specs[0][1]
    for _, next_reducer in specs[1:]:
        reducer = reducer.combine(next_reducer, sharedInputs=False)
    # Apply one consistent pixel-centre weighting mode to the complete
    # heterogeneous reducer. Earth Engine does not permit weighted and
    # unweighted child reducers to be combined in the same reduceRegions call.
    return [name for name, _ in specs], reducer.unweighted()


def _reduce_tiles(cfg: AppConfig) -> ee.FeatureCollection:
    """Internal configured tiles used only to chunk reduceRegions."""
    region = aoi_geometry(cfg)
    tile_m = cfg.grid.reduce_tile_m
    proj = ee.Projection(cfg.grid.crs).atScale(tile_m)
    try:
        return region.coveringGrid(proj, tile_m)
    except Exception:
        return ee.FeatureCollection([ee.Feature(region)])


def _cells_for_tile(
    cfg: AppConfig,
    grid: ee.FeatureCollection,
    tile_geom: ee.Geometry,
) -> ee.FeatureCollection:
    """
    Assign each 1 km cell to exactly one reduce tile via integer ix/iy ranges.

    Faster than per-cell centroid transforms: uses properties set in build_grid
    and half-open index intervals [ix0, ix1) × [iy0, iy1).
    """
    res = cfg.grid.resolution_m
    proj = ee.Projection(cfg.grid.crs)
    candidates = grid.filterBounds(tile_geom)
    bounds = tile_geom.bounds(1, proj)
    ring = ee.List(ee.List(bounds.coordinates()).get(0))
    xs = ring.map(lambda p: ee.List(p).get(0))
    ys = ring.map(lambda p: ee.List(p).get(1))
    xmin = ee.Number(xs.reduce(ee.Reducer.min()))
    ymin = ee.Number(ys.reduce(ee.Reducer.min()))
    xmax = ee.Number(xs.reduce(ee.Reducer.max()))
    ymax = ee.Number(ys.reduce(ee.Reducer.max()))
    ix0 = xmin.divide(res).round().int()
    iy0 = ymin.divide(res).round().int()
    ix1 = xmax.divide(res).round().int()
    iy1 = ymax.divide(res).round().int()
    return candidates.filter(
        ee.Filter.And(
            ee.Filter.gte("ix", ix0),
            ee.Filter.lt("ix", ix1),
            ee.Filter.gte("iy", iy0),
            ee.Filter.lt("iy", iy1),
        )
    )


def aggregate_to_grid(
    cfg: AppConfig,
    window: TimeWindow,
    grid: ee.FeatureCollection,
) -> ee.FeatureCollection:
    """
    Aggregate composite + indices onto the 1 km grid.

    California-scale AOIs exceed EE's single ``reduceRegions`` limit, so
    reductions run per ``grid.reduce_tile_m`` tile and are flattened.
    Each grid cell is assigned to exactly one tile (ix/iy half-open).
    """
    collection = filtered_collection(cfg, window)
    observation_count = collection.size()

    def _reduce() -> ee.FeatureCollection:
        image, _ = build_feature_image(cfg, window, collection=collection)
        bands = cfg.sentinel2.bands
        select_bands, reducer = _selective_reduction(cfg)
        stats_img = image.select(select_bands)
        tiles = _reduce_tiles(cfg)

        # Map tiles → List of feature-lists, then flatten (valid EE pattern).
        def _tile_to_list(tile: ee.Feature) -> ee.List:
            geom = ee.Feature(tile).geometry()
            cells = _cells_for_tile(cfg, grid, geom)
            return (
                stats_img.reduceRegions(
                    collection=cells,
                    reducer=reducer,
                    scale=cfg.sentinel2.reduce_scale_m,
                    crs=cfg.grid.crs,
                    tileScale=cfg.sentinel2.tile_scale,
                )
                .toList(100_000)
            )

        reduced = ee.FeatureCollection(
            tiles.toList(2_000).map(_tile_to_list).flatten()
        )

        def _format(feature: ee.Feature) -> ee.Feature:
            props = {
                "schema_version": SCHEMA_VERSION,
                "grid_id": ee.String(feature.get("grid_id")),
                "latitude": feature.get("latitude"),
                "longitude": feature.get("longitude"),
                "window_start": window.start_str,
                "window_end": window.end_str,
            }

            for band in bands:
                props[f"{band}_mean"] = feature.get(f"{band}_mean")
                props[f"{band}_std"] = feature.get(f"{band}_std")

            for name in INDEX_FULL_STATS:
                props[f"{name}_mean"] = feature.get(f"{name}_mean")
                props[f"{name}_std"] = feature.get(f"{name}_std")
                props[f"{name}_min"] = feature.get(f"{name}_min")
                props[f"{name}_max"] = feature.get(f"{name}_max")

            for name in INDEX_MEAN_STD:
                props[f"{name}_mean"] = feature.get(f"{name}_mean")
                props[f"{name}_std"] = feature.get(f"{name}_std")

            valid_sample_count = ee.Number(
                ee.List([feature.get("valid_sample_count"), 0]).reduce(
                    ee.Reducer.firstNonNull()
                )
            )
            valid_fraction = ee.Number(
                ee.List([feature.get("valid_fraction"), 0]).reduce(
                    ee.Reducer.firstNonNull()
                )
            )
            props.update(
                {
                    "valid_sample_count": valid_sample_count,
                    "valid_fraction": valid_fraction,
                    "cloud_percentage": feature.get("cloud_percentage"),
                    "observation_count_mean": feature.get(
                        "observation_count_mean"
                    ),
                    "clear_observation_count_mean": feature.get(
                        "clear_observation_count_mean"
                    ),
                    "evi_valid_fraction": feature.get("evi_valid_fraction"),
                    "s2_data_available": valid_sample_count.gt(0),
                }
            )
            return ee.Feature(None, props)

        formatted = reduced.map(_format)
        return formatted.select(OUTPUT_PROPERTIES)

    def _empty_window() -> ee.FeatureCollection:
        numeric_features = OUTPUT_PROPERTIES[6:-1]

        def _format_empty(feature: ee.Feature) -> ee.Feature:
            props = {name: None for name in numeric_features}
            props.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "grid_id": ee.String(feature.get("grid_id")),
                    "latitude": feature.get("latitude"),
                    "longitude": feature.get("longitude"),
                    "window_start": window.start_str,
                    "window_end": window.end_str,
                    "valid_sample_count": 0,
                    "valid_fraction": 0,
                    "observation_count_mean": 0,
                    "clear_observation_count_mean": 0,
                    "s2_data_available": False,
                }
            )
            return ee.Feature(None, props)

        return grid.map(_format_empty).select(OUTPUT_PROPERTIES)

    return ee.FeatureCollection(
        ee.Algorithms.If(observation_count.gt(0), _reduce(), _empty_window())
    )
