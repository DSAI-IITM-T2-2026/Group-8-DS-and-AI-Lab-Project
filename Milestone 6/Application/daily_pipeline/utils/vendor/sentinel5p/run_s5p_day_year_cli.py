#!/usr/bin/env python3
"""Local runner for Sentinel5P 2019 year submit (non-Colab)."""
from __future__ import annotations
import ee

print("=== Initialize EE ===")
ee.Initialize(project="plated-mechanic-418917")
print("EE ready")

print("=== Config ===")
GEE_PROJECT_ID = "plated-mechanic-418917"
YEAR = 2019   # already correct in the notebook

GCS_BUCKET = "plated-mechanic-s5p-2016-2025"   # your new bucket
GCS_PREFIX = "sentinel5p_features_daily"       # keep as-is

# Safe defaults.
RESUBMIT_EXISTING_TASKS = False
TASK_START_RETRIES = 3

if not GEE_PROJECT_ID.strip():
    raise ValueError("Enter an Earth Engine-enabled GEE_PROJECT_ID above.")
if not 2018 <= int(YEAR) <= 2025:
    raise ValueError("YEAR must be a complete Sentinel-5P year from 2018 through 2025.")

print(f"Earth Engine project: {GEE_PROJECT_ID}")
print(f"Assigned year:        {YEAR}")
print(f"Output root:          gs://{GCS_BUCKET}/{GCS_PREFIX}/year={YEAR}/")

print("=== Load workflow ===")
import base64
import gzip
import json
import time
from collections import Counter
from datetime import date, timedelta

import pandas as pd

CALIFORNIA_GEOJSON_GZIP_B64 = "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"
CALIFORNIA_GEOJSON = json.loads(
    gzip.decompress(base64.b64decode(CALIFORNIA_GEOJSON_GZIP_B64))
)
AOI = ee.FeatureCollection(CALIFORNIA_GEOJSON["features"]).geometry(100)

GRID_CRS = "EPSG:3310"
GRID_RESOLUTION_M = 1000
REDUCE_TILE_M = 100000
REDUCE_SCALE_M = 1113.2
TILE_SCALE = 4
AAI_COLLECTION = "COPERNICUS/S5P/OFFL/L3_AER_AI"
AAI_BAND = "absorbing_aerosol_index"
CO_COLLECTION = "COPERNICUS/S5P/OFFL/L3_CO"
CO_BAND = "CO_column_number_density"
SCHEMA_VERSION = "1.0"

OUTPUT_PROPERTIES = [
    "schema_version", "grid_id", "latitude", "longitude",
    "window_start", "window_end", "window_days",
    "s5p_aai_mean", "s5p_aai_max", "s5p_aai_std",
    "s5p_aai_valid_days_mean", "s5p_aai_valid_fraction",
    "s5p_co_mean", "s5p_co_max", "s5p_co_std",
    "s5p_co_valid_days_mean", "s5p_co_valid_fraction",
    "s5p_aai_observation_count_mean", "s5p_co_observation_count_mean",
    "s5p_aai_data_available", "s5p_co_data_available",
    "s5p_data_available",
]

def build_grid():
    projection = ee.Projection(GRID_CRS).atScale(GRID_RESOLUTION_M)
    cells = AOI.coveringGrid(projection, GRID_RESOLUTION_M)
    crs_m = ee.Projection(GRID_CRS)

    def annotate(feature):
        feature = ee.Feature(feature)
        geometry = feature.geometry()
        ring = ee.List(geometry.bounds(1, crs_m).coordinates().get(0))
        xs = ring.map(lambda point: ee.List(point).get(0))
        ys = ring.map(lambda point: ee.List(point).get(1))
        ix = ee.Number(xs.reduce(ee.Reducer.min())).divide(GRID_RESOLUTION_M).round().int()
        iy = ee.Number(ys.reduce(ee.Reducer.min())).divide(GRID_RESOLUTION_M).round().int()
        lonlat = geometry.centroid(1).transform("EPSG:4326", 1).coordinates()
        return feature.set({
            "grid_id": ix.format("%d").cat("_").cat(iy.format("%d")),
            "ix": ix, "iy": iy,
            "longitude": lonlat.get(0), "latitude": lonlat.get(1),
        })

    return ee.FeatureCollection(cells.map(annotate))

def reduce_tiles():
    projection = ee.Projection(GRID_CRS).atScale(REDUCE_TILE_M)
    return AOI.coveringGrid(projection, REDUCE_TILE_M)

def cells_for_tile(grid, tile_geometry):
    projection = ee.Projection(GRID_CRS)
    ring = ee.List(tile_geometry.bounds(1, projection).coordinates().get(0))
    xs = ring.map(lambda point: ee.List(point).get(0))
    ys = ring.map(lambda point: ee.List(point).get(1))
    ix0 = ee.Number(xs.reduce(ee.Reducer.min())).divide(GRID_RESOLUTION_M).round().int()
    iy0 = ee.Number(ys.reduce(ee.Reducer.min())).divide(GRID_RESOLUTION_M).round().int()
    ix1 = ee.Number(xs.reduce(ee.Reducer.max())).divide(GRID_RESOLUTION_M).round().int()
    iy1 = ee.Number(ys.reduce(ee.Reducer.max())).divide(GRID_RESOLUTION_M).round().int()
    return grid.filterBounds(tile_geometry).filter(ee.Filter.And(
        ee.Filter.gte("ix", ix0), ee.Filter.lt("ix", ix1),
        ee.Filter.gte("iy", iy0), ee.Filter.lt("iy", iy1),
    ))

def masked_image(band):
    return ee.Image.constant(0).rename(band).updateMask(ee.Image.constant(0))

def source_collection(collection_id, band, day):
    start = ee.Date(day.isoformat())
    return (ee.ImageCollection(collection_id)
            .filterBounds(AOI)
            .filterDate(start, start.advance(1, "day"))
            .select(band))

def one_day_collection(source, band, day):
    start = ee.Date(day.isoformat())
    image = ee.Image(ee.Algorithms.If(source.size().gt(0), source.mean(), masked_image(band)))
    return ee.ImageCollection.fromImages([
        image.set({"system:time_start": start.millis(), "source_orbit_count": source.size()})
    ])

def product_stats(daily, prefix):
    valid_days = daily.count().rename(f"{prefix}_valid_days_mean")
    return (daily.mean().rename(f"{prefix}_mean")
            .addBands(daily.max().rename(f"{prefix}_max"))
            .addBands(daily.reduce(ee.Reducer.stdDev()).rename(f"{prefix}_std"))
            .addBands(valid_days)
            .addBands(valid_days.rename(f"{prefix}_valid_fraction")))

def build_feature_image(day):
    aai_source = source_collection(AAI_COLLECTION, AAI_BAND, day)
    co_source = source_collection(CO_COLLECTION, CO_BAND, day)
    aai_daily = one_day_collection(aai_source, AAI_BAND, day)
    co_daily = one_day_collection(co_source, CO_BAND, day)
    return (product_stats(aai_daily, "s5p_aai")
            .addBands(product_stats(co_daily, "s5p_co"))
            .addBands(aai_source.count().rename("s5p_aai_observation_count_mean"))
            .addBands(co_source.count().rename("s5p_co_observation_count_mean"))
            .clip(AOI))

def aggregate_day(day, grid):
    image = build_feature_image(day)
    tiles = reduce_tiles()

    def tile_to_list(tile):
        geometry = ee.Feature(tile).geometry()
        cells = cells_for_tile(grid, geometry)
        return image.reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            scale=REDUCE_SCALE_M,
            crs=GRID_CRS,
            tileScale=TILE_SCALE,
        ).toList(100000)

    reduced = ee.FeatureCollection(tiles.toList(2000).map(tile_to_list).flatten())
    day_text = day.isoformat()

    def format_feature(feature):
        feature = ee.Feature(feature)
        aai_days = ee.Number(ee.List([
            feature.get("s5p_aai_valid_days_mean"), 0
        ]).reduce(ee.Reducer.firstNonNull()))
        co_days = ee.Number(ee.List([
            feature.get("s5p_co_valid_days_mean"), 0
        ]).reduce(ee.Reducer.firstNonNull()))
        aai_available = aai_days.gt(0)
        co_available = co_days.gt(0)
        return ee.Feature(None, {
            "schema_version": SCHEMA_VERSION,
            "grid_id": ee.String(feature.get("grid_id")),
            "latitude": feature.get("latitude"),
            "longitude": feature.get("longitude"),
            "window_start": day_text,
            "window_end": day_text,
            "window_days": 1,
            "s5p_aai_mean": feature.get("s5p_aai_mean"),
            "s5p_aai_max": feature.get("s5p_aai_max"),
            "s5p_aai_std": feature.get("s5p_aai_std"),
            "s5p_aai_valid_days_mean": aai_days,
            "s5p_aai_valid_fraction": aai_days,
            "s5p_co_mean": feature.get("s5p_co_mean"),
            "s5p_co_max": feature.get("s5p_co_max"),
            "s5p_co_std": feature.get("s5p_co_std"),
            "s5p_co_valid_days_mean": co_days,
            "s5p_co_valid_fraction": co_days,
            "s5p_aai_observation_count_mean": feature.get("s5p_aai_observation_count_mean"),
            "s5p_co_observation_count_mean": feature.get("s5p_co_observation_count_mean"),
            "s5p_aai_data_available": aai_available,
            "s5p_co_data_available": co_available,
            "s5p_data_available": aai_available.Or(co_available),
        })

    return reduced.map(format_feature).select(OUTPUT_PROPERTIES)

def task_description(day):
    stamp = day.strftime("%Y%m%d")
    return f"s5pfeat_{stamp}_{stamp}"

def export_prefix(day):
    window_number = day.timetuple().tm_yday
    return (
        f"{GCS_PREFIX}/year={day.year:04d}/month={day.month:02d}/"
        f"window={window_number:03d}/features"
    )

def descending_days(year):
    current = date(year, 12, 31)
    first = date(year, 1, 1)
    while current >= first:
        yield current
        current -= timedelta(days=1)

print("Workflow loaded.")

print("=== Preflight ===")
GRID = build_grid()
grid_count = GRID.size().getInfo()
aoi_area_km2 = AOI.area(100).divide(1_000_000).getInfo()
days = list(descending_days(int(YEAR)))

assert days[0] == date(int(YEAR), 12, 31)
assert days[-1] == date(int(YEAR), 1, 1)
assert len(days) in (365, 366)

print(f"AOI area:       {aoi_area_km2:,.0f} km²")
print(f"Grid cells:     {grid_count:,}")
print(f"Daily tasks:    {len(days)}")
print(f"Submission:     {days[0]} → {days[-1]} (descending)")
print(f"Destination:    gs://{GCS_BUCKET}/{GCS_PREFIX}/year={YEAR}/")
print("\nPreflight passed. Run the next cell once to submit the year.")

print("=== Submit year ===")
existing_rows = ee.data.getTaskList()
existing_by_description = {
    row.get("description"): row.get("state", "UNKNOWN")
    for row in existing_rows
    if row.get("description")
}

manifest = []
for position, day in enumerate(days, start=1):
    description = task_description(day)
    prefix = export_prefix(day)
    existing_state = existing_by_description.get(description)

    if existing_state and not RESUBMIT_EXISTING_TASKS:
        manifest.append({
            "date": day.isoformat(), "action": "SKIPPED_EXISTING",
            "task_id": None, "state": existing_state,
            "gcs_csv": f"gs://{GCS_BUCKET}/{prefix}.csv",
        })
        print(f"[{position:03d}/{len(days)}] skip  {day} ({existing_state})")
        continue

    collection = aggregate_day(day, GRID)
    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        description=description,
        bucket=GCS_BUCKET,
        fileNamePrefix=prefix,
        fileFormat="CSV",
        selectors=OUTPUT_PROPERTIES,
    )

    error = None
    for attempt in range(1, TASK_START_RETRIES + 1):
        try:
            task.start()
            error = None
            break
        except Exception as exc:
            error = str(exc)
            if attempt < TASK_START_RETRIES:
                time.sleep(2 ** attempt)

    if error is not None:
        manifest.append({
            "date": day.isoformat(), "action": "START_FAILED",
            "task_id": None, "state": "FAILED_TO_START", "error": error,
            "gcs_csv": f"gs://{GCS_BUCKET}/{prefix}.csv",
        })
        print(f"[{position:03d}/{len(days)}] ERROR {day}: {error}")
        continue

    manifest.append({
        "date": day.isoformat(), "action": "SUBMITTED",
        "task_id": task.id, "state": "SUBMITTED",
        "gcs_csv": f"gs://{GCS_BUCKET}/{prefix}.csv",
    })
    print(f"[{position:03d}/{len(days)}] start {day} → {task.id}")

manifest_df = pd.DataFrame(manifest)
manifest_path = f"sentinel5p_submission_manifest_{YEAR}.csv"
manifest_df.to_csv(manifest_path, index=False)

print("\nSubmission summary")
print(manifest_df["action"].value_counts().to_string())
print(f"Manifest saved in the notebook runtime: {manifest_path}")
display(manifest_df.head())
display(manifest_df.tail())

print("=== Status snapshot ===")
year_prefix = f"s5pfeat_{int(YEAR):04d}"
year_tasks = [
    row for row in ee.data.getTaskList()
    if str(row.get("description", "")).startswith(year_prefix)
]
counts = Counter(row.get("state", "UNKNOWN") for row in year_tasks)

print(f"Visible Sentinel-5P tasks for {YEAR}: {len(year_tasks)}")
for state, count in sorted(counts.items()):
    print(f"  {state:12s} {count}")

failed = [row for row in year_tasks if row.get("state") in {"FAILED", "CANCELLED"}]
if failed:
    print("\nFailed/cancelled tasks (first 20):")
    for row in failed[:20]:
        print(row.get("description"), row.get("error_message", row.get("error", "")))

print("DONE")
