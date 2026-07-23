"""AOI intersection across FIRMS, Sentinel-2, Sentinel-5P, ERA5, DEM."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from src import config
from src.data.gcs import get_fs
from src.data.loaders import (
    list_era5_year_folder,
    list_sentinel2_tiles,
    list_sentinel5p_files,
    load_firms_raster,
    load_sentinel2_tile,
    load_sentinel5p_file,
    open_era5_file,
)


def _bounds_from_rio(da_or_src) -> dict:
    """Return west/south/east/north from an rioxarray DA or rasterio dataset."""
    if hasattr(da_or_src, "rio"):
        b = da_or_src.rio.bounds()  # (minx, miny, maxx, maxy)
        return {"west": float(b[0]), "south": float(b[1]), "east": float(b[2]), "north": float(b[3])}
    b = da_or_src.bounds
    return {"west": float(b.left), "south": float(b.bottom), "east": float(b.right), "north": float(b.top)}


def _intersect(a: dict, b: dict) -> dict:
    return {
        "west": max(a["west"], b["west"]),
        "south": max(a["south"], b["south"]),
        "east": min(a["east"], b["east"]),
        "north": min(a["north"], b["north"]),
    }


def _union(a: Optional[dict], b: dict) -> dict:
    if a is None:
        return dict(b)
    return {
        "west": min(a["west"], b["west"]),
        "south": min(a["south"], b["south"]),
        "east": max(a["east"], b["east"]),
        "north": max(a["north"], b["north"]),
    }


def compute_source_bounds(sample_s2_tiles: int = 8) -> dict:
    """Probe real file bounds for each of the five remaining sources."""
    print("Computing per-source bounds...")

    firms = load_firms_raster("2024-08-15")
    firms_b = _bounds_from_rio(firms)
    print(f"  FIRMS: {firms_b}")

    # Sentinel-2: union of ALL 2024 tile *headers* (bounds only — no pixel load)
    s2_tiles = list_sentinel2_tiles()
    s2_2024 = [t for t in s2_tiles if "_2024_" in t] or s2_tiles
    s2_union = None
    import rasterio as rio

    for path in s2_2024:
        uri = path if path.startswith("gs://") else f"gs://{path}"
        try:
            with rio.open(uri) as src:
                s2_union = _union(s2_union, _bounds_from_rio(src))
        except Exception as exc:
            print(f"    S2 skip {path.split('/')[-1]}: {exc}")
    if s2_union is None:
        s2_union = dict(config.DEFAULT_AOI_BOUNDS)
    print(f"  Sentinel-2 (header union of {len(s2_2024)} tiles): {s2_union}")
    s5p_files = sorted(list_sentinel5p_files())
    s5p_2024 = [f for f in s5p_files if "s5p_2024_" in f] or s5p_files
    s5p = load_sentinel5p_file(f"gs://{s5p_2024[0]}")
    s5p_b = _bounds_from_rio(s5p)
    print(f"  Sentinel-5P: {s5p_b}")

    era5_files = list_era5_year_folder(2024)
    era5 = open_era5_file(era5_files[0])
    lat = np.asarray(era5["latitude"].values)
    lon = np.asarray(era5["longitude"].values)
    era5_b = {
        "west": float(np.min(lon)),
        "south": float(np.min(lat)),
        "east": float(np.max(lon)),
        "north": float(np.max(lat)),
    }
    print(f"  ERA5: {era5_b}")

    # DEM: open GeoTIFF header only via GDAL/rasterio (no full download)
    dem_uri = f"gs://{config.GCS_BUCKET}/{config.PREFIX_DEM_TERRAIN}/elevation.tif"
    try:
        with rasterio.open(dem_uri) as src:
            dem_b = _bounds_from_rio(src)
            if src.crs and "4326" not in str(src.crs):
                left, bottom, right, top = transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds
                )
                dem_b = {"west": left, "south": bottom, "east": right, "north": top}
    except Exception as exc:
        print(f"  DEM remote open failed ({exc}); falling back to DEFAULT_AOI_BOUNDS")
        dem_b = dict(config.DEFAULT_AOI_BOUNDS)
    print(f"  DEM: {dem_b}")

    return {
        "FIRMS": firms_b,
        "Sentinel-2": s2_union,
        "Sentinel-5P": s5p_b,
        "ERA5": era5_b,
        "DEM": dem_b,
    }


def lock_intersection_aoi(source_bounds: Optional[dict] = None) -> dict:
    if source_bounds is None:
        source_bounds = compute_source_bounds()

    keys = ["FIRMS", "Sentinel-2", "Sentinel-5P", "ERA5", "DEM"]
    inter = source_bounds[keys[0]]
    for k in keys[1:]:
        inter = _intersect(inter, source_bounds[k])

    if inter["west"] >= inter["east"] or inter["south"] >= inter["north"]:
        raise RuntimeError(f"Empty AOI intersection: {inter}")

    out = {
        "aoi_bounds": inter,
        "per_source_bounds": source_bounds,
        "notes": (
            "Intersection of FIRMS, Sentinel-2 (sampled tile union), "
            "Sentinel-5P, ERA5, and DEM. Landsat excluded."
        ),
    }
    out_path = config.METADATA_DIR / "aoi_bounds.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Locked AOI written to {out_path}")
    print(f"  AOI_BOUNDS = {inter}")
    return out


def load_locked_aoi() -> dict:
    path = config.METADATA_DIR / "aoi_bounds.json"
    if not path.exists():
        raise FileNotFoundError(f"Run AOI lock first: {path}")
    with open(path) as f:
        return json.load(f)["aoi_bounds"]
