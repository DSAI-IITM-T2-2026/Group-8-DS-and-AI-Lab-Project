"""Deterministic 1 km fishnet grid over the AOI."""

from __future__ import annotations

import json
import math
from functools import lru_cache

import ee

from .config import AppConfig


@lru_cache(maxsize=8)
def _geojson_features(path: str) -> tuple[dict, ...]:
    with open(path) as f:
        payload = json.load(f)
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": payload}]
    if not features:
        raise ValueError(f"AOI GeoJSON has no features: {path}")
    return tuple(features)


def aoi_geometry(cfg: AppConfig) -> ee.Geometry:
    """Exact AOI geometry loaded from the configured local GeoJSON."""
    features = list(_geojson_features(str(cfg.aoi.geojson_path)))
    return ee.FeatureCollection(features).geometry(cfg.aoi.error_margin_m)


def _projected_bounds(cfg: AppConfig) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in the grid CRS (client-side)."""
    # Transform AOI corners via EE once and get info — cached per config identity.
    region = aoi_geometry(cfg)
    proj = ee.Projection(cfg.grid.crs)
    bounds = region.transform(proj, 1).bounds(1, proj)
    coords = bounds.coordinates().getInfo()
    # coords = [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], ...]]
    ring = coords[0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def build_grid(
    cfg: AppConfig, *, use_asset: bool = True
) -> ee.FeatureCollection:
    """
    Build (or load) the reusable 1 km grid FeatureCollection.

    Each feature has: grid_id, latitude, longitude, geometry.
    ``grid_id`` is ``{ix}_{iy}`` from the cell SW corner in CRS **meters**
    (EPSG:3310), rounded to ``resolution_m``. IDs are independent of reduce
    tiles / export shards.
    """
    if use_asset and cfg.grid.asset_id:
        return ee.FeatureCollection(cfg.grid.asset_id)

    return _build_grid_ee(cfg)


def _build_grid_ee(cfg: AppConfig) -> ee.FeatureCollection:
    """Server-side fishnet — no client iteration over cells."""
    res = cfg.grid.resolution_m
    # atScale is required for coveringGrid cell size, but MUST NOT be used when
    # reading coordinates for grid_id (it scales the affine transform; dividing
    # by res again collapses millions of cells into a handful of IDs like -1_0).
    proj_grid = ee.Projection(cfg.grid.crs).atScale(res)
    crs_m = ee.Projection(cfg.grid.crs)
    region = aoi_geometry(cfg)

    try:
        cells = region.coveringGrid(proj_grid, res)
    except Exception:
        cells = _covering_grid_fallback(region, proj_grid, res)

    def _annotate(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry()
        # Stable ID from cell SW corner in CRS meters (not atScale units).
        bounds = geom.bounds(1, crs_m)
        ring = ee.List(ee.List(bounds.coordinates()).get(0))
        xs = ring.map(lambda p: ee.List(p).get(0))
        ys = ring.map(lambda p: ee.List(p).get(1))
        xmin = ee.Number(xs.reduce(ee.Reducer.min()))
        ymin = ee.Number(ys.reduce(ee.Reducer.min()))
        # coveringGrid snaps to res; round guards tiny float error
        ix = xmin.divide(res).round().int()
        iy = ymin.divide(res).round().int()
        grid_id = ix.format("%d").cat("_").cat(iy.format("%d"))
        lonlat = geom.centroid(1).transform("EPSG:4326", 1).coordinates()
        return feature.set(
            {
                "grid_id": grid_id,
                "ix": ix,
                "iy": iy,
                "longitude": lonlat.get(0),
                "latitude": lonlat.get(1),
            }
        )

    return ee.FeatureCollection(cells.map(_annotate))


def _covering_grid_fallback(
    region: ee.Geometry, proj: ee.Projection, res: float
) -> ee.FeatureCollection:
    """Fallback fishnet if coveringGrid is unavailable."""
    bounds = region.transform(proj, 1).bounds(1, proj)
    coords = ee.List(ee.List(bounds.coordinates()).get(0))
    xs = coords.map(lambda p: ee.List(p).get(0))
    ys = coords.map(lambda p: ee.List(p).get(1))
    xmin = ee.Number(xs.reduce(ee.Reducer.min())).divide(res).floor().multiply(res)
    ymin = ee.Number(ys.reduce(ee.Reducer.min())).divide(res).floor().multiply(res)
    xmax = ee.Number(xs.reduce(ee.Reducer.max())).divide(res).ceil().multiply(res)
    ymax = ee.Number(ys.reduce(ee.Reducer.max())).divide(res).ceil().multiply(res)

    xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(res), res)
    yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(res), res)

    def _row(y: ee.Number) -> ee.FeatureCollection:
        y = ee.Number(y)

        def _cell(x: ee.Number) -> ee.Feature:
            x = ee.Number(x)
            rect = ee.Geometry.Rectangle(
                [x, y, x.add(res), y.add(res)], proj, False
            )
            return ee.Feature(rect)

        return ee.FeatureCollection(xx.map(_cell))

    return ee.FeatureCollection(yy.map(_row)).flatten()


def grid_size_estimate(cfg: AppConfig) -> int:
    """Approximate number of 1 km cells (for status / logging)."""
    xmin, ymin, xmax, ymax = _projected_bounds(cfg)
    res = cfg.grid.resolution_m
    nx = max(1, int(math.ceil((xmax - xmin) / res)))
    ny = max(1, int(math.ceil((ymax - ymin) / res)))
    return nx * ny
