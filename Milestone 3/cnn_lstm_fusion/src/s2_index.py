from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_TILE_RE = re.compile(r"s2_(\d{4})_(\d{2})(\d+)-(\d+)\.tif$")

_LAT_SPLIT = 33.616754562320736
_LON_SPLIT = -115.90153627238479


def list_s2_tiles(gcs_prefix: str) -> pd.DataFrame:
    """List monthly S2 tiles on GCS and parse year/month/offsets."""
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    from google.cloud.storage import Client

    without = gcs_prefix.replace("gs://", "", 1).rstrip("/")
    bucket_name, prefix = without.split("/", 1)
    client = Client.create_anonymous_client()
    rows = []
    for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
        name = blob.name.rsplit("/", 1)[-1]
        m = _TILE_RE.match(name)
        if not m:
            continue
        year, month, yoff, xoff = m.groups()
        rows.append(
            {
                "filename": name,
                "year": int(year),
                "month": int(month),
                "y_offset": int(yoff),
                "x_offset": int(xoff),
                "gs_uri": f"gs://{bucket_name}/{blob.name}",
                "vsigs_path": f"/vsigs/{bucket_name}/{blob.name}",
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No S2 tiles found under {gcs_prefix}")
    logger.info(
        "Indexed %d S2 tiles (%d year-months)",
        len(df),
        df.groupby(["year", "month"]).ngroups,
    )
    return df.sort_values(["year", "month", "y_offset", "x_offset"]).reset_index(drop=True)


def available_year_months(tile_index: pd.DataFrame) -> list[tuple[int, int]]:
    pairs = (
        tile_index[["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
        .itertuples(index=False, name=None)
    )
    return list(pairs)


def resolve_month_for_date(
    date: pd.Timestamp,
    available: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Forward-fill: most recent available (year, month) <= date's year-month."""
    y, m = date.year, date.month
    candidates = [(ay, am) for ay, am in available if (ay, am) <= (y, m)]
    if not candidates:
        return None
    return candidates[-1]


def offsets_for_point(lon: float, lat: float) -> tuple[int, int]:
    y_offset = 0 if lat >= _LAT_SPLIT else 9472
    x_offset = 0 if lon < _LON_SPLIT else 9472
    return y_offset, x_offset


def pick_tile_for_point(
    tile_index: pd.DataFrame,
    year: int,
    month: int,
    lon: float,
    lat: float,
) -> dict | None:
    y_off, x_off = offsets_for_point(lon, lat)
    match = tile_index[
        (tile_index["year"] == year)
        & (tile_index["month"] == month)
        & (tile_index["y_offset"] == y_off)
        & (tile_index["x_offset"] == x_off)
    ]
    if len(match) == 0:
        return _pick_tile_bounds_fallback(tile_index, year, month, lon, lat)
    return match.iloc[0].to_dict()


@lru_cache(maxsize=64)
def _tile_bounds(vsigs_path: str) -> tuple[float, float, float, float]:
    import rasterio

    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    with rasterio.open(vsigs_path) as src:
        b = src.bounds
        return (b.left, b.bottom, b.right, b.top)


def _pick_tile_bounds_fallback(
    tile_index: pd.DataFrame,
    year: int,
    month: int,
    lon: float,
    lat: float,
) -> dict | None:
    month_tiles = tile_index[(tile_index["year"] == year) & (tile_index["month"] == month)]
    for row in month_tiles.itertuples(index=False):
        left, bottom, right, top = _tile_bounds(row.vsigs_path)
        if left <= lon <= right and bottom <= lat <= top:
            return {
                "filename": row.filename,
                "year": row.year,
                "month": row.month,
                "y_offset": row.y_offset,
                "x_offset": row.x_offset,
                "gs_uri": row.gs_uri,
                "vsigs_path": row.vsigs_path,
            }
    return None


def resolve_read_path(tile: dict, tiles_dir: Path | None) -> str:
    if tiles_dir is not None:
        local = Path(tiles_dir) / tile["filename"]
        if local.exists() and local.stat().st_size > 0:
            return str(local)
    return tile["vsigs_path"]
