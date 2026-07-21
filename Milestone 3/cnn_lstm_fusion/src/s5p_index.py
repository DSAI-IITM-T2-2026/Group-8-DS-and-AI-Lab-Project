from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_S5P_RE = re.compile(r"s5p_(\d{4})_(\d{2})\.tif$")


def list_s5p_tiles(gcs_prefix: str) -> pd.DataFrame:
    """List monthly S5P mosaics: s5p_YYYY_MM.tif (one file per month)."""
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    from google.cloud.storage import Client

    without = gcs_prefix.replace("gs://", "", 1).rstrip("/")
    bucket_name, prefix = without.split("/", 1)
    client = Client.create_anonymous_client()
    rows = []
    for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
        name = blob.name.rsplit("/", 1)[-1]
        m = _S5P_RE.match(name)
        if not m:
            continue
        year, month = m.groups()
        rows.append(
            {
                "filename": name,
                "year": int(year),
                "month": int(month),
                "gs_uri": f"gs://{bucket_name}/{blob.name}",
                "vsigs_path": f"/vsigs/{bucket_name}/{blob.name}",
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No S5P tiles found under {gcs_prefix}")
    logger.info("Indexed %d S5P monthly tiles", len(df))
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def available_year_months(tile_index: pd.DataFrame) -> list[tuple[int, int]]:
    return list(
        tile_index[["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
        .itertuples(index=False, name=None)
    )


def resolve_month_for_date(
    date: pd.Timestamp,
    available: list[tuple[int, int]],
) -> tuple[int, int] | None:
    y, m = date.year, date.month
    candidates = [(ay, am) for ay, am in available if (ay, am) <= (y, m)]
    if not candidates:
        return None
    return candidates[-1]


def pick_tile(tile_index: pd.DataFrame, year: int, month: int) -> dict | None:
    match = tile_index[(tile_index["year"] == year) & (tile_index["month"] == month)]
    if len(match) == 0:
        return None
    return match.iloc[0].to_dict()


def resolve_read_path(tile: dict, tiles_dir: Path | None) -> str:
    if tiles_dir is not None:
        local = Path(tiles_dir) / tile["filename"]
        if local.exists() and local.stat().st_size > 0:
            return str(local)
    return tile["vsigs_path"]
