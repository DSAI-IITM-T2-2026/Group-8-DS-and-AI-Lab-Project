"""One GCS list per prefix so skip_existing is not one HEAD per day.

Positive cache only: a name in the listing → skip. A name missing from the
listing → caller downloads (listing is not treated as proof of absence later).
Wait loops must keep using live blob.exists() so newly written objects are seen.
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger("download.gcs_listing")

_CACHE: dict[tuple[str, str], set[str]] = {}


def _norm_prefix(prefix: str) -> str:
    return prefix.rstrip("/") + "/"


def list_names(bucket: str, prefix: str, *, project: str | None = None) -> set[str]:
    key = (bucket, _norm_prefix(prefix))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    from google.cloud import storage

    client = storage.Client(project=project) if project else storage.Client()
    pfx = key[1]
    names = {blob.name for blob in client.list_blobs(bucket, prefix=pfx)}
    _CACHE[key] = names
    logger.info("GCS listed gs://%s/%s (%d objects)", bucket, pfx, len(names))
    return names


def blob_listed(
    bucket: str,
    blob_name: str,
    *,
    prefix: str | None = None,
    project: str | None = None,
) -> bool:
    pfx = prefix if prefix is not None else blob_name.rsplit("/", 1)[0]
    return blob_name in list_names(bucket, pfx, project=project)


def remember(bucket: str, blob_name: str) -> None:
    """Record a blob written in this process so later skip checks see it."""
    pfx = blob_name.rsplit("/", 1)[0]
    key = (bucket, _norm_prefix(pfx))
    names = _CACHE.get(key)
    if names is not None:
        names.add(blob_name)


def era5_daily_blob(prefix: str, day: date) -> str:
    return (
        f"{prefix.rstrip('/')}/{day.year:04d}/"
        f"era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
    )


def prefetch_download_prefixes(
    bucket: str,
    prefixes: dict,
    *,
    years: list[int],
    project: str | None = None,
) -> None:
    for key in ("firms", "sentinel2", "sentinel5p"):
        if key in prefixes:
            list_names(bucket, prefixes[key], project=project)
    era5 = prefixes.get("era5")
    if not era5:
        return
    base = era5.rstrip("/")
    for year in years:
        list_names(bucket, f"{base}/{year}", project=project)
        list_names(bucket, f"{base}/raw/{year}", project=project)
