from __future__ import annotations

import logging
import os
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    """gs://bucket/path/to/obj → (bucket, path/to/obj)."""
    without = gs_uri.replace("gs://", "", 1)
    bucket, _, blob_name = without.partition("/")
    if not bucket or not blob_name:
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    return bucket, blob_name


def download_gs_uri(gs_uri: str, dest: Path) -> Path:
    """
    Download a GCS object with the Python client (no gsutil subprocess).

    Avoids macOS SIGSEGV from forking after Network/OpenSSL init
    ('crashed on child side of fork pre-exec'), which is common when
    calling gsutil from a process that already imported google-cloud-storage
    / cryptography — especially on Python 3.14.
    """
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    from google.cloud.storage import Client

    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    logger.info("Downloading %s → %s", gs_uri, dest.name)
    client = Client.create_anonymous_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        blob.download_to_filename(str(tmp))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return dest


def download_tiles(
    gs_uris: list[str],
    tiles_dir: Path,
    skip_existing: bool = True,
) -> list[Path]:
    """Download unique GeoTIFFs from GCS into tiles_dir."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for uri in tqdm(gs_uris, desc="Download tiles"):
        name = uri.rstrip("/").rsplit("/", 1)[-1]
        dest = tiles_dir / name
        local_paths.append(dest)
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            continue
        download_gs_uri(uri, dest)
    return local_paths


def ensure_local_tile(gs_uri: str, dest: Path) -> Path:
    return download_gs_uri(gs_uri, dest)
