from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_tiles(
    gs_uris: list[str],
    tiles_dir: Path,
    skip_existing: bool = True,
) -> list[Path]:
    """Download unique S2 GeoTIFFs from GCS into tiles_dir via gsutil."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for uri in tqdm(gs_uris, desc="Download S2 tiles"):
        name = uri.rstrip("/").rsplit("/", 1)[-1]
        dest = tiles_dir / name
        local_paths.append(dest)
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("Downloading %s", uri)
        subprocess.check_call(
            ["gsutil", "-q", "cp", uri, str(dest)],
        )
    return local_paths
