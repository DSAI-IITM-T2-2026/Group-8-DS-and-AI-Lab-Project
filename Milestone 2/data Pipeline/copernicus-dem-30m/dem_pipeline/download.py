"""Parallel tile download with retry support."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from dem_pipeline.config import PipelineConfig
from dem_pipeline.tiles import DemTile

logger = logging.getLogger("dem_pipeline.download")


def _download_tile(
    tile: DemTile,
    config: PipelineConfig,
    session: requests.Session,
) -> tuple[DemTile, Path | None, str | None]:
    url = f"{config.base_url}/{tile.s3_key}"
    output_path = config.raw_dir / tile.filename

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Skipping existing tile: %s", tile.filename)
        return tile, output_path, None

    last_error: str | None = None
    for attempt in range(1, config.download_retries + 1):
        try:
            response = session.get(url, stream=True, timeout=120)
            if response.status_code == 404:
                return tile, None, f"Tile not found (404): {url}"

            response.raise_for_status()

            temp_path = output_path.with_suffix(".tif.part")
            with temp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            temp_path.replace(output_path)
            logger.info("Downloaded %s (attempt %d)", tile.filename, attempt)
            return tile, output_path, None

        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning(
                "Download failed for %s (attempt %d/%d): %s",
                tile.filename,
                attempt,
                config.download_retries,
                exc,
            )
            if attempt < config.download_retries:
                time.sleep(config.retry_delay_seconds)

    return tile, None, last_error


def download_tiles(tiles: list[DemTile], config: PipelineConfig) -> dict[str, Path]:
    """Download all tiles in parallel and return mapping of tile name to local path."""
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    failures: list[str] = []

    with requests.Session() as session, ThreadPoolExecutor(
        max_workers=config.workers
    ) as executor:
        futures = {
            executor.submit(_download_tile, tile, config, session): tile
            for tile in tiles
        }
        for future in as_completed(futures):
            tile, path, error = future.result()
            if path is not None:
                downloaded[tile.name] = path
            elif error:
                failures.append(f"{tile.name}: {error}")

    if failures:
        logger.warning("%d tile(s) failed to download", len(failures))
        for failure in failures:
            logger.warning("  %s", failure)

    logger.info("Downloaded %d / %d tiles", len(downloaded), len(tiles))
    return downloaded
