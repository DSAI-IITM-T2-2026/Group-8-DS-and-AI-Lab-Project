"""Copernicus DEM tile identification utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dem_pipeline.config import StudyArea


@dataclass(frozen=True)
class DemTile:
    lat: int
    lon: int

    @property
    def name(self) -> str:
        return f"Copernicus_DSM_COG_10_{self._format_lat()}_{self._format_lon()}_DEM"

    @property
    def filename(self) -> str:
        return f"{self.name}.tif"

    @property
    def s3_key(self) -> str:
        return f"{self.name}/{self.filename}"

    def _format_lat(self) -> str:
        if self.lat >= 0:
            return f"N{self.lat:02d}_00"
        return f"S{abs(self.lat):02d}_00"

    def _format_lon(self) -> str:
        if self.lon >= 0:
            return f"E{self.lon:03d}_00"
        return f"W{abs(self.lon):03d}_00"


def identify_tiles(study_area: StudyArea) -> list[DemTile]:
    """Return all 1-degree Copernicus DEM tiles intersecting the study area."""
    lat_min = int(math.floor(study_area.south))
    lat_max = int(math.ceil(study_area.north)) - 1
    if study_area.north > lat_max + 1:
        lat_max = int(math.floor(study_area.north))

    lon_min = int(math.floor(study_area.west))
    lon_max = int(math.ceil(study_area.east)) - 1
    if study_area.east > lon_max + 1:
        lon_max = int(math.floor(study_area.east))

    tiles: list[DemTile] = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tiles.append(DemTile(lat=lat, lon=lon))

    return tiles
