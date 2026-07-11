"""Point-based terrain feature lookup for the processed California DEM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import rasterio

from dem_pipeline.config import PipelineConfig, StudyArea

TERRAIN_LAYERS = ("elevation", "slope", "aspect", "hillshade", "tri", "tpi")


@dataclass(frozen=True)
class TerrainFeatures:
    latitude: float
    longitude: float
    elevation_m: float | None
    slope_deg: float | None
    aspect_deg: float | None
    hillshade: float | None
    tri: float | None
    tpi: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


class TerrainLookup:
    """Sample terrain features at latitude/longitude from processed raster files."""

    def __init__(
        self,
        terrain_dir: Path,
        study_area: StudyArea | None = None,
        nodata: float = -32768.0,
    ) -> None:
        self.terrain_dir = Path(terrain_dir)
        self.study_area = study_area
        self.nodata = nodata
        self._datasets: dict[str, rasterio.DatasetReader] = {}

    @classmethod
    def from_config(
        cls,
        config_path: Path | str = "config.yaml",
        project_root: Path | None = None,
    ) -> TerrainLookup:
        path = Path(config_path)
        config = PipelineConfig.from_yaml(path, project_root or path.parent)
        return cls(config.terrain_dir, config.study_area, config.nodata_value)

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()

    def __enter__(self) -> TerrainLookup:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _open(self, layer: str) -> rasterio.DatasetReader:
        if layer not in self._datasets:
            path = self.terrain_dir / f"{layer}.tif"
            if not path.exists():
                raise FileNotFoundError(f"Terrain layer not found: {path}")
            self._datasets[layer] = rasterio.open(path)
        return self._datasets[layer]

    def _in_bounds(self, latitude: float, longitude: float) -> bool:
        if self.study_area is None:
            return True
        area = self.study_area
        return (
            area.south <= latitude <= area.north
            and area.west <= longitude <= area.east
        )

    def _sample_layer(self, layer: str, longitude: float, latitude: float) -> float | None:
        value = next(self._open(layer).sample([(longitude, latitude)]))[0]
        if value is None or value == self.nodata:
            return None
        return float(value)

    def get_features(self, latitude: float, longitude: float) -> TerrainFeatures:
        """Return terrain features for a single latitude/longitude point."""
        if not self._in_bounds(latitude, longitude):
            raise ValueError(
                f"Point ({latitude}, {longitude}) is outside the California study area "
                f"[{self.study_area.south}, {self.study_area.north}] x "
                f"[{self.study_area.west}, {self.study_area.east}]"
                if self.study_area
                else f"Point ({latitude}, {longitude}) is outside the configured study area"
            )

        return TerrainFeatures(
            latitude=latitude,
            longitude=longitude,
            elevation_m=self._sample_layer("elevation", longitude, latitude),
            slope_deg=self._sample_layer("slope", longitude, latitude),
            aspect_deg=self._sample_layer("aspect", longitude, latitude),
            hillshade=self._sample_layer("hillshade", longitude, latitude),
            tri=self._sample_layer("tri", longitude, latitude),
            tpi=self._sample_layer("tpi", longitude, latitude),
        )

    def get_features_batch(
        self,
        points: Iterable[tuple[float, float]],
    ) -> list[TerrainFeatures]:
        """Return terrain features for multiple (latitude, longitude) points."""
        return [self.get_features(lat, lon) for lat, lon in points]


def get_terrain_features(
    latitude: float,
    longitude: float,
    *,
    terrain_dir: Path | str = "DEM/terrain",
    study_area: StudyArea | None = None,
    config_path: Path | str | None = None,
) -> TerrainFeatures:
    """
    Look up terrain features at a latitude/longitude in the California region.

    Examples:
        features = get_terrain_features(37.7749, -122.4194)  # San Francisco
        features.elevation_m
        features.to_dict()
    """
    if config_path is not None:
        with TerrainLookup.from_config(config_path) as lookup:
            return lookup.get_features(latitude, longitude)

    with TerrainLookup(Path(terrain_dir), study_area) as lookup:
        return lookup.get_features(latitude, longitude)
