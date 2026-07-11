"""Copernicus DEM GLO-30 acquisition and processing pipeline."""

from dem_pipeline.lookup import TerrainFeatures, TerrainLookup, get_terrain_features

__version__ = "1.0.0"

__all__ = [
    "TerrainFeatures",
    "TerrainLookup",
    "get_terrain_features",
]
