"""
Composite builder — for each active product, filters to the window and
composites per-band with a `mean` reducer, falling back to a fully masked
(nodata) band if the product has no scenes in that window. Then stacks all
active product bands into one multi-band image.
"""

import ee

from .config import get_active_products


def _nodata_band(output_band: str):
    """Constant band that is fully masked so GeoTIFF writes nodata, not zeros."""
    return ee.Image.constant(0.0).rename(output_band).updateMask(ee.Image.constant(0))


def build_product_band(product: dict, start: str, end: str, aoi, composite_fn: str):
    coll = ee.ImageCollection(product["collection"]).filterBounds(aoi).filterDate(start, end)
    has_data = coll.size().gt(0)
    band_img = ee.Algorithms.If(
        has_data,
        getattr(coll, composite_fn)().select(product["source_band"]).rename(product["output_band"]),
        _nodata_band(product["output_band"]),
    )
    return ee.Image(band_img), coll.size()


def build_composite(start: str, end: str, aoi, cfg: dict):
    """
    Returns (composite_image_or_None, total_scene_count_across_products).
    composite is None if every active product had zero scenes in this window.
    """
    s5p_cfg = cfg["sentinel5p"]
    products = get_active_products(cfg)

    band_images = []
    total_scenes = 0
    for product in products:
        band_img, size = build_product_band(product, start, end, aoi, s5p_cfg["composite"])
        band_images.append(band_img)
        total_scenes += size.getInfo()

    if total_scenes == 0:
        return None, 0

    stacked = ee.Image.cat(band_images).clip(aoi).resample("bilinear")
    return stacked, total_scenes
