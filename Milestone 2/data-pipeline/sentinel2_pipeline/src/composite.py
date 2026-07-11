"""
Composite builder — filters the S2 collection to a window, applies the
logic-mode-appropriate cloud filter/mask, and builds the median composite.
"""

import ee

from .indices import add_indices_alt
from .masking import mask_clouds_qa60_confirmed, mask_clouds_scl_alt


def build_composite(start: str, end: str, aoi, cfg: dict):
    """
    Returns (composite_image_or_None, scene_count).
    composite is None if no scenes matched the filters for this window.
    """
    s2cfg = cfg["sentinel2"]
    mode = cfg.get("logic_mode", "confirmed")
    cloud_pct = 30 if mode == "alt_design" else s2cfg["cloud_filter_pct"]

    coll = (
        ee.ImageCollection(s2cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )

    count = coll.size().getInfo()
    if count == 0:
        return None, count

    masked = coll.map(mask_clouds_scl_alt if mode == "alt_design" else mask_clouds_qa60_confirmed)
    composite = getattr(masked, s2cfg["composite"])().select(s2cfg["bands"])

    if mode == "alt_design":
        composite = add_indices_alt(composite)

    return composite.clip(aoi), count
