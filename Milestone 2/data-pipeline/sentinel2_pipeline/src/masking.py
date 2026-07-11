"""
Cloud masking — two implementations, kept in separate functions so it's
never ambiguous which one is running.
"""


def mask_clouds_qa60_confirmed(image):
    """CONFIRMED logic: SCL (reliable on S2_SR_HARMONIZED) AND QA60 when present.

    QA60 alone is insufficient: on COPERNICUS/S2_SR_HARMONIZED after ~2022,
    QA60 is often unpopulated (all zeros), which would leave clouds unmasked.
    SCL classes kept: 4 vegetation, 5 not vegetated, 6 water, 7 unclassified.
    QA60 bits 10/11 still applied so older scenes retain the original bitmask.
    """
    scl = image.select("SCL")
    scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))

    qa = image.select("QA60")
    cloud_bit, cirrus_bit = 1 << 10, 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))

    return image.updateMask(scl_mask.And(qa_mask))


def mask_clouds_scl_alt(image):
    """ALT_DESIGN logic: Scene Classification Layer, keep classes 4/5/6/7."""
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(mask)
