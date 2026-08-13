"""Reusable Sentinel-2 spectral index functions.

Optical bands are expected as surface reflectance in [0, 1]
(i.e. S2 DN * 0.0001). Ratio indices cancel scale, but EVI / SAVI / MSAVI
require absolute reflectance.
"""

from __future__ import annotations

import ee


def ndvi(nir: ee.Image, red: ee.Image) -> ee.Image:
    return nir.subtract(red).divide(nir.add(red)).rename("NDVI")


def ndmi(nir: ee.Image, swir1: ee.Image) -> ee.Image:
    return nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")


def nbr(nir: ee.Image, swir2: ee.Image) -> ee.Image:
    return nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")


def ndwi(green: ee.Image, nir: ee.Image) -> ee.Image:
    """McFeeters NDWI."""
    return green.subtract(nir).divide(green.add(nir)).rename("NDWI")


def evi(
    blue: ee.Image,
    red: ee.Image,
    nir: ee.Image,
    *,
    min_abs_denominator: float = 0.05,
    max_abs: float = 2.0,
) -> ee.Image:
    """EVI with explicit numerical QA; invalid observations remain masked."""
    # 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    numerator = nir.subtract(red).multiply(2.5)
    denominator = nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    value = numerator.divide(denominator)
    stable = (
        denominator.abs().gte(min_abs_denominator)
        .And(value.abs().lte(max_abs))
    )
    return value.updateMask(stable).rename("EVI")


def savi(nir: ee.Image, red: ee.Image, l: float = 0.5) -> ee.Image:
    # (1 + L) * (NIR - Red) / (NIR + Red + L)
    return (
        nir.subtract(red)
        .multiply(1.0 + l)
        .divide(nir.add(red).add(l))
        .rename("SAVI")
    )


def msavi(nir: ee.Image, red: ee.Image) -> ee.Image:
    # 0.5 * (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red)))
    two_nir_plus_1 = nir.multiply(2).add(1)
    inside = two_nir_plus_1.pow(2).subtract(nir.subtract(red).multiply(8))
    return two_nir_plus_1.subtract(inside.sqrt()).multiply(0.5).rename("MSAVI")


def add_indices(
    image: ee.Image,
    *,
    evi_min_abs_denominator: float = 0.05,
    evi_max_abs: float = 2.0,
) -> ee.Image:
    """
    Attach NDVI, NDMI, NBR, NDWI, EVI, SAVI, MSAVI to an S2 reflectance image.

    Expects bands B2, B3, B4, B8, B11, B12 (reflectance 0–1).
    """
    blue = image.select("B2")
    green = image.select("B3")
    red = image.select("B4")
    nir = image.select("B8")
    swir1 = image.select("B11")
    swir2 = image.select("B12")

    return image.addBands(
        [
            ndvi(nir, red),
            ndmi(nir, swir1),
            nbr(nir, swir2),
            ndwi(green, nir),
            evi(
                blue,
                red,
                nir,
                min_abs_denominator=evi_min_abs_denominator,
                max_abs=evi_max_abs,
            ),
            savi(nir, red),
            msavi(nir, red),
        ]
    )


INDEX_NAMES = ("NDVI", "NDMI", "NBR", "NDWI", "EVI", "SAVI", "MSAVI")

# Indices that get mean/std/min/max vs mean/std only (per output schema)
INDEX_FULL_STATS = ("NDVI", "NDMI", "NBR", "NDWI", "EVI")
INDEX_MEAN_STD = ("SAVI", "MSAVI")
