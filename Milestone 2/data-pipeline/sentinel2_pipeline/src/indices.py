"""
Derived vegetation/moisture/burn indices.

Only used in `alt_design` mode — the confirmed/production logic does not
compute any of these; it exports raw reflectance bands only.
"""


def add_indices_alt(image):
    b2, b4, b8, b11, b12 = (
        image.select("B2"),
        image.select("B4"),
        image.select("B8"),
        image.select("B11"),
        image.select("B12"),
    )
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    ndmi = b8.subtract(b11).divide(b8.add(b11)).rename("NDMI")
    nbr = b8.subtract(b12).divide(b8.add(b12)).rename("NBR")
    evi = (
        b8.subtract(b4)
        .multiply(2.5)
        .divide(b8.add(b4.multiply(6.0)).subtract(b2.multiply(7.5)).add(1.0))
        .rename("EVI")
    )
    savi = b8.subtract(b4).multiply(1.5).divide(b8.add(b4).add(0.5)).rename("SAVI")
    return image.addBands([ndvi, ndmi, nbr, evi, savi])
