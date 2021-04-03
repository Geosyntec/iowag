import numpy
import pandas

import rasterio
import rasterstats
import fiona
import geopandas


def get_corn_soy_pixels(array, corncode=10, soycode=11):
    return numpy.isin(array, [corncode, soycode]).astype(numpy.int8)


def get_aggplus_pixels(array, cornsoy=None, agplus=None):
    # default values
    if not cornsoy:
        cornsoy = [10, 11]
    if not agplus:
        agplus = [4, 5, 7, 8, 9, 12]

    return numpy.select(
        [numpy.isin(array, cornsoy), numpy.isin(array, agplus)],
        [numpy.int8(10), numpy.int8(20)],
        default=numpy.int8(0),
    )


def zone_stats(treated_landcover, raster_affine, polygons, crs, valid_hucs):
    codemap = {
        0: "Other Untreated",
        1: "Other Above Pond Dams",
        2: "Other Above Terraces",
        3: "Other Above WACOBS",
        10: "Corn/Soy Untreated",
        11: "Corn/Soy Above Pond Dams",
        12: "Corn/Soy Above Terraces",
        13: "Corn/Soy Above WACOBS",
        20: "Agplus Untreated",
        21: "Agplus Above Pond Dams",
        22: "Agplus Above Terraces",
        23: "Agplus Above WACOBS",
    }

    ## 2021 to include MLRAs
    _by_huc = rasterstats.zonal_stats(
        filter(lambda r: r["properties"]["huc12"] in valid_hucs, polygons),
        treated_landcover,
        affine=raster_affine,
        categorical=True,
        geojson_out=True,
        cmap=codemap,
    )

    by_huc = (
        geopandas.GeoDataFrame.from_features(_by_huc, crs=crs)
        .rename(columns=codemap)
        .fillna(0)
    )

    return by_huc
