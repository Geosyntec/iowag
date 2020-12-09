import numpy
from shapely.geometry import Point, LineString, Polygon, box, MultiPoint
import geopandas

from . import dem


BMPCOLS = ["PRACTICE", "Present80s", "Present2010", "Present2016", "geometry"]


def get_boundary_geom(in_ds, offset=1000):
    return box(*in_ds.bounds).buffer(offset)


def _smooth_line(line: LineString, res: int = 3):
    return MultiPoint(
        [line.interpolate(dist) for dist in numpy.arange(0, line.length, res)]
    )


def _mid_point(line: LineString):
    return line.interpolate(0.5, normalized=True)


def _fix_ints(df, cols):
    df = df.assign(**{c: df[c].replace({None: 0}).astype(int) for c in cols})
    return df


def process_terraces(bmp_gdf: geopandas.GeoDataFrame, name: str = "terrace"):
    """
    Get representative points (locations) for each BMP feature
    in a geopandas geodataframe
    """

    points = (
        bmp_gdf.explode()
        .reset_index(drop=True)
        .loc[:, BMPCOLS]
        .pipe(_fix_ints, filter(lambda c: c.startswith("Present"), BMPCOLS))
        .rename(columns=lambda c: c.lower().replace("present", "isin"))
        .assign(geometry=lambda gdf: gdf.geometry.apply(_smooth_line))
        .explode()
        .rename_axis(["obj_id", "geo_id"], axis="index")
        .reset_index()
    )

    return points


def process_WASCOBs(bmp_gdf: geopandas.GeoDataFrame):
    return process_terraces(bmp_gdf, name="wascob")


def process_pond_dams(bmp_gdf: geopandas.GeoDataFrame):
    """
    Get representative points (locations) for each BMP feature
    in a geopandas geodataframe
    """
    return process_terraces(bmp_gdf, name="pond dam")
