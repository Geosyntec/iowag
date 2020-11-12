import numpy
from shapely.geometry import Point, LineString, Polygon, box, MultiPoint
import geopandas

from . import dem


def get_boundary_geom(in_ds, offset=1000):
    return box(*in_ds.bounds).buffer(offset)


def _smooth_line(line: LineString, res: int = 3):
    return [line.interpolate(dist) for dist in numpy.arange(0, line.length, res)]


def _mid_point(line: LineString):
    return line.interpolate(0.5, normalized=True)


def process_terraces(bmp_gdf: geopandas.GeoDataFrame, name: str = "terrace"):
    """
    Get representative points (locations) for each BMP feature
    in a geopandas geodataframe
    """

    points = (
        bmp_gdf.explode()
        .rename_axis(["obj_id", "geo_id"], axis="index")
        .geometry.apply(_smooth_line)
        .explode()
        .rename("geometry")
        .reset_index()
        .pipe(geopandas.GeoDataFrame, geometry="geometry", crs=bmp_gdf.crs)
        .assign(bmp=name)
    )

    return points


def process_WASCOBs(bmp_gdf: geopandas.GeoDataFrame):
    return process_terraces(bmp_gdf, name="wascob")


def process_pond_dams(bmp_gdf: geopandas.GeoDataFrame):
    """
    Get representative points (locations) for each BMP feature
    in a geopandas geodataframe
    """

    points = (
        bmp_gdf.explode()
        .rename_axis(["obj_id", "geo_id"], axis="index")
        .geometry.apply(_mid_point)
        .to_frame()
        .reset_index()
        .assign(bmp="pond dam")
    )

    return points
