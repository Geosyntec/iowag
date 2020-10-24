import os
from pathlib import Path
from contextlib import contextmanager

from shapely.geometry import Polygon, Point, box
import rasterio
from rasterio.windows import Window, get_data_window, transform
from rasterio.merge import merge as merge_tool

import whitebox

from . import validate


@contextmanager
def get_back_workingdir(workingdir):
    """
    path. It's dumb. This is how we work around it while it's
    not fixed.
    """
    yield workingdir
    os.chdir(workingdir)


class DEMProcessor:
    def __init__(self, wbt, dem_path, output_path):
        self.wbt = wbt
        self.raw_dem_path = Path(dem_path).resolve()
        self.name = Path(dem_path).stem
        self.out_folder = Path(output_path).resolve()

        self.dem_path = self.out_folder / f"{self.name}_01_filled_dem.tif"
        self.d8_pntr = self.out_folder / f"{self.name}_02_flow_dir_d8.tif"
        self.flow_acc = self.out_folder / f"{self.name}_03_flow_accum.tif"

    def flow_accumulation(self):
        if not self.flow_acc.exists():
            with get_back_workingdir(self.wbt.work_dir):
                self.wbt.flow_accumulation_full_workflow(
                    str(self.raw_dem_path),
                    out_dem=str(self.dem_path),
                    out_pntr=str(self.d8_pntr),
                    out_accum=str(self.flow_acc),
                )
        return

    def watershed(self, in_flowdir, path_to_points, bmp):
        wshed_path = self.out_folder / f"{self.name}_{bmp}_04_watershed.tif"
        with get_back_workingdir(self.wbt.work_dir):
            self.wbt.watershed(
                d8_pntr=str(self.d8_pntr),
                pour_pts=str(Path(path_to_points).resolve()),
                output=str(wshed),
            )
        return wshed_path


def extract_raster_window(in_ds, upperleft, lowerright):
    meta = in_ds.meta.copy()
    start_row, start_col = in_ds.index(upperleft.x, upperleft.y)
    stop_row, stop_col = in_ds.index(lowerright.x, lowerright.y)
    width = stop_col - start_col
    height = stop_row - start_row
    zone = Window(start_col, start_row, width, height)
    data = in_ds.read(1, window=zone)

    meta.update(
        {
            "height": zone.height,
            "width": zone.width,
            "transform": transform(zone, in_ds.transform),
        }
    )

    return data, meta, zone


def raster_values_at_points(ds, geoseries):
    validate.same_crs(ds.crs, geoseries.crs)
    raster = ds.read(1)
    values = geoseries.apply(lambda p: raster[ds.index(p.x, p.y)])
    return values


def flag_treated_areas(in_ds, flag_value):
    outmeta = in_ds.meta.copy()
    data = in_ds.read(1)

    outmeta["dtype"] = rasterio.dtypes.int8
    values = (data > 0).astype(numpy.int8) * flag_value
    return values, outmeta


def get_boundary_geom(in_ds):
    w = in_ds.meta["width"]
    h = in_ds.meta["height"]

    upperleft = Point(in_ds.xy(0, 0))
    lowerright = Point(in_ds.xy(h, w))
    return box(upperleft.x, lowerright.y, lowerright.x, upperleft.y)


def merge_rasters(*in_paths, out_path):
    dest, xform = merge_tool([str(d) for d in in_paths])

    with rasterio.open(in_paths[0], "r") as first:
        profile = first.profile
        profile["transform"] = xform
        profile["height"] = dest.shape[1]
        profile["width"] = dest.shape[2]
        profile["count"] = dest.shape[0]

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dest)


"""
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

dst_crs = 'EPSG:4326'

with rasterio.open('rasterio/tests/data/RGB.byte.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
"""
