import os
from pathlib import Path
from contextlib import contextmanager

from shapely.geometry import Polygon, Point, box
import rasterio
from rasterio.windows import Window, get_data_window, transform
from rasterio.merge import merge as merge_tool

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
        with get_back_workingdir(self.wbt.work_dir):
            self.wbt.flow_accumulation_full_workflow(
                str(self.raw_dem_path),
                out_dem=str(self.dem_path),
                out_pntr=str(self.d8_pntr),
                out_accum=str(self.flow_acc),
            )
        return 0

    def snap_points(self, inpoints, outpoints, snapdist=50):
        with get_back_workingdir(self.wbt.work_dir):
            self.wbt.snap_pour_points(
                pour_pts=str(inpoints.resolve()),
                flow_accum=str(self.flow_acc.resolve()),
                output=str(outpoints),
                snap_dist=snapdist,
            )
        return 0

    def watershed(self, inpoints, suffix="04_watershed"):
        wshed_path = self.out_folder / f"{self.name}_{suffix}.tif"
        with get_back_workingdir(self.wbt.work_dir):
            self.wbt.watershed(
                d8_pntr=str(self.d8_pntr),
                pour_pts=str(Path(inpoints).resolve()),
                output=str(wshed_path),
            )
        return 0


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
            "driver": "GTiff",
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
