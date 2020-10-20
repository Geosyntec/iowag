"""Console script for iowag."""
import sys
from pathlib import Path

import click
from tqdm import tqdm

import pandas
import numpy

import rasterio
import fiona
import geopandas

from . import dem
from . import bmp


@click.group()
def main():
    pass


@click.group()
def dem():
    pass


@click.group()
def bmp():
    pass


def _get_raster_boundary(rasterpath):
    with rasterio.open(rasterpath, "r") as ds:
        boundary = dem.get_boundary_geom(ds)

    return {
        "folder": str(rasterpath.parent),
        "filename": rasterpath.stem,
        "geometry": boundary,
    }


def _get_vectory_bounds(vectorpath):
    with fiona.open(vectorpath, "r") as ds:
        boundary = bmp.get_boundary_geom(ds)

    return {
        "folder": str(vectorpath.parent),
        "filename": vectorpath.stem,
        "geometry": boundary,
    }


@dem.command()
@click.option("--demfolder", default=".", help="Path to the collection of DEMs")
@click.option("--ext", default="tif", help="the file extension of the DEMs")
@click.option(
    "--dstfile",
    default="rasters.geojson",
    help="file where the boundaries will be saved",
)
def build_raster_boundaries(demfolder, ext, dstfile):
    pbar = tqdm(Path(demfolder).glob(f"*/*.{ext}"))
    boundaries = [_get_raster_boundary(dempath) for dempath in pbar]

    gdf = geopandas.GeoDataFrame(boundaries)
    gdf.to_file(dstfile, driver="GeoJSON")
    return 0


@bmp.command()
@click.option("--gdbfolder", default=".", help="Path to the collection of GDBs")
@click.option(
    "--dstfile",
    default="rasters.geojson",
    help="file where the boundaries will be saved",
)
def build_gdb_boundaries(gdbfolder, dstfile):
    pbar = tqdm(Path(gdbfolder).glob(f"*/*.gdb"))
    boundaries = [_get_raster_boundary(dempath) for dempath in pbar]

    gdf = geopandas.GeoDataFrame(boundaries)
    gdf.to_file(dstfile, driver="GeoJSON")
    return 0


@main.command
@click.argument("srcdem", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    "dstfolder", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("bmppath", type=click.Path(exists=False))
@click.option("--offset", type=click.INT)
def clip_dem_to_bmps(srcdem, dstfolder, bmppath, offset=1000):
    with rasterio.open(srcdem, "r") as src, fiona.open(bmppath, "r") as shp:
        xmin, ymin, xmax, ymax = shp.bounds
        upperleft = Point(xmin - offset, ymax + offset)
        lowerright = Point(xmax + offset, ymin - offset)

        extracted, meta, zone = dem.extract_raster_window(src, upperleft, lowerright)

    with rasterio.open(dstfolder, "w", **meta) as dst:
        dst.write(extracted, indexes=1)


def process_input_data(gdbpath, dempath, outputpath):
    for gdb in tqdm(Path(gdbpath).glob("*.gdb")):
        points = pandas.concat(
            [
                geopandas.read_file(gdb, layer=lyr)
                .pipe(fxn)
                .assign(bmp=lyr)
                .reset_index()
                for lyr, fxn in zip(
                    ["TERRACE", "WASCOB", "POND_DAM"],
                    [bmp.process_terraces, bmp.process_WASCOBs, bmp.process_pond_dams],
                )
            ],
            ignore_index=True,
        ).drop(columns=["index"])
