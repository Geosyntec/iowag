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


def _get_raster_boundary(rasterpath):
    with rasterio.open(rasterpath, "r") as ds:
        boundary = dem.get_boundary_geom(ds)

    return {
        "folder": str(rasterpath.parent),
        "filename": rasterpath.stem,
        "geometry": boundary,
    }


def _get_vector_bounds(vectorpath):
    with fiona.open(vectorpath, "r") as ds:
        boundary = bmp.get_boundary_geom(ds)

    return {
        "folder": str(vectorpath.parent),
        "filename": vectorpath.stem,
        "geometry": boundary,
    }


@click.group()
def iowadem():
    pass


@iowadem.command()
@click.option("--demfolder", default=".", help="Path to the collection of DEMs")
@click.option("--ext", default="tif", help="the file extension of the DEMs")
@click.option(
    "--dstfile",
    default="rasters.geojson",
    help="file where the boundaries will be saved",
)
@click.option("--dry-run", is_flag=True)
def build_raster_boundaries(demfolder, ext, dstfile, dry_run=False):
    pbar = tqdm(Path(demfolder).glob(f"*/*.{ext}"))
    if dry_run:
        paths = [p for p in pbar]
        print("\n".join(paths))
    else:
        boundaries = [_get_raster_boundary(dempath) for dempath in pbar]

        gdf = geopandas.GeoDataFrame(boundaries)
        gdf.to_file(dstfile)
    return 0


@click.group()
def iowabmp():
    pass


@iowabmp.command()
@click.option("--gdbfolder", default=".", help="Path to the collection of GDBs")
@click.option("--dstfolder", default=".", help="Output folder for shapefiles")
@click.option("--dry-run", is_flag=True)
def preprocess_gdbs(gdbfolder, dstfolder, dry_run=False):
    pbar = tqdm(Path(gdbfolder).glob(f"*.gdb"))
    for gdb in pbar:
        pbar.set_description(gdb.stem)
        if not dry_run:
            points = pandas.concat(
                [
                    geopandas.read_file(gdb, layer=lyr)
                    .pipe(fxn)
                    .assign(bmp=lyr)
                    .reset_index()
                    for lyr, fxn in zip(
                        ["TERRACE", "WASCOB", "POND_DAM"],
                        [
                            bmp.process_terraces,
                            bmp.process_WASCOBs,
                            bmp.process_pond_dams,
                        ],
                    )
                ],
                ignore_index=True,
            ).drop(columns=["index"])
            points.to_file(Path(dstfolder) / gdb.stem / "BMPs.shp")
    return 0


@iowabmp.command()
@click.option(
    "--bmpfolder", default=".", help="Path to the collection of BMP shapefiles"
)
@click.option(
    "--dstfile",
    default="BMPs.geojson",
    help="file where the boundaries will be saved",
)
@click.option("--dry-run", is_flag=True)
def build_gdb_boundaries(bmpfolder, dstfile, dry_run=False):
    pbar = tqdm(Path(bmpfolder).glob("*.shp"))
    if dry_run:
        paths = [p for p in pbar]
        print("\n".join(paths))
    else:
        boundaries = [_get_vector_bounds(dempath) for dempath in pbar]

        gdf = geopandas.GeoDataFrame(boundaries)
        gdf.to_file(dstfile)
    return 0


@click.group()
def main():
    pass


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
