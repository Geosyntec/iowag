"""Console script for iowag."""
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory


import click
from tqdm import tqdm

import pandas
import numpy

import rasterio
from shapely.geometry import Point
import fiona
import geopandas
import whitebox

from . import dem
from . import bmp


@click.group()
def main():
    pass


@main.command()
@click.option("--gdbfolder", default=".", help="Path to the collection of GDBs")
@click.option("--dstfolder", default=".", help="Output folder for shapefiles")
@click.option("--dry-run", is_flag=True)
@click.option("--overwrite", is_flag=True)
def preprocess_gdbs(gdbfolder, dstfolder, dry_run=False, overwrite=False):
    pbar = tqdm(list(Path(gdbfolder).glob(f"*.gdb")))
    for gdb in pbar:
        pbar.set_description(gdb.stem)
        outfile = Path(dstfolder) / gdb.stem / "BMPs.shp"
        if not dry_run and (overwrite or not outfile.exists()):
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
            outfile.parent.mkdir(exist_ok=True, parents=True)
            if not points.empty:
                points.to_file(outfile)
    return 0


def _get_vector_bounds(vectorpath):
    with fiona.open(vectorpath, "r") as ds:
        boundary = bmp.get_boundary_geom(ds)

    return {
        "bmppath": str(vectorpath),
        "geometry": boundary,
    }


@main.command()
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
    pbar = tqdm(Path(bmpfolder).glob("*/BMPs.shp"))
    if dry_run:
        paths = [p for p in pbar]
        print("\n".join(paths))
    else:
        boundaries = [_get_vector_bounds(dempath) for dempath in pbar]

        gdf = geopandas.GeoDataFrame(boundaries)
        gdf.to_file(dstfile)
    return 0


def _get_raster_boundary(rasterpath):
    with rasterio.open(rasterpath, "r") as ds:
        boundary = dem.get_boundary_geom(ds)

    return {
        "dempath": str(rasterpath),
        "geometry": boundary,
    }


@main.command()
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


@main.command()
@click.option("--bmppath", help="filepath of the spatial extents of the BMP layers")
@click.option("--dempath", help="filepath of the spatial extents of the DEM layers")
@click.option(
    "--outjson", help="filepath of the output JSON that will map BMPs to DEMs"
)
def assign_DEMs_to_BMPs(bmppath, dempath, outjson):
    # load the files of data extents
    bmps = geopandas.read_file(bmppath)
    dems = geopandas.read_file(dempath)

    # find DEMs that wholly contain the BMP extents (don't cross DEM boundires)
    bmpdem = geopandas.sjoin(
        bmps, dems, how="left", op="within", lsuffix="bmp", rsuffix="dem"
    ).pipe(lambda df: df.join(df.bounds))

    # select only the BMP extents that found a single DEM to fit inside
    has_dem = bmpdem.loc[
        bmpdem["dempath"].notnull(),
        ["bmppath", "dempath", "minx", "miny", "maxx", "maxy"],
    ].assign(dempath=lambda df: df["dempath"].apply(lambda x: [x]))

    # find the multiple DEMs for BMP extents that cross DEM boundaries
    bdcols = ["bmppath", "minx", "miny", "maxx", "maxy"]
    needs_dem = (
        bmpdem.loc[bmpdem["dempath"].isnull()]
        .dropna(axis="columns", how="all")
        .pipe(
            geopandas.sjoin,
            dems,
            how="left",
            op="intersects",
            lsuffix="bmp",
            rsuffix="dem",
        )
        .groupby("bmppath")["dempath"]
        .apply(lambda g: list(sorted(g)))
        .to_frame()
        .reset_index()
        .merge(bmpdem.loc[:, bdcols], how="left", on="bmppath")
    )

    all_bmps_dems = pandas.concat([has_dem, needs_dem], ignore_index=True)

    with Path(outjson).open("w") as out:
        json.dump(all_bmps_dems.to_dict(orient="records"), out)

    return 0


@main.command()
@click.argument("mapfile")
@click.option("--overwrite", is_flag=True)
def extract_zones(mapfile, overwrite):
    with open(mapfile, "r") as mapper:
        bmp_to_dems = json.load(mapper)

    for row in tqdm(bmp_to_dems):
        outfolder = Path(row["bmppath"]).parent
        final_dem = outfolder / "DEM.tif"
        if overwrite or not final_dem.exists():
            # spatial extent of the BMPs
            upperleft = Point(row["minx"], row["maxy"])
            lowerright = Point(row["maxx"], row["miny"])

            # create an emphemeral directory to store intermediate output
            with TemporaryDirectory() as tmpdir:
                if len(row["dempath"]) > 1:
                    # when the BMPs overlap more than one DEM, merge
                    # them and storm in the emphemeral directory
                    dempath = Path(tmpdir, "dem.img")
                    dem.merge_rasters(*row["dempath"], out_path=dempath)
                else:
                    # when the BMPs touch only one raster, use it directly
                    dempath = row["dempath"][0]

                # extract only the window of the DEM that overlaps the BMPs
                with rasterio.open(dempath, "r") as ds:
                    data, meta, zone = dem.extract_raster_window(
                        ds, upperleft, lowerright
                    )

            # save the extracted DEM to same folder as the BMP file
            with rasterio.open(final_dem, "w", **meta) as out:
                out.write(data, indexes=1)
    return 0


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
def determine_treated_areas(srcfolder):
    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = str(Path(srcfolder).resolve())
    wbt.verbose = False

    pbar = tqdm(list(Path(wbt.work_dir).glob("*/BMPs.shp")))
    for shpfile in pbar:
        pbar.set_description(f"Processing {shpfile.stem}")
        demfile = (shpfile.parent / "DEM.tif").resolve()
        snapfile = shpfile.parent.joinpath("BMPs_snapped.shp").resolve()
        dp = dem.DEMProcessor(wbt, demfile, shpfile.parent)

        pbar.set_description(f"Processing {shpfile.stem} (Flow Acc)")
        dp.flow_accumulation()

        pbar.set_description(f"Processing {shpfile.stem} (Snap Points)")
        dp.snap_points(shpfile, snapfile)

        pbar.set_description(f"Processing {shpfile.stem} (Watersheds)")
        dp.watershed(shpfile, suffix="04a_watershed")

        pbar.set_description(f"Processing {shpfile.stem} (Watersheds, snapped)")
        dp.watershed(snapfile, suffix="04b_watershed_snapped")


def lkjsdfljsdf():
    pass
