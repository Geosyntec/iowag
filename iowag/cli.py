"""Console script for iowag."""
from contextlib import closing
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from geopandas.tools.overlay import overlay
from tqdm import tqdm

import numpy
import pandas

import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, box
import fiona
import geopandas
from tqdm import cli
import whitebox

from . import dem
from . import bmp
from . import validate
from . import landcover


LAYERS = [
    {"name": "TERRACE", "fxn": bmp.process_terraces, "snap": False},
    {"name": "WASCOB", "fxn": bmp.process_WASCOBs, "snap": False},
    {"name": "POND_DAM", "fxn": bmp.process_pond_dams, "snap": False},
]


@click.group()
def main():
    pass


@main.command()
@click.option("--gdbfolder", default=".", help="Path to the collection of GDBs")
@click.option("--dstfolder", default=".", help="Output folder for shapefiles")
@click.option("--dry-run", is_flag=True)
@click.option("--overwrite", is_flag=True)
def merge_gdb_layers(
    gdbfolder,
    dstfolder,
    dry_run=False,
    overwrite=False,
):
    pbar = tqdm(list(Path(gdbfolder).glob(f"*.gdb")))
    for gdb in pbar:
        pbar.set_description(gdb.stem)
        outfile = Path(dstfolder) / gdb.stem / "_for_extents.shp"
        if not dry_run and (overwrite or not outfile.exists()):
            points = pandas.concat(
                [geopandas.read_file(gdb, layer=lyrinfo["name"]) for lyrinfo in LAYERS],
                ignore_index=True,
            )
            outfile.parent.mkdir(exist_ok=True, parents=True)
            if not points.empty:
                points.to_file(outfile)
    return 0


def _get_vector_bounds(vectorpath, offset):
    with fiona.open(vectorpath, "r") as ds:
        boundary = bmp.get_boundary_geom(ds, offset=offset)

    return {
        "bmppath": str(vectorpath),
        "geometry": boundary,
    }


@main.command()
@click.option(
    "--bmpfolder", default=".", help="Path to the collection of BMP shapefiles"
)
@click.option("--hucfile", default=".", help="Path to the collection of BMP shapefiles")
@click.option(
    "--dstfile",
    default="BMPs.geojson",
    help="file where the boundaries will be saved",
)
@click.option(
    "--huccol",
    default="huc12",
    help="Name of the column in `hucfile` with the HUC IDs",
)
@click.option("--offset", default=1000, help="Path to the layer of HUC12 boundaries")
def build_gdb_boundaries(bmpfolder, hucfile, dstfile, huccol="huc12", offset=1000):
    hucnames = [p.stem.split("_")[-1] for p in Path(bmpfolder).glob("Final_*.gdb")]

    dstfolder = Path(dstfile).parent
    hucgeos = geopandas.read_file(hucfile).loc[lambda df: df[huccol].isin(hucnames)]
    buffers = hucgeos.assign(
        bmppath=lambda df: str(dstfolder) + "/Final_" + df[huccol],
        geometry=lambda df: df.bounds.apply(lambda x: box(*x), axis=1)
        .pipe(geopandas.GeoSeries, crs=hucgeos.crs)
        .buffer(offset),
    )
    assert buffers.crs == hucgeos.crs
    buffers.to_file(dstfile)
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

        gdf = geopandas.GeoDataFrame(boundaries).set_crs(epsg=26915)
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
    )

    # select only the BMP extents that found a single DEM to fit inside
    has_dem = bmpdem.loc[bmpdem["dempath"].notnull(), ["bmppath", "dempath"]]

    # find the multiple DEMs for BMP extents that cross DEM boundaries
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
        .loc[:, ["bmppath", "dempath"]]
    )

    all_bmps_dems = (
        pandas.concat([has_dem, needs_dem], ignore_index=True)
        .groupby("bmppath")["dempath"]
        .apply(lambda x: list(set(x)))
        .to_frame()
        .join(bmps.set_index("bmppath").bounds)
        .reset_index()
    )

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
        outfolder = Path(row["bmppath"])
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
@click.option("--overwrite", is_flag=True)
def flow_accumulation(srcfolder, overwrite):
    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = str(Path(srcfolder).resolve())
    wbt.verbose = False

    pbar = tqdm(list(Path(wbt.work_dir).glob("*/DEM.tif")))
    for demfile in pbar:
        pbar.set_description(f"Processing {demfile.parent.name} (flow acc.)")
        dp = dem.DEMProcessor(wbt, demfile.resolve(), demfile.parent)
        if overwrite or (not dp.d8_pntr.exists()):
            dp.flow_accumulation()

    return 0


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
@click.option("--dstfolder", default=".", help="Output folder for shapefiles")
@click.option("--bmps_to_exclude", help="JSON file of BMPs to skip")
def compile_pourpoints(srcfolder, dstfolder, bmps_to_exclude):

    skippers = (
        pandas.read_json(bmps_to_exclude, dtype={"huc": str})
        .set_index(["huc", "layer"])
        .loc[:, "object_id"]
        .to_dict()
    )

    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = str(Path(".").resolve())
    wbt.verbose = False

    dstpath = Path(dstfolder)
    pbar = tqdm(list(Path(srcfolder).glob("*.gdb")))
    for gdb in pbar:
        pbar.set_description(gdb.stem)
        with TemporaryDirectory() as td:
            for lyrinfo in LAYERS:
                tmpfile = Path(td) / f"{lyrinfo['name']}.shp"
                raw_shapes = geopandas.read_file(gdb, layer=lyrinfo["name"])
                if not raw_shapes.empty:
                    huc = gdb.stem.split("_")[-1]
                    to_skip = skippers.get((huc, lyrinfo["name"]), None)
                    to_skip = validate.at_least_empty_list(to_skip)
                    points = (
                        raw_shapes.loc[lambda df: ~df.index.isin(to_skip)]
                        .pipe(lyrinfo["fxn"])
                        .assign(snapped=lyrinfo["snap"])
                        .reset_index()
                        .drop(columns=["index"])
                    )
                    points.to_file(tmpfile)
                    if lyrinfo["snap"]:
                        demfile = dstpath / gdb.stem / "DEM.tif"
                        dp = dem.DEMProcessor(wbt, demfile, dstpath / gdb.stem)
                        dp.snap_points(tmpfile, tmpfile)

            outdir = dstpath / gdb.stem
            _gdfs = [geopandas.read_file(shp) for shp in Path(td).glob("*.shp")]
            if any([not gdf.empty for gdf in _gdfs]):
                all_points = pandas.concat(
                    _gdfs,
                    ignore_index=True,
                    axis=0,
                )
                outdir.parent.mkdir(exist_ok=True, parents=True)
                for col in ["isin80s", "isin2010", "isin2016"]:
                    subset = (
                        all_points.loc[lambda df: df[col].eq(1)]
                        .reset_index(drop=True)
                        .rename_axis("usid", axis="index")
                    )
                    outfile = outdir / f"BMP_{col}.shp"
                    if not subset.empty:
                        subset.to_file(outfile)
    return 0


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
def determine_treated_areas(srcfolder):
    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = str(Path(srcfolder).resolve())
    wbt.verbose = False

    pbar = tqdm(list(Path(wbt.work_dir).glob("*/BMP*.shp")))
    for shpfile in pbar:
        pbar.set_description(f"Processing {shpfile.stem}")
        demfile = (shpfile.parent / "DEM.tif").resolve()
        dp = dem.DEMProcessor(wbt, demfile, shpfile.parent)
        pbar.set_description(f"Processing {shpfile.stem} (Watersheds)")
        dp.watershed(shpfile, suffix=f"04_watershed_{shpfile.stem}")


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
@click.option(
    "--globber", help="wildcard string that will glob all of the watershed files"
)
@click.option(
    "--shpprefix",
    default="BMP",
    help="filename prefix of the shapefiles with BMPs for each era",
)
@click.option(
    "--outprefix",
    default="DEM_05_BMPTypes",
    help="filename prefix of the shapefiles with BMPs for each era",
)
def categorize_watersheds(
    srcfolder, globber, shpprefix="BMP", outprefix="DEM_05_BMPTypes"
):
    for wshed in tqdm(Path(srcfolder).glob(globber)):
        era = wshed.stem.split("_")[-1]
        bmptypes = wshed.parent / f"{outprefix}_{era}.tif"
        bmpshp = wshed.parent / f"{shpprefix}_{era}.shp"
        dem.categorize_watersheds(wshed, bmptypes, bmpshp)
    return 0


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
@click.option(
    "--globber", help="wildcard string that will glob all of the watershed files"
)
@click.option(
    "--outfilename", help="filename of the CSV and pickle files that will be saved"
)
def compile_treated_areas(srcfolder, globber, outfilename):
    treated_areas = pandas.concat(
        [
            dem.get_treated_areas(bmpraster)
            for bmpraster in tqdm(Path(srcfolder).glob(globber))
        ],
        ignore_index=True,
    )
    treated_areas.to_csv(
        Path(srcfolder).joinpath(outfilename + ".csv"), encoding="utf-8", index=False
    )

    treated_areas.to_pickle(Path(srcfolder).joinpath(outfilename + ".pkl"))
    return 0


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
@click.option(
    "--globber", help="wildcard string that will glob all of the watershed files"
)
@click.option("--outfilename", help="filename of the merged tif that will be saved")
def merge_treatment_rasters(srcfolder, globber, outfilename):
    eras = ["isin80s", "isin2010", "isin2016"]
    for era in eras:
        dem.merge_rasters(
            *list(Path(srcfolder).glob(f"{globber}_{era}.tif")),
            out_path=Path(f"01-Pilot-Watersheds/02-prepared/{outfilename}_{era}.tif"),
        )
    return 0


@main.command()
@click.option("--to_clip", help="full path of the raster to be clipped")
@click.option("--template", help="full path of the template raster")
@click.option("--dstfile", help="full path of the output raster")
def clip_raster_to_another(to_clip, template, dstfile):
    with rasterio.open(to_clip, "r") as ds_to_clip:
        with rasterio.open(template, "r") as ds_template:
            ul_row, ul_col = ds_to_clip.index(
                ds_template.bounds.left, ds_template.bounds.top
            )
            lr_row, lr_col = ds_to_clip.index(
                ds_template.bounds.right, ds_template.bounds.bottom
            )
            window = Window(ul_col, ul_row, (lr_col - ul_col), (lr_row - ul_row))

            options = ds_template.meta.copy()
            clipped_data = ds_to_clip.read(1, window=window)
            assert ds_template.meta["crs"] == ds_to_clip.meta["crs"]

    options["dtype"] = rasterio.dtypes.uint8
    with rasterio.open(dstfile, "w", **options) as ds_clipped:
        ds_clipped.write(clipped_data, 1)

    return 0


@main.command()
@click.option("--file1", help="full path the first polygon file")
@click.option("--file2", help="full path the second polygon file")
@click.option("--dstfile", help="full path of the layer")
def vector_intersection(file1, file2, dstfile):
    mlra = geopandas.read_file(file1)
    huc = geopandas.read_file(file2)
    assert mlra.crs == huc.crs

    mlra_hucs = geopandas.overlay(huc, mlra, how="intersection")
    mlra_hucs.to_file(dstfile)
    return 0


@main.command()
@click.option(
    "--srcfolder", help="top-level directory containing all of the pre-processed data"
)
@click.option(
    "--globber", help="wildcard string that will glob all of the watershed files"
)
@click.option("--lcfile", help="land cover/treatement raster data set to summarize")
@click.option("--polygonfile", help="shapefile of polygons for the raster summary")
@click.option("--gpkg", help="path of the GeoPackage where results will be written")
@click.option(
    "--layerprefix",
    help="name of the layer within --gpkg where results will be written",
)
def treated_landuse_stats(srcfolder, globber, lcfile, polygonfile, gpkg, layerprefix):
    topdir = Path(srcfolder)
    validhucs = [folder.name.split("Final_")[-1] for folder in topdir.glob("Final_*")]
    with rasterio.open(topdir / lcfile, "r") as ds_lc:
        lcdata = landcover.get_aggplus_pixels(ds_lc.read(1))

    for treatedpath in tqdm(topdir.glob(globber)):
        era = treatedpath.stem.split("_")[-1]
        with rasterio.open(treatedpath, "r") as ds_treated:
            aff = ds_treated.meta["transform"]
            crs = ds_treated.meta["crs"]
            treated = ds_treated.read(1)

        # do this inplace to save memory
        treated += lcdata

        with fiona.open(topdir / polygonfile, "r") as hucs:
            rstats = landcover.zone_stats(treated, aff, hucs, crs, validhucs)

        rstats.to_file(topdir / gpkg, layer=f"{layerprefix}_{era}", driver="GPKG")
    return 0
