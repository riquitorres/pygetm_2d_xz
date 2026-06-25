# If PyFVCOM is not installed, you can install it with pip:
# pip install PyFVCOM
# or the development version from GitHub

# Script to download the necesary data to run a 2D simulation
from datetime import datetime, timedelta
import os

import pyproj

# import PyFVCOM as pf

# import calendar
from pathlib import Path
import cmems_download as cmems

import matplotlib.pyplot as plt
import geopandas as gpd

import numpy as np

from setup_2D_model import build_parser, DEFAULTS
# """Script to create a centerline of nodes following the deepest part of an estuarine channel, define roughly euqlly seperated sections and estimate a representative mean depth for each section.
# If the estuary has an FVCOM mesh, extract the centerline with PyFVCOM """

# DEFAULTS = {
#     "start_date": "2023-01-01",
#     "end_date": "2023-01-31",
#     "estuary_name": "tamar",
#     "utm_epsg": 32630,  # UTM zone 30N
#     "output_data_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data"),
#     "output_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/outputs"),
#     "output_fig_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/figures"),
#     "tamar_mesh": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/tamar_grid/tamar_v2_grd.dat"),
#     "transect_pickle_out": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/transect_grid/tamar_v2_transect.pk"),
#     "channel_key": "channel_nodes",
#     "section_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/tamar_grid/tamar_sections.geojson"),
#     "coastline_shapefile": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/shapefile/0_depth_limited_coastline_singleline.geojsonl"),
#     "out_png": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/figures/tamar_axial_50m.png"),
#     "bbox" : [-5.5, 50.0, -4.5, 51.0],
#     "minimum_depth": 0.5057600140571594,
#     "maximum_depth": 200,
#     "check_polygons_plot": False,}
# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Create estuary setup for 2D model")
#     parser.add_argument(
#         "--estuary-name",
#         type=str,
#         default=DEFAULTS["estuary_name"],
#         help="Name of the estuary (used for file naming)",
#     )
#     parser.add_argument(
#         "--utm-epsg",
#         type=int,
#         default=DEFAULTS["utm_epsg"],
#         help="EPSG code for UTM zone of the estuary (used for coordinate transformations)",
#     )
#     parser.add_argument(
#         "--mesh",
#         type=Path,
#         default=DEFAULTS["tamar_mesh"],
#         help="Path to mesh file (.2dm)",
#     )
#     # process or load pre-configured section nodes
#     parser.add_argument(
#         "--calculate-transects",
#         action="store_true",
#         help="Flag to calculate transect sections from mesh instead of loading from geojson file",
#     )
#     parser.add_argument(
#         "--transect-pickle",
#         type=Path,
#         default=DEFAULTS["transect_pickle_out"],
#         help="Pickle with ordered channel node IDs",
#     )
#     parser.add_argument(
#         "--channel-key",
#         default=DEFAULTS["channel_key"],
#         help="Pickle key for channel node sequence",
#     )
#     parser.add_argument(
#         "--out-png",
#         type=Path,
#         default=DEFAULTS["out_png"],
#         help="Output PNG path",
#     )
#     parser.add_argument(
#         "--mesh-linewidth",
#         type=float,
#         default=0.12,
#         help="Line width for mesh edges",
#     )
#     parser.add_argument(
#         "--mesh-alpha",
#         type=float,
#         default=0.28,
#         help="Alpha for mesh lines",
#     )
#     parser.add_argument(
#         "--node-size",
#         type=float,
#         default=5.0,
#         help="Size of transect node markers",
#     )
#     parser.add_argument(
#         "--bbox",
#         nargs=4,
#         type=float,
#         default=DEFAULTS["bbox"],
#         help="Bounding box to restrict the search for channel nodes, format: min_lon min_lat max_lon max_lat",
#     )
#     # do-plots
#     parser.add_argument(
#         "--do-plots",
#         action="store_true",
#         help="Flag to enable plotting",
#     )
#     # start and end date
#     parser.add_argument(
#         "--start-date",
#         type=str,
#         default=DEFAULTS["start_date"],
#         help="Start date for CMEMS data download (YYYY-MM-DD)",
#     )
#     parser.add_argument(
#         "--end-date",
#         type=str,
#         default=DEFAULTS["end_date"],
#         help="End date for CMEMS data download (YYYY-MM-DD)",
#     )
#     # data_dir 
#     parser.add_argument(
#         "--output-data-dir",
#         type=Path,
#         default=DEFAULTS["output_data_dir"],
#         help="Directory to save downloaded data",
#     )
#     # fig dir
#     parser.add_argument(
#         "--output-fig-dir",
#         type=Path,
#         default=DEFAULTS["output_fig_dir"],
#         help="Directory to save figures",
#     )
#     # section file
#     parser.add_argument(
#         "--section-file",
#         type=Path,
#         default=DEFAULTS["section_file"],
#         help="Path to section geojson file",
#     )
#     # coastline shapefile
#     parser.add_argument(
#         "--coastline-shapefile",
#         type=Path,
#         default=DEFAULTS["coastline_shapefile"],
#         help="Path to coastline shapefile",
#     )
#     return parser
if __name__ == "__main__":
    args = build_parser().parse_args()
    # Extract the estuary middle point 
    # convert start and end date to datetime objects for use in CMEMS data download
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.download_cmems_data:
        # download T and S cmems data for the same period
        # salt is only available at daily frequency in the reanalysis product. For more recent data, the near-real-time product is available at hourly frequency, but only for the last 4 years. The reanalysis product is available from 1993 to present.
        cmems.download_cmems_data(
            bbox=[args.bbox[0], args.bbox[2], args.bbox[1], args.bbox[3]],
            start_date=args.start_date,
            end_date=args.end_date,
            download_vars=["temp", "salt", "N1_p", "N3_n", "N4_n", "N5_s", "O2_o", "O3_c"] # "u1", "v1", "zt",  have a different time frequency so will need processing separately if wanted instead of TPXO forcing 
        )
        # load coastline shapefile to get the boundary point for CMEMS data extraction
        coastline = gpd.read_file(args.coastline_shapefile)
        # extract a single point time series of T and S at the boundary point for use as boundary conditions in the 2D model.
        # fill in depth profiles with nearest non-nan value to avoid issues with missing data at the surface or bottom.
        bdy_lon, bdy_lat = coastline.to_crs("EPSG:32630").centroid.x.item(), coastline.to_crs("EPSG:32630").centroid.y.item()  # use the centroid of the coastline as the boundary point for CMEMS data extraction
        transformer = pyproj.Transformer.from_crs("EPSG:32630",  "EPSG:4326",  always_xy=True) #(UTM zone 30N to WGS84 lat/lon)
        bdy_lon, bdy_lat = transformer.transform(bdy_lon, bdy_lat)

        cmems.extract_boundary_conditions(
            bbox=[args.bbox[0], args.bbox[2], args.bbox[1], args.bbox[3]],
            start_date=args.start_date,
            end_date=args.end_date,
            bdy_lon=bdy_lon,
            bdy_lat=bdy_lat,
            output_dir=args.output_data_dir,
            vars2proc=["temp", "salt", "N1_p", "N3_n", "N4_n", "N5_s", "O2_o", "O3_c", ] # "u1", "v1", "zt",  have a different time frequency so will need processing separately if wanted instead of TPXO forcing 
        )
    if args.download_era5_data:
        # download meteorological data for the same period for potential later use in forcing the model
        import runpy
        import sys
        for year in range(int(start_date.year), int(end_date.year) + 1):
            sys.argv = ["igotm",  str(bdy_lon),  str(bdy_lat), str(year), os.path.join(args.output_data_dir, f"era5_{year}.nc")]
            runpy.run_module("pygetm.input.igotm", run_name="__main__")
            # alternatively, you can call the function directly


    # save the centerline, distance along channel and depths to a pickle for later use
    # python plot_transect.py --mesh ../tamar_v0/tamar_v2_grd.dat  --transect-pickle ../transect_grid/tamar_v2_transect.pk --channel-key "channel_nodes"  --out-png ../tamar_axial_50m.png