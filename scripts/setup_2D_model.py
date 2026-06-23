# If PyFVCOM is not installed, you can install it with pip:
# pip install PyFVCOM
# or the development version from GitHub

# Script to download the necesary data to run a 2D simulation
import importlib
from datetime import datetime, timedelta
import os
import pygetm
import argparse
import getpass

import pyproj

import estimate_channel_section_volumes as ecs
import make_grid_areas as mga
# import PyFVCOM as pf
from itertools import islice
import numpy as np

# import calendar
from pathlib import Path
# import subprocess
# import shutil
# import itertools
import Slice2D_estuary as s2d
import cmems_download as cmems
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, mapping, shape, MultiLineString, GeometryCollection
from shapely.validation import explain_validity, make_valid

from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LinearRing, box
from shapely.validation import make_valid, explain_validity
from shapely.geometry import box
# order linestrings with nearest neighbour to try to get a single linestring. This is needed to get the area of the intersection with the coastline
from shapely.ops import linemerge, unary_union, snap

import numpy as np


# """Script to create a centerline of nodes following the deepest part of an estuarine channel, define roughly euqlly seperated sections and estimate a representative mean depth for each section.
# If the estuary has an FVCOM mesh, extract the centerline with PyFVCOM """

DEFAULTS = {
    "start_date": "2023-01-01",
    "end_date": "2023-01-31",
    "estuary_name": "tamar",
    "utm_epsg": 32630,  # UTM zone 30N
    "output_data_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data"),
    "output_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/outputs"),
    "output_fig_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/figures"),
    "tamar_mesh": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/tamar_grid/tamar_v2_grd.dat"),
    "transect_pickle_out": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/transect_grid/tamar_v2_transect.pk"),
    "channel_key": "channel_nodes",
    "section_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/tamar_grid/tamar_sections.geojson"),
    "coastline_shapefile": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/shapefile/0_depth_limited_coastline_singleline.geojsonl"),
    "out_png": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/figures/tamar_axial_50m.png"),
    "bbox" : [-5.5, 50.0, -4.5, 51.0],
    "minimum_depth": 0.5057600140571594,
    "maximum_depth": 200,
    "check_polygons_plot": False,}
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create estuary setup for 2D model")
    parser.add_argument(
        "--estuary-name",
        type=str,
        default=DEFAULTS["estuary_name"],
        help="Name of the estuary (used for file naming)",
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=DEFAULTS["utm_epsg"],
        help="EPSG code for UTM zone of the estuary (used for coordinate transformations)",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULTS["tamar_mesh"],
        help="Path to mesh file (.2dm)",
    )
    # process or load pre-configured section nodes
    parser.add_argument(
        "--calculate-transects",
        action="store_true",
        help="Flag to calculate transect sections from mesh instead of loading from geojson file",
    )
    parser.add_argument(
        "--transect-pickle",
        type=Path,
        default=DEFAULTS["transect_pickle_out"],
        help="Pickle with ordered channel node IDs",
    )
    parser.add_argument(
        "--channel-key",
        default=DEFAULTS["channel_key"],
        help="Pickle key for channel node sequence",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=DEFAULTS["out_png"],
        help="Output PNG path",
    )
    parser.add_argument(
        "--mesh-linewidth",
        type=float,
        default=0.12,
        help="Line width for mesh edges",
    )
    parser.add_argument(
        "--mesh-alpha",
        type=float,
        default=0.28,
        help="Alpha for mesh lines",
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=5.0,
        help="Size of transect node markers",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=DEFAULTS["bbox"],
        help="Bounding box to restrict the search for channel nodes, format: min_lon min_lat max_lon max_lat",
    )
    # do-plots
    parser.add_argument(
        "--do-plots",
        action="store_true",
        help="Flag to enable plotting",
    )
    # start and end date
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULTS["start_date"],
        help="Start date for CMEMS data download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULTS["end_date"],
        help="End date for CMEMS data download (YYYY-MM-DD)",
    )
    # data_dir 
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULTS["output_data_dir"],
        help="Directory to save downloaded data",
    )
    # fig dir
    parser.add_argument(
        "--output-fig-dir",
        type=Path,
        default=DEFAULTS["output_fig_dir"],
        help="Directory to save figures",
    )
    # section file
    parser.add_argument(
        "--section-file",
        type=Path,
        default=DEFAULTS["section_file"],
        help="Path to section geojson file",
    )
    # coastline shapefile
    parser.add_argument(
        "--coastline-shapefile",
        type=Path,
        default=DEFAULTS["coastline_shapefile"],
        help="Path to coastline shapefile",
    )
    return parser

args = build_parser().parse_args()
# Extract the estuary middle point 
mesh = ecs.load_mesh(args.mesh)
if args.calculate_transects:
    if args.mesh.suffix == ".dat":
        # boundaring box to restrict the search for the channel nodes, this is to avoid picking up nodes from nearby estuaries or coastal areas
        # in the coordinates of the mesh
        bounding_box = [[406648.5, 421944.5], [5571183.0, 5599817.8]]
        # this is hard coded for the tamar as the initial point to start the node transect is split in two sections.
        output = s2d._extract_transect_nodes_from_mesh(mesh, bounding_box)
        channel_nodes = output["channel_nodes"]
    else:
        channel_nodes = ecs.load_channel_nodes(args.transect_pickle, key=args.channel_key)

    if args.do_plots:
        s2d.plot_mesh_and_transect(
            mesh=mesh,
            channel_nodes=channel_nodes,
            out_png=args.out_png,
            mesh_linewidth=args.mesh_linewidth,
            mesh_alpha=args.mesh_alpha,
            node_size=args.node_size,
        )
        print(f"Saved: {args.out_png}")
    # save transect nodes to pickle for later use
    if args.transect_pickle:
        with args.transect_pickle.open("wb") as f:
            pickle.dump({args.channel_key: channel_nodes}, f)
    print(f"Mesh nodes: {mesh.nodes.size}, elements: {mesh.triangles.shape[0]}")
    print(f"Transect nodes: {len(channel_nodes)}")

    # Calculate distance along the channel and extract depths
    # get x and y positions of the channel nodes
    # resample nodes to desired spacing along the channel. 
    # channel nodes have a repeated first node ... drop it 
    channel_nodes_slim = np.array(list(dict.fromkeys(channel_nodes)), dtype=int)
    channel_nodes_fixed_spacing, distance = ecs.resample_centerline(channel_nodes_slim,  spacing=200, distance=output["distance"])
    centerline = ecs.centerline_from_nodes(mesh, channel_nodes_fixed_spacing)
    depths = mesh.depth[channel_nodes_fixed_spacing]
    # make quick plot of s vs depth to check it looks reasonable, save the figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distance, depths, marker="o", markersize=4, linestyle="-", color="#1e40af", label="Depth along channel")   
    ax.set_xlabel("Distance along channel (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Depth along channel")
    ax.legend()
    plt.savefig(DEFAULTS["output_fig_dir"] / "depth_along_channel.png")
    plt.close(fig)

    # get perpendicular transects for each centerline point
    # wThe normals should be slowly varying along the channel... using local tangents makes it difficult not to have wildly varying normals. Instead, calculate the normal segments using a smoothed centerline with a large smoothing window to get more stable normals. The half_width is set to 5000m to ensure the normal segments extend well beyond the channel width for better visualization and potential later use in defining cross-channel sections.

    norm_segments = ecs.normal_segments(centerline, half_width=5000)
    norm_segments = ecs.smooth_normals(centerline, smoothing_factor=1000, half_width=5000)
    # print the section dict to a JSON file for potential later use

    # ecs.write_geojson_sections(args.output_data_dir / f"{"tamar"}_sections.geojson", [
    #         {"id": i, "start": norm_segments[i][0], "end": norm_segments[i][1]} for i in range(len(norm_segments))], section_area=None, utm_epsg="32630")
    ecs.write_geojson_sections(args.output_data_dir / f"{args.estuary_name}_sections.geojson", [
            {"id": i, "start": norm_segments[i][0], "end": norm_segments[i][1]} for i in range(len(norm_segments))], section_area=None, utm_epsg=args.utm_epsg)

    # load coastline shapefile and smooth it for easier area calculations
    # ....... TO BE CONTINUED .......
    print("This section is not fully working but I used the outputs to manually create the tamar_500m_sections.geojson file for use in the 2D model setup.") 
else:
    print(f"Loading sections from GEOJSON file to extract mesh nodes: '{args.section_file}'")


    section_file = args.section_file
    # read coastline geojson
    # coastline_shapefile = '../shapefile/ordered_coastline.geojson'
    coastline_shapefile = args.coastline_shapefile
    os.makedirs(args.output_fig_dir, exist_ok=True)
    # read section file as geopandas dataframe
    section = gpd.read_file(section_file)
    # reorder according to id (should be in order but just to be sure)
    section = section.sort_values("id").reset_index(drop=True)
    # read coastline shapefile as geopandas dataframe
    coastline = gpd.read_file(coastline_shapefile)

    sea_polygon, closed_lines = mga.clean_coastline(coastline)

    gpd.GeoDataFrame(geometry=[sea_polygon], crs=coastline.crs).to_file(os.path.join(args.output_data_dir, args.estuary_name+ "sea_polygon.geojson"), driver="GeoJSON")

    # build polygons from section start and end points
    cells = []
    for i in range(len(section) - 1):
        p0 = section.geometry[i].coords[0]
        p1 = section.geometry[i].coords[-1]
        p2 = section.geometry[i+1].coords[-1]
        p3 = section.geometry[i+1].coords[0]
        poly = Polygon([p0, p1, p2, p3])
        if poly.is_valid:
            cells.append({"id": i, "geometry": poly})
        else:
            print(f"Warning: Cell {i} is invalid. Attempting to fix with make_valid.")
            poly_fixed = make_valid(poly)
            if poly_fixed.is_valid:
                print(f"Cell {i} was successfully fixed.")
                cells.append({"id": i, "geometry": poly_fixed})
            else:
                print(f"Error: Cell {i} is still invalid after attempting to fix. Skipping this cell.")
        
        # cells.append({"id": i, "geometry": poly})
    # convert to geopandas dataframe
    cells_gdf = gpd.GeoDataFrame(cells, crs=section.crs)

    if args.do_plots:
        fig, ax = plt.subplots(figsize=(10, 10))
        coastline.plot(ax=ax, color='grey', linewidth=0.5)
        section.to_crs(coastline.crs).plot(ax=ax, color='red', linewidth=1)
        plt.title('Sections and Coastline')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(args.output_fig_dir, args.estuary_name+'sections_and_coastline.png'), dpi=100)

    # --- reproject to UTM for area in m² ---
    utm_crs = section.crs.to_epsg()  # or hardcode e.g. "EPSG:32630"
    cells_proj = cells_gdf.to_crs(f"EPSG:{utm_crs}")

    sea_polygon_proj = gpd.GeoSeries([sea_polygon], crs=coastline.crs).to_crs(f"EPSG:{utm_crs}")[0]
    # plot polygones to check they look right

    # intersect cells with sea polygon coastline
    cells_proj["geometry"] = cells_proj.geometry.apply(
        lambda g: make_valid(g).intersection(sea_polygon_proj)
    )
    # extract area of each cell in m² and filter out cells with zero area
    cells_proj["wet_area_m2"] = cells_proj.geometry.area
    cells_proj = cells_proj[cells_proj["wet_area_m2"] > 0]

    cells_proj.to_crs("EPSG:4326").to_file(args.output_data_dir / (args.estuary_name+"_cells_masked.geojson"), driver="GeoJSON")

    section_proj = section.to_crs(cells_proj.crs)
    # get intersection of section lines with sea polygon boundary to get the intersection points
    section_proj["intersection"] = section_proj.geometry.apply(
        lambda g: make_valid(g).intersection(sea_polygon_proj.boundary)
    )
    # extract the intersection points and save to a geojson file
    points_list = []
    for idx, row in section_proj.iterrows():
        for pt in mga.extract_points(row["intersection"], row.geometry):
            points_list.append({"section_id": idx, "geometry": pt})
    # save nodes id, lat and lon and utm projected x and y to csv
    points_gdf = gpd.GeoDataFrame(points_list, crs=section_proj.crs)
    points_gdf_sph = points_gdf.to_crs("EPSG:4326")
    points_gdf["lon"] = points_gdf_sph.geometry.x
    points_gdf["lat"] = points_gdf_sph.geometry.y
    points_gdf[["section_id", "lon", "lat"]].to_csv(args.output_data_dir / (args.estuary_name+"_intersection_nodes.csv"), index=False)
    # now in projected coordinates for use in mesh generation as csv
    section_id, x, y = points_gdf["section_id"], points_gdf.geometry.x, points_gdf.geometry.y
    points_gdf["x"] = x
    points_gdf["y"] = y
    points_gdf[["section_id", "x", "y"]].to_csv(args.output_data_dir / (args.estuary_name+"_intersection_nodes_proj.csv"), index=False)

    points_gdf.to_crs("EPSG:4326").to_file(args.output_data_dir / (args.estuary_name+"_intersection_nodes.geojson"), driver="GeoJSON")

    # limit the section lines to the wet geometry by intersecting with the sea polygon
    section_proj["wet_geometry"] = section_proj.geometry.apply(
        lambda g: make_valid(g).intersection(sea_polygon_proj)
    )
    section_proj["wet_length_m"] = section_proj["wet_geometry"].length
    section_proj['wet_geometry'].to_crs("EPSG:4326").to_file(args.output_data_dir / (args.estuary_name+"_sections_masked.geojson"), driver="GeoJSON")

    # build wet polygons between adjacent sections
    cells, nodes_list = mga.build_wet_polygons(section_proj)

    # save nodes_list to csv for use in mesh generation
    # this has the correct order for the nodes to be used in mesh generation, with the section id and the node coordinates in lon/lat and projected x/y
    nodes_df = pd.DataFrame(nodes_list)
    nodes_df.to_csv(args.output_data_dir / (args.estuary_name+"_nodes_dict_manual.csv"), index=False)
    if DEFAULTS['check_polygons_plot']:
        mga.plot_polygons_stepwise(gpd.GeoDataFrame(cells, crs=section_proj.crs), coastline_gdf=coastline.to_crs(section_proj.crs), id_col="id")
    wet_cells_gdf = gpd.GeoDataFrame(cells, crs=section_proj.crs)
    wet_cells_gdf["wet_area_m2"] = wet_cells_gdf.geometry.area
    wet_cells_gdf = wet_cells_gdf[wet_cells_gdf["wet_area_m2"] > 0]
    wet_cells_gdf.to_crs("EPSG:4326").to_file(args.output_data_dir / (args.estuary_name+"_wet_cells.geojson"), driver="GeoJSON")
    # Use the FVCOM mesh to estimate the mean depth for each wet cell by intersecting the FVCOM elements with the wet cells and calculating the area-weighted mean depth for each cell. 
    # This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
    # build triangle polygons from node coordinates
    wet_cells_gdf, fvcom_proj, min_depth, all_pieces = mga.get_fvcom_elements_depths(mesh=mesh, wet_cells_gdf=wet_cells_gdf, cells_proj=cells_proj)
    # save element pieces for plotting/QGIS
    pieces_gdf = gpd.GeoDataFrame(all_pieces, crs=fvcom_proj.crs)
    # remove depth offset from depths
    pieces_gdf["depth"] = pieces_gdf["depth"] + min_depth - 1
    pieces_gdf.to_crs("EPSG:4326").to_file(args.output_data_dir / (args.estuary_name+"_intersection_elements.geojson"), driver="GeoJSON")
    # plot_polygons_stepwise(gpd.GeoDataFrame(cells, crs=section_proj.crs), coastline_gdf=coastline.to_crs(section_proj.crs), id_col="id", mesh = pieces_gdf)
    fig, ax = plt.subplots(figsize=(12, 12))

    wet_cells_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, label='wet cells')
    pieces_gdf.plot(ax=ax, column='depth', cmap='viridis', edgecolor='black', linewidth=0.2, legend=True)

    plt.title('FVCOM elements intersecting wet cells (coloured by depth)')
    fig.savefig(os.path.join(args.output_fig_dir, args.estuary_name+"_fvcom_elements_intersecting_wet_cells.png"), dpi=300)
    # limite the axis to the top northern section by setting the limit to the northern 10% of the y range.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min + 0.9 * y_range, y_max)
    fig.savefig(os.path.join(args.output_fig_dir, args.estuary_name+"_fvcom_elements_intersecting_wet_cells_zoom.png"), dpi=300)

    # estimate a "real" depth by dividing the volume by the real intersection area (which is the area of the cell intersected with the sea polygon, i.e. the actual wet area). This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
    wet_cells_gdf["real_mean_depth_m"] = wet_cells_gdf["element_volume_m3"] / wet_cells_gdf["element_area_m2"]
    # remove depth offset from real_mean_depth_m
    wet_cells_gdf["real_mean_depth_m"] = wet_cells_gdf["real_mean_depth_m"] + min_depth - 1
    wet_cells_gdf.to_crs("EPSG:4326").to_file(args.output_data_dir / (args.estuary_name+"_wet_cells_with_depths.geojson"), driver="GeoJSON")
    # plot figures for along_axis distance vs real_area and real_mean_depth and also mean_depth and intersection_area for sanity check
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Section ID')
    ax1.set_ylabel('Real Intersection Area (m²)', color=color)
    ax1.plot(wet_cells_gdf["id"], wet_cells_gdf["real_intersection_area_m2"], color=color, marker='o', label='Real Intersection Area')
    ax1.plot(wet_cells_gdf["id"], wet_cells_gdf["element_area_m2"], color='tab:cyan', marker='x', label='Calculated Intersection Area')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    # make y axis log scale to better show the smaller areas
    ax1.set_yscale('log')
    # make a secondary axis with the ratios
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Area Ratio (Calculated/Real)', color=color)
    area_ratio = wet_cells_gdf["element_area_m2"] / wet_cells_gdf["real_intersection_area_m2"]
    ax2.plot(wet_cells_gdf["id"], area_ratio, color=color, marker='d', label='Area Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    # save the figure
    fig.savefig(os.path.join(args.output_fig_dir, args.estuary_name+'_intersection_area_comparison.png'), dpi=300)
    fig, ax2 = plt.subplots(figsize=(10, 6))
    color = 'tab:green'
    ax2.set_xlabel('Section ID')
    ax2.set_ylabel('Mean Depth (m)', color=color)
    ax2.plot(wet_cells_gdf["id"], wet_cells_gdf["q75_depth_m"], color=color, marker='o', label='Real Q75 Depth')
    ax2.plot(wet_cells_gdf["id"], wet_cells_gdf["real_mean_depth_m"], color='tab:olive', marker='s', label='Estimated Real Mean Depth')
    ax2.plot(wet_cells_gdf["id"], wet_cells_gdf["mean_depth_m"], color='tab:olive', marker='x', label='Calculated Mean Depth')
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.legend(loc='upper left')
    # save the figure
    fig.savefig(os.path.join(args.output_fig_dir, args.estuary_name+'_mean_depth_comparison.png'), dpi=300)
    # also mean_volume and real_mean_volume
    fig, ax3 = plt.subplots(figsize=(10, 6))
    color = 'tab:purple'
    ax3.set_xlabel('Section ID')
    ax3.set_ylabel('Volume (m³)', color=color)
    ax3.plot(wet_cells_gdf["id"], wet_cells_gdf["element_volume_m3"], color=color, marker='o', label='Element Volume')
    ax3.plot(wet_cells_gdf["id"], wet_cells_gdf["mean_volume_m3"], color='tab:pink', marker='x', label='Real Mean Volume')
    # make y axis log scale to better show the smaller areas
    ax3.set_yscale('log')
    ax3.tick_params(axis='y', labelcolor=color)
    # add a secondary axis with the ratio of the volumes
    ax4 = ax3.twinx()
    color = 'tab:orange'
    ax4.set_ylabel('Volume Ratio (Element/Real)', color=color)
    volume_ratio = wet_cells_gdf["element_volume_m3"] / wet_cells_gdf["mean_volume_m3"]   
    ax4.plot(wet_cells_gdf["id"], volume_ratio, color=color, marker='d', label='Volume Ratio')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.legend(loc='upper right')

    # save the figure
    fig.savefig(os.path.join(args.output_fig_dir, args.estuary_name+'_volume_comparison.png'), dpi=300)
    # save a dictionary of list of nodes and their corresponding section id and depths for use in mesh generation
    nodes_dict = {}
    depth_dict = {}
    for idx, row in points_gdf.iterrows():
        section_id = row["section_id"]
        nodes_dict[idx] = {"section_id": section_id, "lon": row["lon"], "lat": row["lat"], "x": row["x"], "y": row["y"]}
        try:
            depth_dict[section_id] = {"section_id": section_id, "real_mean_depth_m": wet_cells_gdf.loc[wet_cells_gdf["id"] == section_id, "q75_depth_m"].values[0], "element_area_m2": wet_cells_gdf.loc[wet_cells_gdf["id"] == section_id, "element_area_m2"].values[0]}
        except IndexError:
            print(f"Warning: No depth found for section_id {section_id}")   
    # save nodes_dict and depth_dict as csv for use in mesh generation
    import csv
    with open(os.path.join(args.output_data_dir, args.estuary_name+ "_nodes_dict.csv"), "w", newline="") as csvfile:
        fieldnames = ["node_id", "section_id", "lon", "lat", "x", "y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node_id, data in nodes_dict.items():
            writer.writerow({"node_id": node_id, **data})
    with open(os.path.join(args.output_data_dir, args.estuary_name+ "_depth_dict.csv"), "w", newline="") as csvfile:
        fieldnames = ["section_id", "real_mean_depth_m", "element_area_m2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for section_id, depth in depth_dict.items():
            writer.writerow({"section_id": section_id, **depth})
# convert start and end date to datetime objects for use in CMEMS data download
start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

# download T and S cmems data for the same period
cmems.download_cmems_data(
    bbox=[args.bbox[0], args.bbox[2], args.bbox[1], args.bbox[3]],
    start_date=args.start_date,
    end_date=args.end_date,
)

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
   output_dir=args.output_data_dir
)

# download meteorological data for the same period for potential later use in forcing the model
import runpy
import sys
for year in range(int(start_date.year), int(end_date.year) + 1):
    sys.argv = ["igotm",  str(bdy_lon),  str(bdy_lat), str(year), os.path.join(args.output_data_dir, f"era5_{year}.nc")]
    runpy.run_module("pygetm.input.igotm", run_name="__main__")
    # alternatively, you can call the function directly


# save the centerline, distance along channel and depths to a pickle for later use
# python plot_transect.py --mesh ../tamar_v0/tamar_v2_grd.dat  --transect-pickle ../transect_grid/tamar_v2_transect.pk --channel-key "channel_nodes"  --out-png ../tamar_axial_50m.png