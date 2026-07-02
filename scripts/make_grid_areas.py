# Script to read a coastline and some sections and calculate the surface area of the intersection for each adjacent section

import geopandas as gpd
from shapely.ops import unary_union
import pandas as pd
import pyproj
import numpy as np
import os
import matplotlib

# Prefer a GUI backend for keypress interaction; switch away from non-interactive Agg/inline.
_requested_backend = os.environ.get("MPLBACKEND", "").strip().lower()
_needs_interactive_backend = _requested_backend in {"", "agg", "module://matplotlib_inline.backend_inline", "inline"}

if _needs_interactive_backend:
    for _backend in ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(_backend, force=True)
            break
        except Exception:
            continue

print(f"Matplotlib backend: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.validation import explain_validity, make_valid

from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import Polygon, LinearRing, box
from shapely.validation import make_valid, explain_validity
from shapely.geometry import box
# order linestrings with nearest neighbour to try to get a single linestring. This is needed to get the area of the intersection with the coastline
from shapely.ops import linemerge, unary_union, snap
from shapely.geometry import LineString, MultiLineString
import numpy as np
import estimate_channel_section_volumes as ecs
from shapely.geometry import Point, MultiPoint, GeometryCollection

def get_endpoints(line):
    coords = list(line.coords)
    return coords[0], coords[-1]

def point_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def extract_points(geom, line):
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, MultiPoint):
        pts = list(geom.geoms)
        pts.sort(key=lambda p: line.project(p))
        return pts
    if isinstance(geom, GeometryCollection):
        pts = []
        for g in geom.geoms:
            pts.extend(extract_points(g, line))
        return pts
    return []
def area_weighted_depth(cell_geom, fvcom_gdf):
    overlaps = fvcom_gdf[fvcom_gdf.intersects(cell_geom)]
    if overlaps.empty:
        return np.nan
    
    weighted_sum = 0.0
    total_area = 0.0
    for _, elem in overlaps.iterrows():
        intersection = make_valid(elem.geometry).intersection(cell_geom)
        area = intersection.area
        if area > 0:
            weighted_sum += elem["depth"] * area
            total_area += area
    
    return weighted_sum / total_area if total_area > 0 else np.nan


def cell_volume_and_elements(cell_geom, fvcom_gdf):
    """Returns total volume and the list of intersecting element pieces for plotting."""
    overlaps = fvcom_gdf[fvcom_gdf.intersects(cell_geom)]
    
    pieces = []
    volume = 0.0
    
    for _, elem in overlaps.iterrows():
        intersection = make_valid(elem.geometry).intersection(cell_geom)
        if intersection.is_empty:
            continue
        area = intersection.area
        depth = max(elem["depth"], 0)  # clip negative (land) depths
        volume += area * depth
        pieces.append({"geometry": intersection, "depth": depth, "area_m2": area, "element_area_m2": elem["area_m2"], "coast": elem["coast"]})
    
    return volume, pieces

def order_quad_clockwise(points):
    """
    points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    pts = np.asarray(points)

    centroid = pts.mean(axis=0)

    angles = np.arctan2(
        pts[:,1] - centroid[1],
        pts[:,0] - centroid[0]
    )

    order = np.argsort(angles)[::-1]  # clockwise

    return pts[order].tolist()


def plot_polygons_stepwise(polygons_gdf, coastline_gdf=None, id_col="id", mesh=None):
    """Plot polygons one by one, label nodes in coordinate order, and wait for key press."""
    if polygons_gdf.empty:
        print("No polygons available for interactive plotting.")
        return

    non_interactive_backends = {
        "agg",
        "pdf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
        "inline",
    }

    backend_name = matplotlib.get_backend().lower()
    if backend_name in non_interactive_backends:
        # Some imported dependencies may switch backend to Agg at runtime.
        for _backend in ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg"):
            try:
                plt.switch_backend(_backend)
                break
            except Exception:
                continue
        backend_name = matplotlib.get_backend().lower()

    print(f"Interactive viewer backend: {matplotlib.get_backend()}")
    if backend_name in non_interactive_backends:
        print(
            "Interactive polygon viewer skipped: non-interactive backend "
            f"'{matplotlib.get_backend()}'."
        )
        return

    # Work in a projected CRS so the 500 m zoom margin is in meters.
    polygons_plot = polygons_gdf
    coastline_plot = coastline_gdf
    if polygons_gdf.crs is not None and polygons_gdf.crs.is_geographic:
        metric_crs = polygons_gdf.estimate_utm_crs()
        polygons_plot = polygons_gdf.to_crs(metric_crs)
        if coastline_gdf is not None and not coastline_gdf.empty:
            coastline_plot = coastline_gdf.to_crs(metric_crs)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()
    plt.show(block=False)

    for idx, row in polygons_plot.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Use largest part when geometry is multipolygon.
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)

        if geom.geom_type != "Polygon":
            continue

        ax.clear()
        if coastline_plot is not None and not coastline_plot.empty:
            coastline_plot.plot(
                ax=ax,
                color="lightgrey",
                linewidth=0.7,
                alpha=0.8,
            )
        gpd.GeoSeries([geom], crs=polygons_plot.crs).plot(
            ax=ax,
            facecolor="lightblue",
            edgecolor="black",
            alpha=0.5,
        )
        if mesh is not None:
            # plot each geometry in the mesh as a red outline with no facecolor and a thin linewidth. This is needed to check the mesh elements are correctly aligned with the polygons and coastline
            all_triangles = mesh[mesh['cell_id'] == row.id]
            all_triangles.plot(
                    ax=ax,
                    # color="none",
                    column='depth',
                    cmap='viridis',
                    edgecolor="red",
                    linewidth=0.5,
                )
            # plot the non coastal nodes as red points with a small marker size and label them with their node id. This is needed to check the mesh nodes are correctly aligned with the polygons and coastline
            for node_num, point in enumerate(all_triangles['coast'].values[0], start=1):
                ax.plot(point.x, point.y, marker="o", color="red", markersize=4)
                # ax.text(point.x, point.y, str(node_num), color="black", fontsize=9, ha="left", va="bottom")
            # mesh.plot(ax=ax, color="none", edgecolor="red", linewidth=0.5)

        # Zoom to local neighborhood: polygon extent plus 500 m margin.
        minx, miny, maxx, maxy = geom.bounds
        margin_m = 500.0
        ax.set_xlim(minx - margin_m, maxx + margin_m)
        ax.set_ylim(miny - margin_m, maxy + margin_m)

        coords = list(geom.exterior.coords)[:-1]
        for node_num, (x, y) in enumerate(coords, start=1):
            ax.plot(x, y, marker="o", color="red", markersize=4)
            ax.text(x, y, str(node_num), color="black", fontsize=9, ha="left", va="bottom")

        poly_id = row[id_col] if id_col in row else idx
        ax.set_title(f"Polygon {poly_id}: press any key for next")
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.waitforbuttonpress()

    plt.close(fig)
def clean_coastline(coastline, ):
        # clean emtpy geometries from coastline
    coastline = coastline[~coastline.is_empty]
    coast_lines = coastline[coastline.geom_type.isin(["LineString", "MultiLineString"])]
    coast_polys = coastline[coastline.geom_type.isin(["Polygon", "MultiPolygon"])]
    closed_lines = []   # islands — already closed rings
    open_lines = []     # coastline segments — need ordering and merging

    for geom in coast_lines.geometry:
        parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
        for part in parts:
            coords = list(part.coords)
            if coords[0] == coords[-1]:
                closed_lines.append(part)
            else:
                open_lines.append(part)

    print(f"Coastline segments: {len(open_lines)}")
    print(f"Island rings: {len(closed_lines)}")
    # reorder and merge open coastline segments as before
    remaining = list(range(len(open_lines)))
    ordered = [remaining.pop(0)]
    current_end = get_endpoints(open_lines[ordered[-1]])[1]

    while remaining:
        best_idx, best_dist, best_flip = None, np.inf, False
        for idx in remaining:
            start, end = get_endpoints(open_lines[idx])
            d_start = point_dist(current_end, start)
            d_end   = point_dist(current_end, end)
            if d_start < best_dist:
                best_dist, best_idx, best_flip = d_start, idx, False
            if d_end < best_dist:
                best_dist, best_idx, best_flip = d_end, idx, True
        remaining.remove(best_idx)
        if best_flip:
            open_lines[best_idx] = LineString(list(open_lines[best_idx].coords)[::-1])
        ordered.append(best_idx)
        current_end = get_endpoints(open_lines[best_idx])[1]

    ordered_lines = [open_lines[i] for i in ordered]
    all_coords = []
    for line in ordered_lines:
        coords = list(line.coords)
        if all_coords:
            # skip the first coord of each segment (duplicate of previous end)
            all_coords.extend(coords[1:])
        else:
            all_coords.extend(coords)

    # force close
    if all_coords[0] != all_coords[-1]:
        all_coords.append(all_coords[0])

    sea_polygon = make_valid(Polygon(all_coords))
    print(sea_polygon.geom_type)  # could become MultiPolygon if self-intersections exist

    if sea_polygon.geom_type == 'MultiPolygon':
        sea_polygon = max(sea_polygon.geoms, key=lambda p: p.area)
    return sea_polygon, closed_lines
def build_wet_polygons(section_proj):
        # make polygons between adjacent sections for the wet part of the section only. This is needed to get the area of the intersection with the coastline
    cells = []
    # extract line 1 nodes and assign them a node id and section id and save to csv for use in mesh generation
    # this assumes only two nodes per section, which should be the case for the way the sections are currently defined. If there are more than two nodes per section, this will need to be modified to assign node ids in the correct order along the section. This is needed to get the area of the intersection with the coastline
    nodes_list = {'node_id': [], 'section_id': [], 'x': [], 'y': [], 'lon': [], 'lat': []}
    j = 0
    for i in range(len(section_proj) - 1):
        line1 = section_proj["wet_geometry"].iloc[i]
        line2 = section_proj["wet_geometry"].iloc[i+1]
        if line1.is_empty or line2.is_empty:
            continue
        p0 = line1.coords[0]
        p1 = line1.coords[-1]
        p2 = line2.coords[-1]
        p3 = line2.coords[0]
        poly = Polygon([p0, p1, p2, p3])
        # check the polygon is valid and not self-intersecting. If it is invalid, try to fix it with make_valid and check again. If it's still invalid, skip this cell and print a warning.
        if poly.is_valid:
            cells.append({"id": i, "geometry": poly})
            nodes_list['node_id'].append(j)
            nodes_list['section_id'].append(i)
            nodes_list['x'].append(p1[0])
            nodes_list['y'].append(p1[1])
            # convert points to lat and lon
            lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p1[0], p1[1])
            nodes_list['lon'].append(lon)
            nodes_list['lat'].append(lat)
            j+=1
            nodes_list['section_id'].append(i)
            nodes_list['node_id'].append(j)
            nodes_list['x'].append(p0[0])
            nodes_list['y'].append(p0[1])
            lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p0[0], p0[1])
            nodes_list['lon'].append(lon)
            nodes_list['lat'].append(lat)
            j+=1

        else:
            print(f"Warning: Cell {i} is invalid. Attempting to fix with make_valid.")
            poly_fixed = make_valid(poly)
            if poly_fixed.is_valid:
                print(f"Cell {i} was successfully fixed.")
                cells.append({"id": i, "geometry": poly_fixed})
                nodes_list['node_id'].append(j)
                nodes_list['section_id'].append(i)
                nodes_list['x'].append(p1[0])
                nodes_list['y'].append(p1[1])
                lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p1[0], p1[1])
                nodes_list['lon'].append(lon)
                nodes_list['lat'].append(lat)
                j+=1
                nodes_list['node_id'].append(j)
                nodes_list['section_id'].append(i)
                nodes_list['x'].append(p0[0])
                nodes_list['y'].append(p0[1])
                lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p0[0], p0[1])
                nodes_list['lon'].append(lon)
                nodes_list['lat'].append(lat)
                j+=1
            else:
                print(f"Error: Cell {i} is still invalid after attempting to fix. Skipping this cell.")
    return cells, nodes_list

def get_fvcom_elements_depths(mesh, wet_cells_gdf, cells_proj, quantile=0.85):
    """
    mesh: xarray Dataset with FVCOM mesh data (x, y, triangles, depth)
    wet_cells_gdf: GeoDataFrame with wet cells polygons
    cells_proj: GeoDataFrame with projected cells polygons
    quantile: float, quantile to calculate for depth
    Returns: wet_cells_gdf with additional columns for mean depth and volume based on FVCOM mesh elements
    """
    triangle_polys = []
    triangle_depths = []
    triangle_nodes = []
    triangle_coast = []
    triangle_area = []
    mesh_coastline, coastal_nodes = ecs.boundary_polygon_from_triangles(mesh.x, mesh.y, mesh.triangles)
    for tri in mesh.triangles:
        coords = [(mesh.x[n], mesh.y[n]) for n in tri]
        poly = Polygon(coords)
        # check if any of the triangle nodes are on the coastline and exclude from the depth calculation if so, as these will be land nodes with negative depths that will skew the depth calculation for the intersecting cells. This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
        # extract the nodes that are not on the coastline and calculate the mean depth from those
        non_coast_nodes = [n for n in tri if n not in coastal_nodes]
        # get the coastal ones as well for plotting
        coastal_nodes_in_tri = [n for n in tri if n in coastal_nodes]
        depth = mesh.depth[non_coast_nodes].mean() # elemen mean only considers non-coastal nodes to avoid skewing the depth calculation with land nodes that have negative depths. This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
        triangle_coast.append([Point(mesh.x[n], mesh.y[n]) for n in coastal_nodes_in_tri ])
        triangle_nodes.append(non_coast_nodes)
        triangle_area.append(poly.area)
        triangle_polys.append(poly)
        triangle_depths.append(depth)

    fvcom_elements = gpd.GeoDataFrame(
        {"depth": triangle_depths, "non_coast_nodes": triangle_nodes, "coast": triangle_coast, "nodes": mesh.triangles.tolist(), "area_m2": triangle_area},
        geometry=triangle_polys,
        crs="EPSG:32630"  # adjust to your projected CRS — looks like UTM 30N given the coords
    )
    # calculate the minimum depth overall and add a temporary offset to ensure all depths are positive for volume calculation. This is needed to get the area of the intersection with the coastline
    min_depth = fvcom_elements["depth"].min()
    if min_depth < 0:
        fvcom_elements["depth"] = fvcom_elements["depth"] - min_depth + 1  # add offset to make all depths positive
    fvcom_proj = fvcom_elements.to_crs(wet_cells_gdf.crs)

    wet_cells_gdf["mean_depth_m"] = wet_cells_gdf.geometry.apply(
        lambda g: area_weighted_depth(g, fvcom_proj)
    )
    wet_cells_gdf["mean_volume_m3"] = wet_cells_gdf["wet_area_m2"] * wet_cells_gdf["mean_depth_m"]
    # correct the depth offset by subtracting the same offset from the mean depth and mean volume
    wet_cells_gdf["mean_depth_m"] = wet_cells_gdf["mean_depth_m"] + min_depth - 1

    # apply to all wet cells
    results = []
    all_pieces = []

    for idx, row in cells_proj.iterrows():
        volume, pieces = cell_volume_and_elements(row.geometry, fvcom_proj)
        results.append(volume)
        for p in pieces:
            p["cell_id"] = row["id"]
            all_pieces.append(p)

    wet_cells_gdf["element_volume_m3"] = results
    # extract area from all_pieces and save it to wet_cells_gdf for plotting/QGIS
    area_dict = {}
    intersect_polys = {}
    depth_dict = {}
    for piece in all_pieces:
        cell_id = piece["cell_id"]
        area = piece["area_m2"]
        intersect_polys[cell_id] = piece["geometry"]
        if cell_id in area_dict:
            area_dict[cell_id] += area
            depth_dict[cell_id].append(piece["depth"])
        else:
            area_dict[cell_id] = area
            depth_dict[cell_id] = [piece["depth"]]
    wet_cells_gdf["element_area_m2"] = wet_cells_gdf["id"].map(area_dict)
    wet_cells_gdf["q_depth_m"] = wet_cells_gdf["id"].map(lambda x: np.quantile(np.array(depth_dict[x]), quantile) if x in depth_dict else np.nan) + min_depth - 1
    wet_cells_gdf["intersection_geometry"] = wet_cells_gdf["id"].map(intersect_polys)
    # add the area calculated from the intersection with the coastline (i.e. the real wet area)
    wet_cells_gdf["real_intersection_area_m2"] = cells_proj["wet_area_m2"]
    return wet_cells_gdf, fvcom_proj, min_depth, all_pieces



if __name__ == "__main__":
    experiment='0_depth_'
    # sections file as geojson
    section_file = '../transect_grid/tamar_500m_sections.geojson'
    # read coastline geojson
    # coastline_shapefile = '../shapefile/ordered_coastline.geojson'
    coastline_shapefile = '../shapefile/0_depth_limited_coastline_singleline.geojsonl'
    quantile=0.95
    mes_file_path="../tamar_grid/tamar_v2_grd.dat"
    # plot the sections and coastlines to check they look right
    plot_sections_and_coastline = True
    plot_polygons_interactive = False
    process_sections = False
    fig_dir = '../figures/section_areas'
    os.makedirs(fig_dir, exist_ok=True)
    # read section file as geopandas dataframe
    section = gpd.read_file(section_file)
    # reorder according to id (should be in order but just to be sure)
    section = section.sort_values("id").reset_index(drop=True)
    # read coastline shapefile as geopandas dataframe
    coastline = gpd.read_file(coastline_shapefile)
    # clean emtpy geometries from coastline
    coastline = coastline[~coastline.is_empty]

    coast_lines = coastline[coastline.geom_type.isin(["LineString", "MultiLineString"])]
    coast_polys = coastline[coastline.geom_type.isin(["Polygon", "MultiPolygon"])]
    closed_lines = []   # islands — already closed rings
    open_lines = []     # coastline segments — need ordering and merging

    for geom in coast_lines.geometry:
        parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
        for part in parts:
            coords = list(part.coords)
            if coords[0] == coords[-1]:
                closed_lines.append(part)
            else:
                open_lines.append(part)

    print(f"Coastline segments: {len(open_lines)}")
    print(f"Island rings: {len(closed_lines)}")
    # reorder and merge open coastline segments as before
    remaining = list(range(len(open_lines)))
    ordered = [remaining.pop(0)]
    current_end = get_endpoints(open_lines[ordered[-1]])[1]

    while remaining:
        best_idx, best_dist, best_flip = None, np.inf, False
        for idx in remaining:
            start, end = get_endpoints(open_lines[idx])
            d_start = point_dist(current_end, start)
            d_end   = point_dist(current_end, end)
            if d_start < best_dist:
                best_dist, best_idx, best_flip = d_start, idx, False
            if d_end < best_dist:
                best_dist, best_idx, best_flip = d_end, idx, True
        remaining.remove(best_idx)
        if best_flip:
            open_lines[best_idx] = LineString(list(open_lines[best_idx].coords)[::-1])
        ordered.append(best_idx)
        current_end = get_endpoints(open_lines[best_idx])[1]

    ordered_lines = [open_lines[i] for i in ordered]
    # plot ordered lines to check they look right
    if plot_sections_and_coastline:
        fig, ax = plt.subplots(figsize=(10, 10))
        coastline.plot(ax=ax, color='lightgrey')
        gpd.GeoDataFrame(geometry=ordered_lines, crs=coastline.crs).plot(ax=ax, color='blue', linewidth=0.5)
        plt.title('Ordered Coastline Segments')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(fig_dir, experiment+'ordered_coastline_segments.png'), dpi=300)


    all_coords = []
    for line in ordered_lines:
        coords = list(line.coords)
        if all_coords:
            # skip the first coord of each segment (duplicate of previous end)
            all_coords.extend(coords[1:])
        else:
            all_coords.extend(coords)

    # force close
    if all_coords[0] != all_coords[-1]:
        all_coords.append(all_coords[0])

    sea_polygon = make_valid(Polygon(all_coords))
    print(sea_polygon.geom_type)  # could become MultiPolygon if self-intersections exist

    if sea_polygon.geom_type == 'MultiPolygon':
        sea_polygon = max(sea_polygon.geoms, key=lambda p: p.area)


    gpd.GeoDataFrame(geometry=[sea_polygon], crs=coastline.crs).to_file(os.path.join(fig_dir, experiment+ "sea_polygon.geojson"), driver="GeoJSON")

    # punch islands out
    island_polygons = [Polygon(ring.coords) for ring in closed_lines]
    islands = make_valid(unary_union(island_polygons))
    sea_masked = make_valid(sea_polygon.difference(islands))
    # order sections from upstream to downstream based on the closest endpoints of the sections. This is needed to get the area of the intersection with the coastline
    if process_sections:
        # --- build cell polygons between consecutive sections ---
        section = section[~section.is_empty]
        # remove any empty geometries from section and reset index to ensure clean integer indexing for loop below
        for i, geom in enumerate(section.geometry):
            # check if geom is empty or NoneType and drop it if so
            if geom is None or geom.is_empty:
                print(f"Warning: Section {i} has empty geometry and will be skipped.")
                section = section.drop(i)
        section = section.reset_index(drop=True)  # ensure clean integer index
        # ordered sections too so adjacent ones are next to each other in the dataframe. This is needed to get the area of the intersection with the coastline
        # do this by finding the closest section to the previous one and ordering by that. Start with the first section and then loop through the rest
        remaining = list(range(len(section)))
        ordered = [remaining.pop(0)]
        current_end = get_endpoints(section.geometry[ordered[-1]])[1]   
        while remaining:
            best_idx, best_dist, best_flip = None, np.inf, False
            for idx in remaining:
                start, end = get_endpoints(section.geometry[idx])
                d_start = point_dist(current_end, start)
                d_end   = point_dist(current_end, end)
                if d_start < best_dist:
                    best_dist, best_idx, best_flip = d_start, idx, False
                if d_end < best_dist:
                    best_dist, best_idx, best_flip = d_end, idx, True
            remaining.remove(best_idx)
            if best_flip:
                section.geometry[best_idx] = LineString(list(section.geometry[best_idx].coords)[::-1])
            ordered.append(best_idx)
            current_end = get_endpoints(section.geometry[best_idx])[1]
        # re-order section dataframe by ordered list of indices
        ordered_section = section.iloc[ordered].reset_index(drop=True)
        # ordered_section = [section[i] for i in ordered]

        section = ordered_section
        # copy index to Id column 
        section["id"] = section.index
        # save ordered sections to new geojson for sanity check
        section.to_file("ordered_sections.geojson", driver="GeoJSON")
    # remove any empty geometries from section and build polygons between consecutive sections. This is needed to get the area of the intersection with the coastline
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

    cells_gdf = gpd.GeoDataFrame(cells, crs=section.crs)
    # if plot_polygons_interactive:
    #     plot_polygons_stepwise(cells_gdf, coastline_gdf=coastline.to_crs(cells_gdf.crs), id_col="id")

    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoDataFrame(geometry=[sea_masked], crs=coastline.crs).plot(ax=ax, color='lightblue', alpha=0.5)
    coastline.plot(ax=ax, color='grey', linewidth=0.5)
    section.to_crs(coastline.crs).plot(ax=ax, color='red', linewidth=1)
    plt.title('Sections and Coastline')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(fig_dir, experiment+'sections_and_coastline.png'), dpi=300)

    # --- reproject to UTM for area in m² ---
    utm_crs = section.crs.to_epsg()  # or hardcode e.g. "EPSG:32630"
    cells_proj = cells_gdf.to_crs(f"EPSG:{utm_crs}")

    # reproject to UTM for area in m²
    cells_proj = cells_gdf.to_crs("EPSG:32630")  # adjust UTM zone for your estuary
    sea_masked_proj = gpd.GeoSeries([sea_masked], crs=coastline.crs).to_crs("EPSG:32630")[0]
    # plot polygones to check they look right

    # intersect cells with sea polygon
    cells_proj["geometry"] = cells_proj.geometry.apply(
        lambda g: make_valid(g).intersection(sea_masked_proj)
    )
    cells_proj["wet_area_m2"] = cells_proj.geometry.area
    cells_proj = cells_proj[cells_proj["wet_area_m2"] > 0]

    cells_proj.to_crs("EPSG:4326").to_file("cells_masked.geojson", driver="GeoJSON")

    section_proj = section.to_crs(cells_proj.crs)

    section_proj["intersection"] = section_proj.geometry.apply(
        lambda g: make_valid(g).intersection(sea_masked_proj.boundary)
    )

    points_list = []
    for idx, row in section_proj.iterrows():
        for pt in extract_points(row["intersection"], row.geometry):
            points_list.append({"section_id": idx, "geometry": pt})

    points_gdf = gpd.GeoDataFrame(points_list, crs=section_proj.crs)
    points_gdf.to_crs("EPSG:4326").to_file(experiment+"intersection_nodes.geojson", driver="GeoJSON")

    # save nodes id, lat and lon and utm projected x and y to csv
    points_gdf_sph = points_gdf.to_crs("EPSG:4326")
    points_gdf["lon"] = points_gdf_sph.geometry.x
    points_gdf["lat"] = points_gdf_sph.geometry.y
    points_gdf[["section_id", "lon", "lat"]].to_csv(experiment+"intersection_nodes.csv", index=False)
    # now in projected coordinates for use in mesh generation as csv
    section_id, x, y = points_gdf["section_id"], points_gdf.geometry.x, points_gdf.geometry.y
    points_gdf["x"] = x
    points_gdf["y"] = y
    points_gdf[["section_id", "x", "y"]].to_csv(experiment+"intersection_nodes_proj.csv", index=False)

    section_proj["wet_geometry"] = section_proj.geometry.apply(
        lambda g: make_valid(g).intersection(sea_masked_proj)
    )
    section_proj["wet_length_m"] = section_proj["wet_geometry"].length

    # make polygons between adjacent sections for the wet part of the section only. This is needed to get the area of the intersection with the coastline
    cells = []
    # extract line 1 nodes and assign them a node id and section id and save to csv for use in mesh generation
    # this assumes only two nodes per section, which should be the case for the way the sections are currently defined. If there are more than two nodes per section, this will need to be modified to assign node ids in the correct order along the section. This is needed to get the area of the intersection with the coastline
    nodes_list = {'node_id': [], 'section_id': [], 'x': [], 'y': [], 'lon': [], 'lat': []}
    j = 0
    for i in range(len(section_proj) - 1):
        line1 = section_proj["wet_geometry"].iloc[i]
        line2 = section_proj["wet_geometry"].iloc[i+1]
        if line1.is_empty or line2.is_empty:
            continue
        p0 = line1.coords[0]
        p1 = line1.coords[-1]
        p2 = line2.coords[-1]
        p3 = line2.coords[0]
        poly = Polygon([p0, p1, p2, p3])
        # check the polygon is valid and not self-intersecting. If it is invalid, try to fix it with make_valid and check again. If it's still invalid, skip this cell and print a warning.
        if poly.is_valid:
            cells.append({"id": i, "geometry": poly})
            nodes_list['node_id'].append(j)
            nodes_list['section_id'].append(i)
            nodes_list['x'].append(p1[0])
            nodes_list['y'].append(p1[1])
            # convert points to lat and lon
            lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p1[0], p1[1])
            nodes_list['lon'].append(lon)
            nodes_list['lat'].append(lat)
            j+=1
            nodes_list['section_id'].append(i)
            nodes_list['node_id'].append(j)
            nodes_list['x'].append(p0[0])
            nodes_list['y'].append(p0[1])
            lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p0[0], p0[1])
            nodes_list['lon'].append(lon)
            nodes_list['lat'].append(lat)
            j+=1

        else:
            print(f"Warning: Cell {i} is invalid. Attempting to fix with make_valid.")
            poly_fixed = make_valid(poly)
            if poly_fixed.is_valid:
                print(f"Cell {i} was successfully fixed.")
                cells.append({"id": i, "geometry": poly_fixed})
                nodes_list['node_id'].append(j)
                nodes_list['section_id'].append(i)
                nodes_list['x'].append(p1[0])
                nodes_list['y'].append(p1[1])
                lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p1[0], p1[1])
                nodes_list['lon'].append(lon)
                nodes_list['lat'].append(lat)
                j+=1
                nodes_list['node_id'].append(j)
                nodes_list['section_id'].append(i)
                nodes_list['x'].append(p0[0])
                nodes_list['y'].append(p0[1])
                lon, lat = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:32630"), pyproj.CRS("EPSG:4326"), always_xy=True).transform(p0[0], p0[1])
                nodes_list['lon'].append(lon)
                nodes_list['lat'].append(lat)
                j+=1
            else:
                print(f"Error: Cell {i} is still invalid after attempting to fix. Skipping this cell.")
        # add line 1 nodes to nodes_list with node id and section id
        # cells.append({"id": i, "geometry": poly})
    # save nodes_list to csv for use in mesh generation
    nodes_df = pd.DataFrame(nodes_list)
    nodes_df.to_csv(experiment+"nodes_dict_manual.csv", index=False)
    if plot_polygons_interactive:
        plot_polygons_stepwise(gpd.GeoDataFrame(cells, crs=section_proj.crs), coastline_gdf=coastline.to_crs(section_proj.crs), id_col="id")
    wet_cells_gdf = gpd.GeoDataFrame(cells, crs=section_proj.crs)
    wet_cells_gdf["wet_area_m2"] = wet_cells_gdf.geometry.area
    wet_cells_gdf = wet_cells_gdf[wet_cells_gdf["wet_area_m2"] > 0]
    wet_cells_gdf.to_crs("EPSG:4326").to_file(experiment+"wet_cells.geojson", driver="GeoJSON")
    # read the mesh nodes from the mesh file and save the ones that are within the sea polygon to a new geojson for sanity check
    mesh = ecs.load_mesh(Path(mes_file_path))
    # build triangle polygons from node coordinates
    triangle_polys = []
    triangle_depths = []
    triangle_nodes = []
    triangle_coast = []
    triangle_area = []
    mesh_coastline, coastal_nodes = ecs.boundary_polygon_from_triangles(mesh.x, mesh.y, mesh.triangles)
    for tri in mesh.triangles:
        coords = [(mesh.x[n], mesh.y[n]) for n in tri]
        poly = Polygon(coords)
        # check if any of the triangle nodes are on the coastline and exclude from the depth calculation if so, as these will be land nodes with negative depths that will skew the depth calculation for the intersecting cells. This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
        # extract the nodes that are not on the coastline and calculate the mean depth from those
        non_coast_nodes = [n for n in tri if n not in coastal_nodes]
        # get the coastal ones as well for plotting
        coastal_nodes_in_tri = [n for n in tri if n in coastal_nodes]
        depth = mesh.depth[non_coast_nodes].mean() # elemen mean only considers non-coastal nodes to avoid skewing the depth calculation with land nodes that have negative depths. This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
        triangle_coast.append([Point(mesh.x[n], mesh.y[n]) for n in coastal_nodes_in_tri ])
        triangle_nodes.append(non_coast_nodes)
        triangle_area.append(poly.area)
        triangle_polys.append(poly)
        triangle_depths.append(depth)

    fvcom_elements = gpd.GeoDataFrame(
        {"depth": triangle_depths, "non_coast_nodes": triangle_nodes, "coast": triangle_coast, "nodes": mesh.triangles.tolist(), "area_m2": triangle_area},
        geometry=triangle_polys,
        crs="EPSG:32630"  # adjust to your projected CRS — looks like UTM 30N given the coords
    )
    # calculate the minimum depth overall and add a temporary offset to ensure all depths are positive for volume calculation. This is needed to get the area of the intersection with the coastline
    min_depth = fvcom_elements["depth"].min()
    if min_depth < 0:
        fvcom_elements["depth"] = fvcom_elements["depth"] - min_depth + 1  # add offset to make all depths positive
    fvcom_proj = fvcom_elements.to_crs(wet_cells_gdf.crs)

    wet_cells_gdf["mean_depth_m"] = wet_cells_gdf.geometry.apply(
        lambda g: area_weighted_depth(g, fvcom_proj)
    )
    wet_cells_gdf["mean_volume_m3"] = wet_cells_gdf["wet_area_m2"] * wet_cells_gdf["mean_depth_m"]
    # correct the depth offset by subtracting the same offset from the mean depth and mean volume
    wet_cells_gdf["mean_depth_m"] = wet_cells_gdf["mean_depth_m"] + min_depth - 1

    # apply to all wet cells
    results = []
    all_pieces = []

    for idx, row in cells_proj.iterrows():
        volume, pieces = cell_volume_and_elements(row.geometry, fvcom_proj)
        results.append(volume)
        for p in pieces:
            p["cell_id"] = row["id"]
            all_pieces.append(p)

    wet_cells_gdf["element_volume_m3"] = results
    # extract area from all_pieces and save it to wet_cells_gdf for plotting/QGIS
    area_dict = {}
    intersect_polys = {}
    depth_dict = {}
    for piece in all_pieces:
        cell_id = piece["cell_id"]
        area = piece["area_m2"]
        intersect_polys[cell_id] = piece["geometry"]
        if cell_id in area_dict:
            area_dict[cell_id] += area
            depth_dict[cell_id].append(piece["depth"])
        else:
            area_dict[cell_id] = area
            depth_dict[cell_id] = [piece["depth"]]
    wet_cells_gdf["element_area_m2"] = wet_cells_gdf["id"].map(area_dict)
    wet_cells_gdf["q_depth_m"] = wet_cells_gdf["id"].map(lambda x: np.quantile(np.array(depth_dict[x]), quantile) if x in depth_dict else np.nan) + min_depth - 1
    wet_cells_gdf["intersection_geometry"] = wet_cells_gdf["id"].map(intersect_polys)
    # add the area calculated from the intersection with the coastline (i.e. the real wet area)
    wet_cells_gdf["real_intersection_area_m2"] = cells_proj["wet_area_m2"]
    # save element pieces for plotting/QGIS
    pieces_gdf = gpd.GeoDataFrame(all_pieces, crs=fvcom_proj.crs)
    # remove depth offset from depths
    pieces_gdf["depth"] = pieces_gdf["depth"] + min_depth - 1
    pieces_gdf.to_crs("EPSG:4326").to_file(experiment+"intersection_elements.geojson", driver="GeoJSON")
    # plot_polygons_stepwise(gpd.GeoDataFrame(cells, crs=section_proj.crs), coastline_gdf=coastline.to_crs(section_proj.crs), id_col="id", mesh = pieces_gdf)
    fig, ax = plt.subplots(figsize=(12, 12))

    wet_cells_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, label='wet cells')
    pieces_gdf.plot(ax=ax, column='depth', cmap='viridis', edgecolor='black', linewidth=0.2, legend=True)

    plt.title('FVCOM elements intersecting wet cells (coloured by depth)')
    fig.savefig(os.path.join(fig_dir, experiment+"fvcom_elements_intersecting_wet_cells.png"), dpi=300)
    # limite the axis to the top northern section by setting the limit to the northern 10% of the y range.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min + 0.9 * y_range, y_max)
    fig.savefig(os.path.join(fig_dir, experiment+"fvcom_elements_intersecting_wet_cells_zoom.png"), dpi=300)

    # estimate a "real" depth by dividing the volume by the real intersection area (which is the area of the cell intersected with the sea polygon, i.e. the actual wet area). This is needed to get a more accurate estimate of the depth for each cell based on the intersection with the coastline
    wet_cells_gdf["real_mean_depth_m"] = wet_cells_gdf["element_volume_m3"] / wet_cells_gdf["element_area_m2"]
    # remove depth offset from real_mean_depth_m
    wet_cells_gdf["real_mean_depth_m"] = wet_cells_gdf["real_mean_depth_m"] + min_depth - 1
    wet_cells_gdf.to_crs("EPSG:4326").to_file(experiment+"wet_cells_with_depths.geojson", driver="GeoJSON")
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
    fig.savefig(os.path.join(fig_dir, experiment+'intersection_area_comparison.png'), dpi=300)
    fig, ax2 = plt.subplots(figsize=(10, 6))
    color = 'tab:green'
    ax2.set_xlabel('Section ID')
    ax2.set_ylabel('Mean Depth (m)', color=color)
    ax2.plot(wet_cells_gdf["id"], wet_cells_gdf["q_depth_m"], color=color, marker='o', label=f'Real Quantile {quantile} Depth')
    ax2.plot(wet_cells_gdf["id"], wet_cells_gdf["real_mean_depth_m"], color='tab:olive', marker='s', label='Estimated Real Mean Depth')
    ax2.plot(wet_cells_gdf["id"], wet_cells_gdf["mean_depth_m"], color='tab:olive', marker='x', label='Calculated Mean Depth')
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.legend(loc='upper left')
    # save the figure
    fig.savefig(os.path.join(fig_dir, experiment+'mean_depth_comparison.png'), dpi=300)
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
    fig.savefig(os.path.join(fig_dir, experiment+'volume_comparison.png'), dpi=300)
    # save a dictionary of list of nodes and their corresponding section id and depths for use in mesh generation
    nodes_dict = {}
    depth_dict = {}
    for idx, row in points_gdf.iterrows():
        section_id = row["section_id"]
        nodes_dict[idx] = {"section_id": section_id, "lon": row["lon"], "lat": row["lat"], "x": row["x"], "y": row["y"]}
        try:
            depth_dict[section_id] = {"section_id": section_id, "real_mean_depth_m": wet_cells_gdf.loc[wet_cells_gdf["id"] == section_id, "q_depth_m"].values[0], "element_area_m2": wet_cells_gdf.loc[wet_cells_gdf["id"] == section_id, "element_area_m2"].values[0]}
        except IndexError:
            print(f"Warning: No depth found for section_id {section_id}")   
    # save nodes_dict and depth_dict as csv for use in mesh generation
    import csv
    with open(os.path.join("../transect_grid", experiment+ "nodes_dict.csv"), "w", newline="") as csvfile:
        fieldnames = ["node_id", "section_id", "lon", "lat", "x", "y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node_id, data in nodes_dict.items():
            writer.writerow({"node_id": node_id, **data})
    with open(os.path.join("../transect_grid", experiment+ "depth_dict.csv"), "w", newline="") as csvfile:
        fieldnames = ["section_id", "real_mean_depth_m", "element_area_m2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for section_id, depth in depth_dict.items():
            writer.writerow({"section_id": section_id, **depth})
# ordered_gdf = gpd.GeoDataFrame(
#     [{"order": i, "geometry": ordered_lines[i]} for i in range(len(ordered_lines))],
#     crs=coastline.crs
# )
# ordered_gdf.to_file("ordered_coastline.geojson", driver="GeoJSON")
# # close the outer boundary
# coords = list(merged.coords)
# if coords[0] != coords[-1]:
#     coords.append(coords[0])
# sea_polygon = Polygon(LinearRing(coords))
# if sea_polygon.geom_type == 'MultiPolygon':
#     sea_polygon = max(sea_polygon.geoms, key=lambda p: p.area)
# # punch island holes out
# island_polygons = [Polygon(ring.coords) for ring in closed_lines]
# islands = make_valid(unary_union(island_polygons))
# sea_masked = make_valid(sea_polygon.difference(islands))

# print(explain_validity(sea_masked))


# # Get all individual linestrings
# lines_list = []
# for geom in coast_lines.geometry:
#     if geom.geom_type == 'MultiLineString':
#         lines_list.extend(list(geom.geoms))
#     else:
#         lines_list.append(geom)

# print(f"Number of segments: {len(lines_list)}")
# # Build ordered chain using nearest-neighbour
# remaining = list(range(len(lines_list)))
# ordered = [remaining.pop(0)]
# # track which end of the last segment we're connecting from
# current_end = get_endpoints(lines_list[ordered[-1]])[1]

# while remaining:
#     # find closest segment endpoint to current_end
#     best_idx = None
#     best_dist = np.inf
#     best_flip = False
    
#     for idx in remaining:
#         start, end = get_endpoints(lines_list[idx])
#         d_start = point_dist(current_end, start)
#         d_end   = point_dist(current_end, end)
#         if d_start < best_dist:
#             best_dist = d_start
#             best_idx = idx
#             best_flip = False
#         if d_end < best_dist:
#             best_dist = d_end
#             best_idx = idx
#             best_flip = True
    
#     remaining.remove(best_idx)
#     if best_flip:
#         # reverse the segment so it connects head-to-tail
#         lines_list[best_idx] = LineString(list(lines_list[best_idx].coords)[::-1])
#     ordered.append(best_idx)
#     current_end = get_endpoints(lines_list[best_idx])[1]

# ordered_lines = [lines_list[i] for i in ordered]

# # merge into single linestring
# merged = linemerge(ordered_lines)
# print(f"Merged type: {merged.geom_type}")  # should be LineString, not Multi

# # close it
# coords = list(merged.coords)
# if coords[0] != coords[-1]:
#     coords.append(coords[0])

# from shapely.geometry import Polygon, LinearRing
# sea_polygon = Polygon(LinearRing(coords))
# print(explain_validity(sea_polygon))

# merged = linemerge(unary_union(coast_lines.geometry.values))

# if merged.geom_type == 'MultiLineString':
#     # gaps in coastline — close via bounding box
#     bbox = box(*merged.bounds)
#     closed = unary_union([merged, bbox.boundary])
#     polys = list(polygonize(closed))
#     sea_polygon = max(polys, key=lambda p: p.area)
# else:
#     # single linestring — force close it
#     coords = list(merged.coords)
#     if coords[0] != coords[-1]:
#         coords.append(coords[0])
#     sea_polygon = Polygon(LinearRing(coords))

# print(explain_validity(sea_polygon))


# merged = linemerge(unary_union(coast_lines.geometry.values))

# # Find the two free endpoints of the coastline
# # For a MultiLineString, collect all endpoints and find the unpaired ones
# from collections import Counter

# endpoint_counts = Counter()
# for geom in (merged.geoms if merged.geom_type == 'MultiLineString' else [merged]):
#     coords = list(geom.coords)
#     endpoint_counts[coords[0]] += 1
#     endpoint_counts[coords[-1]] += 1

# # Free endpoints appear only once
# free_endpoints = [pt for pt, count in endpoint_counts.items() if count == 1]
# print(f"Free endpoints: {free_endpoints}")  # should be exactly 2

# # Connect them with a straight line across the open boundary
# closing_line = LineString([free_endpoints[0], free_endpoints[1]])

# # Close the domain
# closed = unary_union([merged, closing_line])
# polys = list(polygonize(closed))
# sea_polygon = max(polys, key=lambda p: p.area)



# fig, ax = plt.subplots(figsize=(10, 10))
# gpd.GeoDataFrame(geometry=[sea_polygon], crs=coastline.crs).plot(ax=ax, color='lightblue', alpha=0.5)
# coastline.plot(ax=ax, color='grey', linewidth=0.5)
# section.plot(ax=ax, color='red', linewidth=1)
# plt.title('Sections and Coastline')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.savefig(os.path.join(fig_dir, 'sections_and_coastline.png'), dpi=300)

# # plot sections and coastline to check they look right
# if plot_sections_and_coastline:
#     fig, ax = plt.subplots(figsize=(10, 10))
#     coastline.plot(ax=ax, color='lightgrey')
#     section.plot(ax=ax, color='blue', edgecolor='black')
#     plt.title('Sections and Coastline')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.savefig(os.path.join(fig_dir, 'sections_and_coastline.png'), dpi=300)

# coast = make_valid(coastline.union_all())
# # Also worth fixing the islands individually before the difference
# print(explain_validity(coast))   # before difference
# section["geometry"] = section.geometry.apply(make_valid)
# # make each section linestring a polygon by connecting adjacent linestrings and filling in the gaps. This is needed to get the area of the intersection with the coastline
# cells = []
# for i in range(len(section) - 1):
#     poly = Polygon([section["geometry"][i].coords[0], section["geometry"][i].coords[-1], section["geometry"][i+1].coords[-1], section["geometry"][i+1].coords[0]])
#     cells.append({"id": i, "geometry": poly})

# cells_gdf = gpd.GeoDataFrame(cells, crs=f"EPSG:{section.crs.to_epsg()}")
# # convert coast to utm 

# # Get all island polygons as a single geometry
# section["geometry"] = section.geometry.intersection(coast)
# cells_proj = cells_proj.to_crs("EPSG:32633")
# cells_proj["wet_area_m2"] = cells_proj.geometry.area
# cells_proj = cells_proj[cells_proj["wet_area_m2"] > 0]
# # write the clipped section polygons to a new GeoJSON for sanity check
# cells_proj.to_file(args.out_dir / f"{args.out_prefix}_sections_clipped.geojson", driver="GeoJSON")
