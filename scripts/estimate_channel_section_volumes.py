#!/usr/bin/env python3
"""Estimate channel volumes in non-overlapping sections along a centreline.

Inputs
- Centreline node list pickle (expects key: channel_nodes by default)
- Triangular mesh (.2dm recommended)
- Optional coastline shapefile for extra clipping

Outputs
- CSV: section metrics (area/volume per section)
- GeoJSON: section polygons with attributes
"""

from __future__ import annotations
import getpass
import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plot_transect as plot_tr
import shapefile
from pyproj import CRS, Transformer
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import polygonize, transform as shp_transform, unary_union
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
DEFAULT_PYFVCOM2_ROOT = Path(f"/home/{getpass.getuser()}/Code/pyfvcom2")
if DEFAULT_PYFVCOM2_ROOT.exists() and str(DEFAULT_PYFVCOM2_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PYFVCOM2_ROOT))

HAS_PYFVCOM2 = False
try:
    from pyfvcom2.mesh_reader import read_mesh_file, read_sms_mesh  # type: ignore[import-not-found]

    HAS_PYFVCOM2 = True
    print("pyfvcom2 import successful. Will attempt to use it for mesh loading if possible.")
except ImportError:
    HAS_PYFVCOM2 = False
    print("pyfvcom2 not importable. Will only be able to read .2dm meshes with simple parser.")


@dataclass
class Mesh:
    nodes: np.ndarray
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    triangles: np.ndarray
    mesh_file: Optional[Path] = None
    mesh_type: Optional[str] = None


def read_2dm_simple(mesh_path: Path) -> Mesh:
    tri_raw = []
    node_xyz: Dict[int, Tuple[float, float, float]] = {}

    with mesh_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().split()
            if not s:
                continue
            if s[0] == "E3T" and len(s) >= 5:
                tri_raw.append((int(s[2]), int(s[3]), int(s[4])))
            elif s[0] == "ND" and len(s) >= 5:
                node_xyz[int(s[1])] = (float(s[2]), float(s[3]), float(s[4]))

    if not tri_raw or not node_xyz:
        raise ValueError(f"Could not parse E3T/ND records in {mesh_path}")

    node_ids = np.array(sorted(node_xyz.keys()), dtype=int)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids.tolist())}
    xyz = np.array([node_xyz[nid] for nid in node_ids], dtype=float)

    tri = np.array([[id_to_idx[a], id_to_idx[b], id_to_idx[c]] for a, b, c in tri_raw], dtype=int)
    return Mesh(nodes=node_ids, x=xyz[:, 0], y=xyz[:, 1], depth=xyz[:, 2], triangles=tri)


def load_mesh(mesh_path: Path) -> Mesh:
    suffix = mesh_path.suffix.lower()

    if HAS_PYFVCOM2:
        print(f"Attempting to load mesh with pyfvcom2: {mesh_path}")
        try:
            if suffix == ".2dm":
                m = read_sms_mesh(str(mesh_path), nodestrings=False)
                mesh_type = "sms_2dm"
                print(f"Loaded mesh with pyfvcom2 SMS 2DM reader: {mesh_path}")
            else:
                print(f"Loading mesh with pyfvcom2 FVCOM reader: {mesh_path}")
                m = read_mesh_file(str(mesh_path), "fvcom")
                mesh_type = "fvcom"

            return Mesh(
                nodes=np.asarray(m.nodes, dtype=int),
                x=np.asarray(m.x1, dtype=float),
                y=np.asarray(m.x2, dtype=float),
                depth=np.asarray(m.x3, dtype=float),
                triangles=np.asarray(m.triangle, dtype=int),
                mesh_file=mesh_path,
                mesh_type=mesh_type,
            )
        except (ValueError, OSError, RuntimeError):
            if suffix != ".2dm":
                raise

    if suffix == ".2dm":
        return read_2dm_simple(mesh_path)

    raise RuntimeError("Mesh loading failed. For non-.2dm meshes, ensure pyfvcom2 is importable.")


def load_channel_nodes(pickle_path: Path, key: str = "channel_nodes") -> np.ndarray:
    with pickle_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Expected pickle payload to be a dict.")
    if key not in payload:
        raise KeyError(f"Key '{key}' not found in pickle payload. Available keys: {list(payload.keys())}")

    nodes = np.asarray(payload[key], dtype=int).reshape(-1)
    if nodes.size < 2:
        raise ValueError("Need at least two centreline nodes.")

    dedup = [int(nodes[0])]
    for n in nodes[1:]:
        ni = int(n)
        if ni != dedup[-1]:
            dedup.append(ni)
    return np.asarray(dedup, dtype=int)


def infer_depth_positive_down(depth_raw: np.ndarray, mode: str = "auto") -> np.ndarray:
    depth = np.asarray(depth_raw, dtype=float).copy()
    if mode == "positive":
        return np.abs(depth)
    if mode == "negative":
        return -np.abs(depth)

    if float(np.nanmedian(depth)) < 0:
        depth = -depth
    return depth


def infer_projected_xy(
    x: np.ndarray,
    y: np.ndarray,
    epsg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str, Optional[Transformer]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    is_geo = (
        np.nanmin(x) >= -180
        and np.nanmax(x) <= 180
        and np.nanmin(y) >= -90
        and np.nanmax(y) <= 90
    )

    if not is_geo:
        return x, y, "native_projected", None

    lon0 = float(np.nanmean(x))
    lat0 = float(np.nanmean(y))

    if epsg is None:
        zone = int(math.floor((lon0 + 180) / 6) + 1)
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone

    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    xp, yp = transformer.transform(x, y)
    return np.asarray(xp), np.asarray(yp), f"epsg:{epsg}", transformer


def centerline_from_nodes(mesh: Mesh, channel_nodes: np.ndarray) -> np.ndarray:
    node_to_idx = {int(nid): i for i, nid in enumerate(mesh.nodes.tolist())}

    missing = [int(n) for n in channel_nodes if int(n) not in node_to_idx]
    if missing:
        preview = missing[:10]
        raise ValueError(f"{len(missing)} centreline node IDs not in mesh. Examples: {preview}")

    idx = np.asarray([node_to_idx[int(n)] for n in channel_nodes], dtype=int)
    return np.column_stack((mesh.x[idx], mesh.y[idx]))


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    dxy = np.diff(points, axis=0)
    seg = np.hypot(dxy[:, 0], dxy[:, 1])
    return np.concatenate(([0.0], np.cumsum(seg)))

def resample_centerline(channel_nodes: np.ndarray, spacing: float, distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from itertools import islice
    it = iter(enumerate(channel_nodes))
    nodes_in_line = [channel_nodes[0]]
    distance_km = np.diff(distance)
    distance_out = [0.0]
    # zero_idx, idx = next(islice(it, 0, None))
    # lapsed_dist = np.cumsum([0] + distance_km[zero_idx:])
    # idx2 = np.where(lapsed_dist > spacing)[0][0]
    # distance_out.append(distance[idx2+zero_idx+1])
    # nodes_in_line.append(channel_nodes[idx2+zero_idx])
    # zero_idx, idx = next(islice(it, idx2, None))
    # distance_out
    # nodes_in_line
    for zero_idx, idx in it:
        print(idx, zero_idx)
        lapsed_dist = np.cumsum([0] + distance_km[zero_idx:])
        if zero_idx == len(distance_km):
            break
        # find first index farther than 100m
        idx2 = np.where(lapsed_dist > spacing)[0][0]
        distance_out.append(distance[idx2+zero_idx+1])
        nodes_in_line.append(channel_nodes[idx2+zero_idx])
        # advance iterable by idx2 amounts
        next(islice(it, idx2, idx2), None)
        # print(zero_idx)
        # zero_idx += idx2
    return np.array(nodes_in_line, dtype=int), np.array(distance_out)

def resample_polyline(points: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    s = cumulative_distance(points)
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Centreline length is zero.")

    stations = np.arange(0.0, total, spacing)
    if stations.size == 0 or stations[-1] < total:
        stations = np.append(stations, total)

    x = np.interp(stations, s, points[:, 0])
    y = np.interp(stations, s, points[:, 1])
    return np.column_stack((x, y)), stations


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    if n == 0:
        return np.array([1.0, 0.0])
    return v / n


def local_tangents(sample_points: np.ndarray) -> np.ndarray:
    n = sample_points.shape[0]
    tangents = np.zeros((n, 2), dtype=float)
    for i in range(n):
        if i == 0:
            tangents[i] = unit(sample_points[1] - sample_points[0])
        elif i == n - 1:
            tangents[i] = unit(sample_points[-1] - sample_points[-2])
        else:
            tangents[i] = unit(sample_points[i + 1] - sample_points[i - 1])
    return tangents
def local_normals(sample_points: np.ndarray) -> np.ndarray:
    tangents = local_tangents(sample_points)
    # Rotate 90 degrees: (tx, ty) -> (-ty, tx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals
def normal_segments(sample_points: np.ndarray, half_width: float) -> list[tuple]:
    """
    Returns list of (x0, y0, x1, y1) segments, one per sample point.
    half_width is in the same units as sample_points.
    """
    normals = local_normals(sample_points)
    segments = []
    for pt, n in zip(sample_points, normals):
        p0 = pt - n * half_width
        p1 = pt + n * half_width
        segments.append((p0, p1))
    return segments
def smooth_normals (sample_points: np.ndarray, smoothing_factor: float, half_width: float) -> np.ndarray:
    from scipy.interpolate import UnivariateSpline

    # Fit splines to x and y separately as a function of cumulative distance
    dist = np.cumsum(np.r_[0, np.hypot(np.diff(sample_points[:, 0]), np.diff(sample_points[:, 1]))])

    spl_x = UnivariateSpline(dist, sample_points[:, 0], s=smoothing_factor)
    spl_y = UnivariateSpline(dist, sample_points[:, 1], s=smoothing_factor)

    # Evaluate at finer resolution
    t = np.linspace(dist[0], dist[-1], 1000)
    x_smooth = spl_x(t)
    y_smooth = spl_y(t)

    # Tangent = derivative of spline
    dx = spl_x.derivative()(t)
    dy = spl_y.derivative()(t)
    # interpolate back to original sample points
    dx_orig = np.interp(dist, t, dx)
    dy_orig = np.interp(dist, t, dy)
    tangent_angle = np.hypot(dy_orig, dx_orig)
    normals = np.column_stack([-dy_orig / tangent_angle, dx_orig / tangent_angle])
    segments = []
    for pt, n in zip(sample_points, normals):
        p0 = pt - n * half_width
        p1 = pt + n * half_width
        segments.append((p0, p1))
    return segments

def boundary_polygon_from_triangles(x: np.ndarray, y: np.ndarray, tri: np.ndarray) -> Any:
    """ Construct the exterior boundary polygon of the mesh from its triangles. This is used for clipping section polygons to the domain. """
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)

    edge_count: Dict[Tuple[int, int], int] = {}
    for a, b in edges.tolist():
        key = (int(a), int(b))
        edge_count[key] = edge_count.get(key, 0) + 1
    # boundary edges are those that appear only once in the triangle list
    boundary = [e for e, c in edge_count.items() if c == 1]
    coast_nodes = set()
    for a, b in boundary:
        coast_nodes.add(a)
        coast_nodes.add(b)
    # convert boundary edges to shapely lines
    lines = [LineString([(x[a], y[a]), (x[b], y[b])]) for a, b in boundary]
    # link all lines into close polygons and take the largest one as the main domain polygon
    polys = list(polygonize(lines))
    if not polys:
        raise RuntimeError("Could not construct mesh boundary polygon.")
    # If multiple polygons are returned, take the largest one as the main domain
    merged = unary_union(polys)
    if merged.geom_type == "Polygon":
        return merged, coast_nodes

    parts = sorted(list(getattr(merged, "geoms", [])), key=lambda g: g.area, reverse=True)
    return parts[0], coast_nodes

def build_coastline_polygons_from_shapefile(shp_path: Path) -> Any:

    coast = gpd.read_file(shp_path)
    print(coast.geom_type.value_counts())
    print(coast.crs)
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import polygonize, unary_union

    # Split by geometry type
    lines = coast[coast.geom_type.isin(["LineString", "MultiLineString"])]
    polygons = coast[coast.geom_type.isin(["Polygon", "MultiPolygon"])]

    # Merge all lines into one geometry, then polygonize
    merged_lines = unary_union(lines.geometry.values)
    land_from_lines = list(polygonize(merged_lines))

    if not land_from_lines:
        # Lines don't form closed rings — need to close the coastline manually
        # by adding a bounding box edge along the domain boundary
        from shapely.geometry import box
        bbox = box(*lines.total_bounds)  # minx, miny, maxx, maxy
        closed = unary_union([merged_lines, bbox.boundary])
        land_from_lines = list(polygonize(closed))
    # Combine polygonized coastline + explicit island polygons
    island_geoms = polygons.geometry.values.tolist()
    all_land = unary_union(land_from_lines + island_geoms)
    return all_land, polygons
def coastline_polygon_from_shapefile(shp_path: Path) -> Optional[Any]:
    reader = shapefile.Reader(str(shp_path))
    geoms = []
    for s in reader.shapes():
        if s is None:
            continue
        geoms.append(shape(dict(s.__geo_interface__)))
    if not geoms:
        return None

    g = unary_union(geoms)
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon":
        return max(list(getattr(g, "geoms", [])), key=lambda gg: gg.area)

    if g.geom_type in ("LineString", "MultiLineString"):
        lines = [g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))
        polys = list(polygonize(lines))
        if not polys:
            return None
        merged = unary_union(polys)
        if merged.geom_type == "Polygon":
            return merged
        return max(list(getattr(merged, "geoms", [])), key=lambda gg: gg.area)

    return None


def half_plane_polygon(c: np.ndarray, n: np.ndarray, t: np.ndarray, keep_positive: bool, radius: float) -> Any:
    a = c + n * radius
    b = c - n * radius
    shift = t * (2.5 * radius if keep_positive else -2.5 * radius)
    return Polygon([tuple(a), tuple(b), tuple(b + shift), tuple(a + shift)])

def cross_section_stats(x, y, tri, depth, bank_left, bank_right, n_points=200):
    """
    Select triangles by discretising the cross-section line and finding
    the nearest triangle to each point, then deduplicating.

    Parameters
    ----------
    x, y        : node coordinates
    tri         : connectivity matrix (nele x 3)
    depth       : depth at each node
    bank_left   : shapely Point
    bank_right  : shapely Point
    n_points    : number of points to discretise the cross-section line
    """
    # Discretise line between the two bank points
    line = LineString([bank_left, bank_right])
    sample_pts = np.array([
        line.interpolate(t, normalized=True).coords[0]
        for t in np.linspace(0, 1, n_points)
    ])

    # Triangle centroids
    cx = (x[tri[:, 0]] + x[tri[:, 1]] + x[tri[:, 2]]) / 3
    cy = (y[tri[:, 0]] + y[tri[:, 1]] + y[tri[:, 2]]) / 3
    centroids = np.column_stack([cx, cy])

    # For each sample point find the nearest triangle centroid
    # Use a KDTree for efficiency
    from scipy.spatial import KDTree
    tree = KDTree(centroids)
    _, idx = tree.query(sample_pts)

    # Deduplicate while preserving order
    seen = set()
    selected = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            selected.append(i)
    selected = np.array(selected)

    if len(selected) == 0:
        return None

    # Triangle areas via cross product
    x0, y0 = x[tri[selected, 0]], y[tri[selected, 0]]
    x1, y1 = x[tri[selected, 1]], y[tri[selected, 1]]
    x2, y2 = x[tri[selected, 2]], y[tri[selected, 2]]
    areas = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    # Area-weighted mean depth
    depth_tri = (depth[tri[selected, 0]] + depth[tri[selected, 1]] + depth[tri[selected, 2]]) / 3
    total_area    = areas.sum()
    average_depth = np.average(depth_tri, weights=areas)

    return {
        "total_area":    total_area,
        "average_depth": average_depth,
        "n_triangles":   len(selected),
        "tri_indices":   selected,
    }
def section_geopandas(section_nodes: dict, args) -> gpd.GeoDataFrame:
    cells = []
    for i in range(len(section_nodes["start"]) - 1):
        poly = Polygon([section_nodes["start"][i], section_nodes["end"][i], section_nodes["end"][i+1], section_nodes["start"][i+1]])
        cells.append({"id": i, "geometry": poly})

    cells_gdf = gpd.GeoDataFrame(cells, crs=f"EPSG:{args.utm_epsg}")
    return cells_gdf
def section_polygons(sample_points: np.ndarray, tangents: np.ndarray, domain_polygon: Any, radius: float) -> List[Any]:
    sections: List[Any] = []
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    for i in range(sample_points.shape[0] - 1):
        c0, c1 = sample_points[i], sample_points[i + 1]
        t0, t1 = tangents[i], tangents[i + 1]
        n0, n1 = normals[i], normals[i + 1]

        hp0 = half_plane_polygon(c0, n0, t0, keep_positive=True, radius=radius)
        hp1 = half_plane_polygon(c1, n1, t1, keep_positive=False, radius=radius)

        sec = domain_polygon.intersection(hp0).intersection(hp1)
        if sec.is_empty:
            sections.append(Polygon())
            continue

        if sec.geom_type == "Polygon":
            sections.append(sec)
        else:
            parts = [g for g in getattr(sec, "geoms", []) if g.geom_type == "Polygon"]
            sections.append(max(parts, key=lambda g: g.area) if parts else Polygon())

    return sections


def triangle_polygons(x: np.ndarray, y: np.ndarray, tri: np.ndarray) -> Tuple[List[Any], np.ndarray]:
    polys: List[Any] = []
    areas = np.zeros(tri.shape[0], dtype=float)

    for i, (a, b, c) in enumerate(tri.tolist()):
        p = Polygon([(x[a], y[a]), (x[b], y[b]), (x[c], y[c])])
        polys.append(p)
        areas[i] = p.area

    return polys, areas


def integrate_sections(section_polys: Sequence[Any], tri_polys: Sequence[Any], tri_depth: np.ndarray) -> List[Dict[str, float]]:
    tree = STRtree(list(tri_polys))
    id_to_index = {id(g): i for i, g in enumerate(tri_polys)}

    rows: List[Dict[str, float]] = []
    for i, sec in enumerate(section_polys):
        if sec.is_empty:
            rows.append(
                {
                    "section_id": i,
                    "section_area_m2": 0.0,
                    "section_volume_m3": 0.0,
                    "n_triangles_touched": 0,
                }
            )
            continue

        candidates = tree.query(sec)

        total_area = 0.0
        total_volume = 0.0
        touched = 0

        use_index_mode = len(candidates) > 0 and isinstance(candidates[0], (int, np.integer))
        if use_index_mode:
            candidate_indices = [int(ii) for ii in candidates]
        else:
            candidate_indices = [id_to_index[id(g)] for g in candidates]

        for tidx in candidate_indices:
            g = tri_polys[tidx]
            inter = sec.intersection(g)
            if inter.is_empty:
                continue
            a = float(inter.area)
            if a <= 0:
                continue
            touched += 1
            total_area += a
            total_volume += a * float(tri_depth[tidx])

        rows.append(
            {
                "section_id": i,
                "section_area_m2": total_area,
                "section_volume_m3": total_volume,
                "n_triangles_touched": touched,
            }
        )

    return rows


def section_index_for_centroids(
    cx: np.ndarray,
    cy: np.ndarray,
    boundaries: np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray:
    """Assign triangle centroids to non-overlapping strips between dividers."""
    n_sections = boundaries.shape[0] - 1
    idx = np.full(cx.shape, -1, dtype=int)

    for k in range(n_sections):
        b0 = boundaries[k]
        b1 = boundaries[k + 1]
        t0 = tangents[k]
        t1 = tangents[k + 1]

        lhs = (cx - b0[0]) * t0[0] + (cy - b0[1]) * t0[1]
        rhs = (cx - b1[0]) * t1[0] + (cy - b1[1]) * t1[1]

        in_k = (lhs >= 0.0) & (rhs < 0.0)
        idx[in_k] = k

    return idx


def integrate_sections_centroid(
    tri: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tri_area: np.ndarray,
    tri_depth: np.ndarray,
    sample_points: np.ndarray,
    tangents: np.ndarray,
) -> List[Dict[str, float]]:
    """Fast approximate integration using full-triangle centroid assignment."""
    cx = x[tri].mean(axis=1)
    cy = y[tri].mean(axis=1)
    tri_volume = tri_area * tri_depth

    sec_idx = section_index_for_centroids(cx, cy, sample_points, tangents)
    n_sections = sample_points.shape[0] - 1

    valid = sec_idx >= 0
    sec_count = np.bincount(sec_idx[valid], minlength=n_sections)
    sec_area = np.bincount(sec_idx[valid], weights=tri_area[valid], minlength=n_sections)
    sec_volume = np.bincount(sec_idx[valid], weights=tri_volume[valid], minlength=n_sections)

    rows: List[Dict[str, float]] = []
    for i in range(n_sections):
        rows.append(
            {
                "section_id": i,
                "section_area_m2": float(sec_area[i]),
                "section_volume_m3": float(sec_volume[i]),
                "n_triangles_touched": int(sec_count[i]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, float]], stations: np.ndarray) -> None:
    fieldnames = [
        "section_id",
        "s_start_m",
        "s_end_m",
        "s_mid_m",
        "section_area_m2",
        "section_volume_m3",
        "n_triangles_touched",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            i = int(r["section_id"])
            out = dict(r)
            out["s_start_m"] = float(stations[i])
            out["s_end_m"] = float(stations[i + 1])
            out["s_mid_m"] = 0.5 * (out["s_start_m"] + out["s_end_m"])
            w.writerow(out)


def write_geojson(path: Path, section_polys: Sequence[Any], rows: Sequence[Dict[str, float]], stations: np.ndarray) -> None:
    features = []
    for r, poly in zip(rows, section_polys):
        i = int(r["section_id"])
        props = dict(r)
        props["s_start_m"] = float(stations[i])
        props["s_end_m"] = float(stations[i + 1])
        props["s_mid_m"] = 0.5 * (props["s_start_m"] + props["s_end_m"])
        features.append({"type": "Feature", "geometry": mapping(poly), "properties": props})

    fc = {"type": "FeatureCollection", "features": features}
    with path.open("w", encoding="utf-8") as f:
        json.dump(fc, f)

def write_geojson_sections(path: Path, section_nodes: Sequence[Any],section_area: Optional[Sequence[float]], utm_epsg: Optional[int]) -> None:
    features = []
    if section_area is None:
        section_area = [None] * len(section_nodes)
    for n, area in zip(section_nodes, section_area):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(n["start"]), list(n["end"])]
            },
            "properties": {"id": n["id"],
                           "section_area": float(area) if area is not None else None,}
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{utm_epsg}" if utm_epsg is not None else "EPSG:4326"}
        },
        "features": features
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
        
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate FVCOM section volumes along a centreline")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("/home/ricardo/OneDrive/Projects/Tamar_pyGETM/tamar_v0/tamar_v0.2dm"),
        help="Path to mesh file (.2dm recommended)",
    )
    parser.add_argument(
        "--transect-pickle",
        type=Path,
        default=Path("/home/ricardo/OneDrive/Projects/Tamar_pyGETM/transect_grid/tamar_transect_nodes.pk"),
        help="Pickle with ordered centreline nodes",
    )
    parser.add_argument("--channel-key", default="channel_nodes", help="Pickle key for centreline nodes")
    parser.add_argument("--spacing", type=float, default=100.0, help="Section spacing in metres")
    parser.add_argument(
        "--coastline-shapefile",
        type=Path,
        default=None,
        help="Optional coastline shapefile for domain clipping",
    )
    parser.add_argument(
        "--depth-mode",
        choices=["auto", "positive", "negative"],
        default="auto",
        help="Node depth sign interpretation",
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=32630,
        help="Force projection EPSG if mesh is lon/lat",
    )
    parser.add_argument(
        "--integration-method",
        choices=["centroid", "intersection"],
        default="centroid",
        help="centroid is fast approximate; intersection is exact but slower",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/ricardo/OneDrive/Projects/Tamar_pyGETM/transect_grid/section_volume_outputs"),
        help="Output directory",
    )
    parser.add_argument("--out-prefix", default="tamar_sections_100m", help="Output filename prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)

    mesh_x, mesh_y, proj_info, mesh_transformer = infer_projected_xy(mesh.x, mesh.y, epsg=args.utm_epsg)
    mesh = Mesh(nodes=mesh.nodes, x=mesh_x, y=mesh_y, depth=mesh.depth, triangles=mesh.triangles)

    channel_nodes = load_channel_nodes(args.transect_pickle, key=args.channel_key)
    centerline = centerline_from_nodes(mesh, channel_nodes)
    # plot mesh and centerline for sanity check
    fig, ax = plot_tr.plot_mesh_and_transect(
        mesh,
        channel_nodes,
        out_png=args.out_dir / f"{args.out_prefix}_mesh_transect.png",
        mesh_linewidth=0.5,
        mesh_alpha=0.28,
        node_size=1.0,
    )
    domain = boundary_polygon_from_triangles(mesh.x, mesh.y, mesh.triangles)
    sample_points, stations = resample_polyline(centerline, spacing=args.spacing)
    tangents = local_tangents(sample_points)
    norm_segments = normal_segments(sample_points, half_width=15000)
    normals = local_normals(sample_points)
    section = {area: [] for area in ["total_area", "average_depth", "n_triangles", "station", "start", "end", "mid"]}
    for pt_center, (p0, p1), normal, station in zip(sample_points, norm_segments, normals, stations):
        
        line = LineString([p0, p1])
        inter = line.intersection(domain.boundary)
        
        if inter.is_empty:
            continue
        
        if inter.geom_type == "Point":
            pts = [inter]
        elif inter.geom_type == "MultiPoint":
            pts = list(inter.geoms)
        elif inter.geom_type == "GeometryCollection":
            pts = [g for g in inter.geoms if g.geom_type == "Point"]
        else:
            pts = []
        
        if not pts:
            continue

        # Split points by which side of the centre they fall on
        # using dot product of (pt - center) with the normal vector
        positive = []
        negative = []
        for pt in pts:
            diff = np.array([pt.x - pt_center[0], pt.y - pt_center[1]])
            side = np.dot(diff, normal)
            if side >= 0:
                positive.append(pt)
            else:
                negative.append(pt)

        # Pick closest on each side
        center = Point(pt_center)
        bank_pts = []
        if positive:
            bank_pts.append(min(positive, key=lambda p: center.distance(p)))
        if negative:
            bank_pts.append(min(negative, key=lambda p: center.distance(p)))

        for pt in bank_pts:
            ax.plot(pt.x, pt.y, 'rx', markersize=6)
        if len(bank_pts) == 2:
            ax.plot([bank_pts[0].x, bank_pts[1].x], [bank_pts[0].y, bank_pts[1].y], 'r-', linewidth=0.8)

            # Adaptive n_points based on cross-section width and approximate mesh resolution
        section_length = bank_pts[0].distance(bank_pts[1]) if len(bank_pts) == 2 else 200.0
        approx_tri_size = 25  # metres, approximate minimum triangle size in your mesh
        n_points = int(section_length / approx_tri_size) * 3  # oversample by 3x to be safe
        # n_points = max(n_points, 100)  # floor
        # if less than 2 points in bank_pts skip stats calculation
        if len(bank_pts) < 2:
            continue
        stats = cross_section_stats(mesh.x, mesh.y, mesh.triangles, mesh.depth, bank_pts[0], bank_pts[1], n_points=n_points)
        if stats:
            print(f"Area: {stats['total_area']:.1f} m²  Mean depth: {stats['average_depth']:.2f} m")
            # append stats to section dict for potential later use or plotting
            section["total_area"].append(stats["total_area"])
            section["average_depth"].append(stats["average_depth"])
            section["n_triangles"].append(stats["n_triangles"])
            section["station"].append(station)
            section["start"].append((bank_pts[0].x, bank_pts[0].y))
            section["end"].append((bank_pts[1].x, bank_pts[1].y))
            section["mid"].append((pt_center[0], pt_center[1]))
            # # Optionally plot the section polygon
            # px, py = stats["polygon"].exterior.xy
            # ax.fill(px, py, alpha=0.3, fc="blue")
    # print(normals)
    # # plot section normal dividers for sanity check
    # ax.plot(sample_points[:, 0] + normals[:, 0], sample_points[:, 1] + normals[:, 1], color="#dc2626", linestyle="--", marker="o", markersize=4, label="Section dividers")
    ax.legend()
    fig.savefig(args.out_dir / f"{args.out_prefix}_mesh_transect_sections.png")
    # plot section area and mean depth along the transect for sanity check
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(
        section["station"],
        section["total_area"],color = "#2563eb", marker="o", label="Section area (m²)"
    )
    ax2.set_xlabel("Distance along transect (m)")
    ax2.set_ylabel("Section area (m²)", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")
    ax3 = ax2.twinx()
    ax3.plot(
        section["station"],
        section["average_depth"],color = "#16a34a", marker="o", label="Mean depth (m)"
    )
    ax3.set_ylabel("Mean depth (m)", color="#16a34a")
    ax3.tick_params(axis="y", labelcolor="#16a34a")
    fig2.savefig(args.out_dir / f"{args.out_prefix}_section_stats.png")
    # print the list of coastal nodes to a file 
    with (args.out_dir / f"{args.out_prefix}_coastal_nodes.txt").open("w") as f:
        for i, (x, y) in enumerate(zip(mesh.x[::2], mesh.y[::2])):
            pt = Point(x, y)
            if domain.boundary.distance(pt) < 1e-6:  # threshold for being considered coastal
                f.write(f"{mesh.nodes[i]}: ({x:.2f}, {y:.2f})\n")
        # finish with the last point
            if domain.boundary.distance(Point(mesh.x[-1], mesh.y[-1])) < 1e-6:
                f.write(f"{mesh.nodes[-1]}: ({mesh.x[-1]:.2f}, {mesh.y[-1]:.2f})\n")
    # print the section dict to a JSON file for potential later use
    write_geojson_sections(args.out_dir / f"{args.out_prefix}_sections.geojson", [
        {"id": i, "start": section["start"][i], "end": section["end"][i]} for i in range(len(section["start"]))
    ], section["total_area"], stations, args)
    cells_proj = section_geopandas(section, args)
    # get coastline shapefile polygon for potential clipping of section polygons to the domain
    if args.coastline_shapefile is not None and args.coastline_shapefile.exists():
        all_land, islands_polygons = build_coastline_polygons_from_shapefile(args.coastline_shapefile) 
        from shapely.validation import make_valid

        coast = make_valid(domain)
        # Also worth fixing the islands individually before the difference
        islands = make_valid(unary_union(islands_polygons.geometry.values))
        sea_masked = make_valid(coast.difference(islands))
        # Fix the sea polygon
        sea_masked = make_valid(sea_masked)
        sea_masked = sea_masked.buffer(0)
        from shapely.validation import explain_validity

        print(explain_validity(coast))   # before difference
        print(explain_validity(islands))       # island geometries
        cells_proj["geometry"] = cells_proj.geometry.apply(make_valid)
    # use domain polygon with islands added back in for clipping if coastline shapefile provided, otherwise just use domain polygon
        # Get all island polygons as a single geometry
        sea_gdf = gpd.GeoDataFrame(geometry=[sea_masked], crs=args.utm_epsg)
        print(explain_validity(sea_masked))    # after difference
        cells_proj["geometry"] = cells_proj.geometry.intersection(sea_masked)
        cells_proj = cells_proj.to_crs("EPSG:32633")
        cells_proj["wet_area_m2"] = cells_proj.geometry.area
        cells_proj = cells_proj[cells_proj["wet_area_m2"] > 0]
        # write the clipped section polygons to a new GeoJSON for sanity check
        cells_proj.to_file(args.out_dir / f"{args.out_prefix}_sections_clipped.geojson", driver="GeoJSON")
    if args.coastline_shapefile is not None and args.coastline_shapefile.exists():
        coast_poly = coastline_polygon_from_shapefile(args.coastline_shapefile)
        if coast_poly is not None and not coast_poly.is_empty:
            if mesh_transformer is not None:
                coast_poly = shp_transform(lambda xx, yy, zz=None: mesh_transformer.transform(xx, yy), coast_poly)
            clipped = domain.intersection(coast_poly)
            if not clipped.is_empty:
                if clipped.geom_type == "Polygon":
                    domain = clipped
                else:
                    parts = [g for g in clipped.geoms if g.geom_type == "Polygon"]
                    if parts:
                        domain = max(parts, key=lambda g: g.area)

    minx, miny, maxx, maxy = domain.bounds
    radius = 2.0 * max(maxx - minx, maxy - miny)
    sec_polys = section_polygons(sample_points, tangents, domain, radius)

    node_depth = infer_depth_positive_down(mesh.depth, mode=args.depth_mode)
    tri_depth = np.mean(node_depth[mesh.triangles], axis=1)

    tri_polys, tri_area = triangle_polygons(mesh.x, mesh.y, mesh.triangles)
    if args.integration_method == "intersection":
        rows = integrate_sections(sec_polys, tri_polys, tri_depth)
    else:
        rows = integrate_sections_centroid(
            mesh.triangles,
            mesh.x,
            mesh.y,
            tri_area,
            tri_depth,
            sample_points,
            tangents,
        )

    csv_path = args.out_dir / f"{args.out_prefix}.csv"
    geojson_path = args.out_dir / f"{args.out_prefix}.geojson"
    write_csv(csv_path, rows, stations)
    write_geojson(geojson_path, sec_polys, rows, stations)

    total_area_sections = float(sum(r["section_area_m2"] for r in rows))
    total_volume_sections = float(sum(r["section_volume_m3"] for r in rows))

    print("Done")
    print(f"Projection: {proj_info}")
    print(f"Sections: {len(rows)}")
    print(f"Triangle area sum (m2): {float(np.sum(tri_area)):.3f}")
    print(f"Integrated section area (m2): {total_area_sections:.3f}")
    print(f"Integrated section volume (m3): {total_volume_sections:.3f}")
    print(f"Integration method: {args.integration_method}")
    print(f"CSV: {csv_path}")
    print(f"GeoJSON: {geojson_path}")


if __name__ == "__main__":
    main()
