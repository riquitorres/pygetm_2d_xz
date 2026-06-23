"""Plot FVCOM mesh with pickle-defined transect overlaid.

This script reuses mesh/pickle helpers from estimate_channel_section_volumes.py.
It creates a static PNG with:
- mesh triangulation
- transect line through ordered channel nodes
- transect nodes as highlighted points
"""

from __future__ import annotations
import pickle
import getpass
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import estimate_channel_section_volumes as ecs
import PyFVCOM as pf
from itertools import islice

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot mesh and channel transect")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(f"/home/{getpass.getuser()}/OneDrive/Projects/Tamar_pyGETM/tamar_v0/tamar_v0.2dm"),
        help="Path to mesh file (.2dm)",
    )
    parser.add_argument(
        "--transect-pickle",
        type=Path,
        default=Path(f"/home/{getpass.getuser()}/OneDrive/Projects/Tamar_pyGETM/transect_grid/tamar_transect_nodes.pk"),
        help="Pickle with ordered channel node IDs",
    )
    parser.add_argument(
        "--channel-key",
        default="channel_nodes",
        help="Pickle key for channel node sequence",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path(f"/home/{getpass.getuser()}/OneDrive/Projects/Tamar_pyGETM/transect_grid/section_volume_outputs/mesh_transect_overlay.png"),
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
    return parser

def _extract_transect_nodes_from_mesh(mesh: ecs.Mesh, bounding_box: tuple) -> dict:
    """Extract transect nodes from mesh within bounding box."""
    trinodes, nodes, X, Y, Z = mesh.triangles, mesh.nodes, mesh.x, mesh.y, mesh.depth
    Zc = pf.grid.nodes2elems(Z, trinodes)
    Xc = pf.grid.nodes2elems(X, trinodes)
    Yc = pf.grid.nodes2elems(Y, trinodes)
    # Get lon and lat
    lon, lat = pf.coordinate.lonlat_from_utm(X, Y, '30N', ellipsoid='WGS84', datum='WGS84', parallel=False)
    lonc, latc = pf.coordinate.lonlat_from_utm(Xc, Yc, '30N', ellipsoid='WGS84', datum='WGS84', parallel=False)
    dist_res = 0.05  # km
     # make a graph object
    tamar_graph = pf.grid.GraphFVCOMdepth(mesh.mesh_file, depth_weight=200, depth_power=8, bounding_box=bounding_box)
    # give start and end points of transect. These are fixed so choose wisely!
    # start with upper tamar and do a second one from channel to outer sound. Doing it one go wiggles too much after the breakwater
    # start_pt = [413389, 5594597]
    start_pt = [414488,  5596963]
    end_pt = [417342, 5576941]
    channel_nodes1 = tamar_graph.get_channel_between_points(start_pt, end_pt) - 1  # Python indexes not FVCOM node numbers
    # channel_nodes1 = pf.grid.line_sample(X, Y, np.array([start_pt, end_pt]))
    # do next segment as a line instead
    start_pt = [417342, 5576941]
    end_pt = [416516, 5574450]
    channel_nodes2 = tamar_graph.get_channel_between_points(start_pt, end_pt) - 1  # Python indexes not FVCOM node numbers
    # channel_nodes2, _ = pf.grid.line_sample(X, Y, np.array([start_pt, end_pt]))
    # combine both lines
    channel_nodes = np.hstack([channel_nodes1, channel_nodes2])
    # sub sample the transect every 100m along the transect
    #
    dS = [pf.grid.haversine_distance((lon[i], lat[i]), (lon[i + 1], lat[i + 1])) for i in channel_nodes[:-1]]
    nodes_in_line = [channel_nodes[0]]
    it = iter(enumerate(channel_nodes))
    for zero_idx, idx in it:
        lapsed_dist = np.cumsum([0] + dS[zero_idx:])
        if zero_idx == len(dS):
            break
        # find first index farther than 100m
        idx2 = np.where(lapsed_dist > dist_res)[0][0]
        # advance iterable by idx2 amounts
        next(islice(it, idx2, idx2), None)
        # zero_idx += idx2
        nodes_in_line.append(idx)
    # plot the bathymetry and lay on top the nodes selected...
    # plt.pyplot.scatter(X, Y, c=Z, cmap='terrain')
    # plt.pyplot.scatter(X[channel_nodes], Y[channel_nodes], c=Z[channel_nodes], cmap='terrain', edgecolor='white')
    # plt.pyplot.scatter(X[nodes_in_line], Y[nodes_in_line], c=Z[nodes_in_line], cmap='terrain', edgecolor='red')
    # plt.pyplot.axis('square')

    channel_nodes = nodes_in_line
    # find the elements associated with the node transect
    element_line, elem_distance = pf.grid.element_sample(lonc, latc, np.array([lon[channel_nodes], lat[channel_nodes]]).T)
    # plt.pyplot.plot(elem_distance, -Zc[element_line], marker='.')
    # subsample elements transect to 100m or less
    dS = [pf.grid.haversine_distance((lonc[i], latc[i]), (lonc[i + 1], latc[i + 1])) for i in element_line[:-1]]
    elem_in_line = [element_line[0]]
    it = iter(enumerate(element_line))
    for zero_idx, idx in it:
        lapsed_dist = np.cumsum([0] + dS[zero_idx:])
        if zero_idx == len(dS) or not np.any(lapsed_dist > .1):
            break
        # find first index farther than 100m
        idx2 = np.where(lapsed_dist > dist_res)[0][0]
        # advance iterable by idx2 amounts
        next(islice(it, idx2, idx2), None)
        # zero_idx += idx2
        elem_in_line.append(idx)

    element_line = elem_in_line
    distance = np.cumsum(
        [0] + [np.hypot(X[i + 1] - X[i], Y[i + 1] - Y[i]) for i in channel_nodes[:-1]])
    distance_km = np.cumsum(
        [0] + [pf.grid.haversine_distance((lon[i], lat[i]), (lon[i + 1], lat[i + 1])) for i in channel_nodes[:-1]])
    distance_elem = np.cumsum(
        [0] + [pf.grid.haversine_distance((lonc[i], latc[i]), (lonc[i + 1], latc[i + 1])) for i in element_line[:-1]])

    return {
        "channel_nodes": channel_nodes,
        "element_line": element_line,
        "distance_km": distance_km,
        "distance_elem": distance_elem,
        "distance": distance,
    }
def plot_mesh_and_transect(
    mesh: ecs.Mesh,
    channel_nodes: np.ndarray,
    out_png: Path,
    mesh_linewidth: float = 0.12,
    mesh_alpha: float = 0.28,
    node_size: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a mesh+transect overlay plot and save as PNG."""

    centerline = ecs.centerline_from_nodes(mesh, channel_nodes)

    node_to_idx = {int(nid): i for i, nid in enumerate(mesh.nodes.tolist())}
    transect_idx = np.asarray([node_to_idx[int(n)] for n in channel_nodes], dtype=int)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=170)

    ax.triplot(
        mesh.x,
        mesh.y,
        mesh.triangles,
        color="#0f172a",
        linewidth=mesh_linewidth,
        alpha=mesh_alpha,
    )

    ax.plot(
        centerline[:, 0],
        centerline[:, 1],
        color="#ef4444",
        linewidth=1.8,
        alpha=0.95,
        label="Transect line",
        zorder=8,
    )

    ax.scatter(
        mesh.x[transect_idx],
        mesh.y[transect_idx],
        s=node_size,
        c="#1d4ed8",
        alpha=0.9,
        label="Transect nodes",
        zorder=6,
    )

    ax.scatter(
        [centerline[0, 0], centerline[-1, 0]],
        [centerline[0, 1], centerline[-1, 1]],
        s=10,
        c=["#16a34a", "#a21caf"],
        zorder=7,
        label="Start/End",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("FVCOM Mesh with Pickle Transect Overlay")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    # limit axes to transect bounding box with some padding
    x_min, x_max = np.min(centerline[:, 0]), np.max(centerline[:, 0])
    y_min, y_max = np.min(centerline[:, 1]), np.max(centerline[:, 1])
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    # return the figure and axis in case caller wants to do more custom plotting before saving
    return fig, ax
    # plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    # grid_file = '/data/sthenno1/backup/mbe/Code/fvcom-projects/immerse/preproc/common/tamar_v2_grd.dat'
    # if mesh ends in .dat, assume it's a FVCOM mesh and read with PFVCOM tools. Otherwise, try to read with Mesh class.
    mesh = ecs.load_mesh(args.mesh)
    if args.mesh.suffix == ".dat":
        bounding_box = [[406648.5, 421944.5], [5571183.0, 5599817.8]]
        output = _extract_transect_nodes_from_mesh(mesh, bounding_box)
        channel_nodes = output["channel_nodes"]
    else:
        channel_nodes = ecs.load_channel_nodes(args.transect_pickle, key=args.channel_key)

    plot_mesh_and_transect(
        mesh=mesh,
        channel_nodes=channel_nodes,
        out_png=args.out_png,
        mesh_linewidth=args.mesh_linewidth,
        mesh_alpha=args.mesh_alpha,
        node_size=args.node_size,
    )
    # save transect nodes to pickle for later use
    if args.transect_pickle:
        with args.transect_pickle.open("wb") as f:
            pickle.dump({args.channel_key: channel_nodes}, f)
    print(f"Saved: {args.out_png}")
    print(f"Mesh nodes: {mesh.nodes.size}, elements: {mesh.triangles.shape[0]}")
    print(f"Transect nodes: {len(channel_nodes)}")


if __name__ == "__main__":
    main()
