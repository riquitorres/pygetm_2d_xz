# Tamar Channel Section Volume Workflow

## Goal
Estimate water volume in **non-overlapping** estuarine sections every 100 m along a centreline defined by node IDs.

## Data Used
- Centreline node pickle: `/home/ricardo/OneDrive/Projects/Tamar_pyGETM/transect_grid/tamar_transect_nodes.pk`
- Mesh: `/home/ricardo/OneDrive/Projects/Tamar_pyGETM/tamar_v0/tamar_v0.2dm`
- Script: `/home/ricardo/OneDrive/Projects/Tamar_pyGETM/scripts/estimate_channel_section_volumes.py`

## Method (Implemented)
1. Read mesh nodes, depths, and triangle connectivity.
2. Read `channel_nodes` from pickle and treat as an ordered centreline.
3. Convert mesh to projected metres (auto UTM if input appears lon/lat).
4. Resample centreline every 100 m.
5. At each sampled centreline point, compute local tangent.
6. Build divider lines perpendicular to the local tangent.
7. Build section polygons between adjacent divider lines, clipped to the wet mesh domain.
8. For each section polygon, intersect mesh triangles and integrate:
   - section area: sum of intersection areas
   - section volume: sum(intersection area * local triangle depth)

## Non-overlap Guarantee
Sections are generated from adjacent half-plane constraints derived from consecutive perpendicular dividers. This partitions the domain into ordered strips, so sections do not overlap.

## Depth Convention
The script supports:
- `--depth-mode auto` (default): flips sign if median depth is negative
- `--depth-mode positive`: force positive-down via absolute value
- `--depth-mode negative`: force negative-down (kept mainly for diagnostics)

## Run Command
```bash
/home/ricardo/OneDrive/Projects/Tamar_pyGETM/.venv/bin/python \
  /home/ricardo/OneDrive/Projects/Tamar_pyGETM/scripts/estimate_channel_section_volumes.py \
  --mesh /home/ricardo/OneDrive/Projects/Tamar_pyGETM/tamar_v0/tamar_v0.2dm \
  --transect-pickle /home/ricardo/OneDrive/Projects/Tamar_pyGETM/transect_grid/tamar_transect_nodes.pk \
  --spacing 100 \
  --depth-mode auto \
  --out-dir /home/ricardo/OneDrive/Projects/Tamar_pyGETM/transect_grid/section_volume_outputs \
  --out-prefix tamar_sections_100m
```

## Optional Coastline Clipping
If you have a coastline shapefile in lon/lat, add:
```bash
--coastline-shapefile /absolute/path/to/coastline.shp
```
The script projects coastline geometry to the mesh projection when needed.

## Outputs
Written to `--out-dir`:
- `<prefix>.csv`: section metrics table
- `<prefix>.geojson`: section polygons + attributes

CSV columns:
- `section_id`
- `s_start_m`
- `s_end_m`
- `s_mid_m`
- `section_area_m2`
- `section_volume_m3`
- `n_triangles_touched`

## Practical Checks
1. Plot the GeoJSON sections and verify divider orientation.
2. Confirm area continuity along the channel (no gaps/spikes unless geomorphically expected).
3. Compare volumes for `--depth-mode auto` vs `--depth-mode positive` to ensure sign handling is correct for your mesh.
4. If sections include unintended side embayments, provide coastline shapefile clipping.

## Notes
- The script uses `pyfvcom2` mesh readers when available; for `.2dm` it can fall back to an internal parser.
- Because section integration is polygon intersection based, this is more accurate than simple centroid assignment.

## A different approach added later

I use pyfvcom to define the center line and export those nodes as a pickle file and then used the scripts here to select cross sections and estimate an average cross-section area and averaged depth. the plot_transect.py does the channel nodes extraction from the mesh. 

I needed to install geopandas in the pyfvcom conda environment so that I can get the area between two sections 

I had to manually adjust a lot of the sections in qgis by editing the geojson file loaded on qgis and also make the coastline manually (or edit the extended tamar shapefile) to find the intersections as valid polygons (non-overlapping and removing the additional coastline not represented in the tamar fvcom mesh)
Sections need to extend beyond the coastline and the order of their end points matters because of the way i build the polygons to find the intersection! sometimes they are back to front ... 
tomorrow i finish it

I also had to manually reorder the sections id because ordering by distance of the start or end points wasn't working due to the curves in the domain... 

i probably should restrict the heomaz basin area manually as the calculated depth is too shallow to make sense... 