# run script to setup and run a 2D slice pyGETM model of the Tamar estuary

import argparse
import csv
import datetime
from pathlib import Path
import cftime
import numpy as np
import pygetm
import matplotlib.pyplot as plt
from matplotlib import colors
from pygetm.input import tpxo
import matplotlib.animation as animation
import pandas as pd
import geopandas as gpd
import xarray as xr
import cmocean
import pyproj
import getpass
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
#python -m pygetm.input.era5 min_lon max_lon min_lat max_lat start_year end_year --source arco -v t2m --no_default_variables
#python -m pygetm.input.era5 -4.25 -4 50.2 50.5 1990 2025 --source arco -v t2m -v ssr -v str  -v ssrd -v strd   -v tp -v u10 -v v10 -v sp   -v d2m -v e -v tcc --no_default_variables
#python -m pygetm.input.era5 -55 30 35 70 1990 2025 --source arco -v ssrd -v strd  --no_default_variables
#python -m pygetm.input.era5 -55 30 35 70 1990 2025 --source arco -v tp -v u10 -v v10  --no_default_variables
#python -m pygetm.input.era5 -55 30 35 70 1990 2025 --source arco -v sp --no_default_variables
#python -m pygetm.input.era5 -55 30 35 70 1990 2025 --source arco -v d2m --no_default_variables
#python -m pygetm.input.era5 -55 30 35 70 1990 2025 --source arco -v e --no_default_variables
#python -m pygetm.input.era5 -55 30 35 70 2024 2024 --source arco -v tcc --no_default_variables
#python -m pygetm.input.igotm -4.25 50.3 2000 era5_2000.nc
DEFAULTS = {
    "estuary_name": "tamar",
    "tamar_mesh": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data/tamar_wet_cells_with_depths.geojson"),
    "nodes_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data/tamar_nodes_dict_manual.csv"),
    "depth_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data/tamar_depth_dict.csv"),
    "out_png": "tamar_2Dslice",
    "start_date": "2023-01-01",
    "end_date": "2023-01-31",
    "output_data_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/output/"),
    "output_fig_dir": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/figures/"),
    "river_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data/EMORID_1993_2024_conc_nemo_TALK_DIC.nc"),
    "cmems_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data/tamar_boundary_conditions_2023-01-01_2023-01-31.nc"),
    "era5_file": Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data/era5_2023.nc"),
    "TPXO9_dir": Path(f"/data/TPXO9/"),
    "fabm_file": None,
    }
river_config = {
    # EMORID default naming (used in most of the original files)
    "emorid": {
        "index": "site",  # integer index
        "name": "site_name",  # name
        "lat": "lat",  # latitude column / coordinate name
        "lon": "lon",  # longitude column / coordinate name
        "Q": "Q",  # discharge (flow) variable
        "Qmean": "Q_mean",  # discharge (flow) variable
        "N3_n": "NO3", # nitrate load in N/day
        "N4_n": "NH4", # ammonium load in N/day
        "N1_p": "PO4", # phosphate load in P/day
        "N5_s": "Si", # silicate load in Si/day
        "O3_TA": "TALK", # total alkalinity load in mol/day?
        "O3_c": "DIC", # dissolved inorganic carbon load in mol/day?
    },
    # TODO: fix the names for the JRC rivers
    "jrc": {
        "lat": "Latitude",
        "lon": "Longitude",
        "Q": "discharge",
        "Qmean": "QQQQan",
    },
}

class MySimulation(pygetm.Simulation):
    def __init__(self, *args, initial, **kwargs):
        self.initial = initial
        super().__init__(*args, **kwargs)

    def _update_forcing_and_diagnostics(self, macro_active: bool):
        if self.initial:
            # Initial ramp parameters: gradually increase boundary conditions over time
            # to avoid shock and instabilities at simulation start
            RAMP_DURATION_TIMESTEPS = 302000  # ~8-10 hours at 60s timestep
            RAMP_STEEPNESS = 5.0  # Controls sigmoid curve steepness (higher = steeper)

            alpha = RAMP_STEEPNESS
            t = self.istep / RAMP_DURATION_TIMESTEPS
            ramp = 1.0 / (1.0 + np.exp(-alpha * (t - 0.5)))  # Sigmoidal ramp function
            # print every 10000 time steps
            # if self.istep % 10000 == 0:
            #     print(
            #         "Applying ramping to open boundary conditions with ramp factor:",
            #         ramp,
            #     )
            self.open_boundaries.z.all_values *= ramp
            self.open_boundaries.u.all_values *= ramp
            self.open_boundaries.v.all_values *= ramp
            self.airsea.taux.all_values *= ramp
            self.airsea.tauy.all_values *= ramp
            self.airsea.shf.all_values *= ramp
            self.airsea.sp.all_values = 101325.0 * ramp + self.airsea.sp.all_values * (1 - ramp)
            self.airsea.pe.all_values *= ramp
            self.rivers.flow *= ramp
        super()._update_forcing_and_diagnostics(macro_active)


def animate_transect(
    x,
    z,
    var,
    time,
    varname,
    units="",
    cmap=None,
    vmin=None,
    vmax=None,
    output=None,
    fps=20,
):

    if cmap is None:
        cmap = cmocean.cm.balance if np.nanmin(var) < 0 else cmocean.cm.haline

    fig, ax = plt.subplots(figsize=(20, 10))

    cbar = None

    # Fixed ranges
    if vmin is None:
        vmin = np.nanmin(var)

    if vmax is None:
        vmax = np.nanmax(var)

    ymin = np.nanmin(z)
    ymax = np.nanmax(z)

    def animate(i):

        nonlocal cbar

        ax.clear()

        pcm = ax.pcolormesh(
            x,
            z[i],
            var[i],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax, 
        ec=(1, 1, 1, 0.2),
        lw=0.03,
        )
        # plot the levels as dashed lines
        # dummy = np.zeros((z[i].shape[0], z[i].shape[1]))
        # pc = ax.pcolormesh(
        #     x,
        #     z[i],
        #     dummy,
        #     ec="k",
        #     lw=0.1,
        #     cmap=colors.ListedColormap(["white"]),
        # )

        # ax.plot(x, z[i], color="k", linestyle="--", linewidth=0.5)
        if cbar is None:
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(f"{varname} ({units})" if units else varname)

        t = pd.to_datetime(time[i])

        ax.set_title(
            f"{varname}\n{t:%Y-%m-%d %H:%M}"
        )

        ax.set_xlabel("Distance from Top of Estuary (km)")
        ax.set_ylabel("Height above bed (m)")
        ax.set_ylim(ymin, ymax)

        return ()

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(time),
        interval=50,
        blit=False
    )

    # plt.show()

    if output is not None:
        ani.save(
            output,
            writer="ffmpeg",
            fps=fps,
            dpi=100
        )

    return ani
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 2D model estuary simulations")
    parser.add_argument(
        "--estuary-name",
        type=str,
        default=DEFAULTS["estuary_name"],
        help="Name of the estuary (used for file naming)",
    )
    parser.add_argument(
        "--domain-file",
        type=Path,
        default=DEFAULTS["tamar_mesh"],
        help="Path to mesh file (.geojson or .dat) defining the estuary domain",
    )
    # process or load pre-configured section nodes
    parser.add_argument(
        "--nodes-file",
        type=Path,
        default=DEFAULTS["nodes_file"],
        help="Path to nodes file (CSV or GeoJSON) containing transect sections",
    )
    parser.add_argument(
        "--depth-file",
        type=Path,
        default=DEFAULTS["depth_file"],
        help="File with depths for each section (CSV)",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=DEFAULTS["out_png"],
        help="Output PNG file prefixes",
    )
    parser.add_argument(
        "--plot-mesh",
        action="store_true",
        help="Plot the domain mesh ",
    )
    # start and end date
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULTS["start_date"],
        help="Start date for simulation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULTS["end_date"],
        help="End date for simulation (YYYY-MM-DD)",
    )
    # data_dir 
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULTS["output_data_dir"],
        help="Directory to save simulation data",
    )
    # fig dir
    parser.add_argument(
        "--output-fig-dir",
        type=Path,
        default=DEFAULTS["output_fig_dir"],
        help="Directory to save figures",
    )
    # cmems_file
    parser.add_argument(
        "--cmems-file",
        type=str,
        default=DEFAULTS.get("cmems_file", None),
        help="Path to CMEMS data file (NetCDF)",
    )
    # river file
    parser.add_argument(
        "--river-file",
        type=Path,
        default=DEFAULTS.get("river_file", None),
        help="Path to river data file (NetCDF)",
    )
    # atmospheric forcing file
    parser.add_argument(
        "--era5-file",
        type=Path,
        default=DEFAULTS.get("era5_file", None),
        help="Path to ERA5 atmospheric forcing file (NetCDF)",
    )
    # TPXO dir
    parser.add_argument(
        "--TPXO9-dir",
        type=Path,
        default=DEFAULTS.get("TPXO9_dir", None),
        help="Path to TPXO9 tidal data directory",
    )
    # fabm_file 
    parser.add_argument(
        "--fabm-file",
        type=Path,
        default=DEFAULTS.get("fabm_file", None),
        help="Path to FABM configuration file (YAML)",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    # create output directories if they don't exist
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_fig_dir.mkdir(parents=True, exist_ok=True)

    # convert start and end date to datetime
    starttime = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    endtime = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        
    # set up the model
    # domain definition file 
    boundaries = True
    rivers = True
    river_file = args.river_file
    river_x=414381
    river_y=5596775
    check_finite = True
    show_plots = False
    run_type = pygetm.RunType.BAROCLINIC
    output_prefix = "adaptive"
    # nodes_file = Path("../transect_grid/"+experiment+"nodes_dict_manual.csv")
    nodes_file = args.nodes_file
    depth_file = args.depth_file
    depth_col = "real_mean_depth_m"
    domain_file = args.domain_file
    era_5_file = args.era5_file
    cmems_file = args.cmems_file
    tpxo9_dir = args.TPXO9_dir
    # era_5_file = None
    timestep=5
    isplit=10
    nlevels = 30
    ddu = 2
    ddl = 1

    plot_period = (starttime, endtime)
    plot_interval = datetime.timedelta(hours=1)
    output_interval2D = datetime.timedelta(hours=1)
    output_interval3D = datetime.timedelta(hours=1)
    # load geojson of the domain to get the distance along the main channel for the 2D slice


    gisdomain = gpd.read_file(domain_file)
    # convert river x and y from lat lon to the same projection as the domain
    crs = gisdomain.crs
    transformer = pyproj.Transformer.from_crs("EPSG:32630",  crs,  always_xy=True)
    river_lon, river_lat = transformer.transform(river_x, river_y)  

    # load nodes_dict and depth_dict from csv files
    nodes = []
    latlon = []
    with open(nodes_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            node_id = int(row["node_id"])
            nodes.append([float(row["x"]), float(row["y"])])
            latlon.append((float(row["lon"]), float(row["lat"])))

    depths = []
    with open(depth_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            section_id = int(row["section_id"])
            depths.append([float(row[depth_col]), float(row["element_area_m2"])])

    x_nodes = np.array([np.array(nodes)[::2, 0], np.array(nodes)[1::2, 0]])
    y_nodes = np.array([np.array(nodes)[::2, 1], np.array(nodes)[1::2, 1]])

    central_lon = np.mean([lon for lon, lat in latlon])
    central_lat = np.mean([lat for lon, lat in latlon])
    domain = pygetm.domain.create_cartesian(x=x_nodes, y=y_nodes, H=np.array(depths)[:, 0], central_lon=central_lon, central_lat=central_lat, interfaces=True, z0=0.001,)
    # replace area in domain with the real intersection area from the depth file
    domain._area[1,1::2] = np.array(depths)[1:, 1]# don't know why I can't use the first one or why i have one extra area value.
    if args.plot_mesh:
        fig = domain.plot(domain.rotation, show_mesh=True).savefig(Path(args.output_fig_dir) / "domain_rotation_with_mesh.png", dpi=300)
        fig = domain.plot(show_mesh=True).savefig(Path(args.output_fig_dir) / "domain_bathy_with_mesh.png", dpi=300)
    vertical_coordinates = pygetm.vertical_coordinates.Adaptive(
                nz=nlevels,
                ddl=ddl,
                ddu=ddu,
                Dgamma=1,
                gamma_surf=ddu >= ddl,
                hmin=1.0,
                vfilter=0.10,
                hfilter=0.10,
                cNN=1,
                drho=4,
                timescale=2.0 * 3600.0,
            )

    # add river by location 
    if rivers:
        domain.rivers.add_by_location("tamar", x=river_x, y=river_y, coordinate_type=pygetm.constants.CoordinateType.XY,)# Q=100, T=10)
    # domain.rivers.add_by_index("tamar_index", i=0, j=0)
    # add open boundaries
    if boundaries:
        domain.open_boundaries.add_right_boundary("south", i=162, jstart=0, jstop=1, type_2d=pygetm.constants.FLATHER_ELEV, type_3d=0)
    # fig = domain.plot(show_mesh=True).savefig("domain_bathy_with_mesh.png", dpi=300)
    if era_5_file:
        humidity_measure = pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE
        airsea = pygetm.airsea.FluxesFromMeteo(
            humidity_measure=humidity_measure, calculate_evaporation=True
        )
    else:
        airsea = pygetm.airsea.Fluxes()

    sim = MySimulation(
            domain,
            runtype=run_type,
        vertical_coordinates=vertical_coordinates,
        airsea=airsea,
        gotm=Path(".") / "gotm.yaml",
        Dcrit=0.5,
        Dmin=0.1,
            initial=True, # to apply ramping to open boundary conditions at the start of the simulation, if using restart file set to False
        fabm=args.fabm_file
        )
    # sim = pygetm.Simulation(
    #     domain,
    #     runtype=run_type,
    #     # runtype=pygetm.BAROTROPIC_3D,
    #     vertical_coordinates=pygetm.vertical_coordinates.Sigma(nlevels),
    #     airsea=airsea,
    #     gotm=Path(".") / "gotm.yaml",
    #     Dcrit=0.5,
    #     Dmin=0.1,
    # )
    for river in sim.rivers.values():
            output_river_file = river_file.parent / (args.estuary_name + "_river.nc")
            if output_river_file.exists():
                tamar = xr.open_dataset(output_river_file, decode_times=time_coder)
                # On reload
                coord_names = tamar.attrs.pop("extra_coords").split(",")
                tamar = tamar.set_coords(coord_names)
                # set time as a coordinate if it is not already
                if "time" not in tamar.coords:
                    tamar = tamar.assign_coords(time=("time", tamar.time.values))
                river.flow.set(tamar.Q)
            else:
                # Tamar or lat lon river 
                river_xr = xr.open_dataset(river_file)
                riv_lon = river_xr.lon.values
                riv_lat = river_xr.lat.values
                dist = np.sqrt((riv_lon - river_lon)**2 + (riv_lat - river_lat)**2)
                closest_index = np.argmin(dist)
                tamar = river_xr.isel(site=closest_index)
                # convert time to cftime if it is not already in that format
                ts = (tamar.time.values.astype("datetime64[s]") - np.datetime64("1970-01-01", "s")) / np.timedelta64(1, "s")
                cftime_dates = cftime.num2date(ts, "seconds since 1970-01-01", calendar="gregorian")
                # Keep time as an explicit coordinate so it is serialized as a coordinate.
                tamar = tamar.assign_coords(time=("time", cftime_dates))
                # copy attributes from the original dataset to the new dataset
                tamar.attrs = river_xr.attrs
                # save tamar file if it doesn't already exist
                coord_names = [k for k in tamar.coords if k not in tamar.dims]
                if not output_river_file.exists():
                    # Materialize non-dimension coordinates as variables in the file.
                    # This ensures scalar coords (e.g. lon/lat after isel) are preserved.
                    tamar.reset_coords().assign_attrs(extra_coords=",".join(coord_names)).to_netcdf(output_river_file)
                river.flow.set(tamar.Q)
            # river.flow.set(10)
            if sim.runtype == pygetm.RunType.BAROCLINIC:
                river["salt"].set(0.5)
                river["temp"].set(10.0)
            if sim.fabm:
                # set river fabm tracer variables 
                vars2process = ["N3_n", "N4_n", "N1_p", "N5_s"] # , "O3_TA", "O3_c"]
                for var in vars2process:
                    if river_config["emorid"][var] in tamar.data_vars:
                        river[var].set(tamar[river_config["emorid"][var]])
                    else:
                        sim.logger.warning(f"Variable {var} not found in river data file")
                        # river[var].set(0.0)
    # setup airsea to constant values too
    if type(airsea) == pygetm.airsea.FluxesFromMeteo:
        if era_5_file:
            era5_xr = xr.open_dataset(era_5_file, decode_times=time_coder)
            if era5_xr['t2m'].units == "degree_Kelvin" or "kelvin" in era5_xr['t2m'].units.lower():
                era5_xr['t2m'] = era5_xr['t2m'] - 273.15
            sim.airsea.t2m.set(era5_xr["t2m"])
            for river in sim.rivers.values():
                river["temp"].set(era5_xr["t2m"])

            # sim.airsea.u10.set(era5_xr["u10"])
            # sim.airsea.v10.set(era5_xr["v10"])
            sim.airsea.sp.set(era5_xr["sp"])
            if humidity_measure == pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE:
                if era5_xr['d2m'].units == "degree_Kelvin" or "kelvin" in era5_xr['d2m'].units.lower():
                    era5_xr['d2m'] = era5_xr['d2m'] - 273.15
                sim.airsea.d2m.set(era5_xr["d2m"])
            elif humidity_measure == pygetm.HumidityMeasure.RELATIVE_HUMIDITY:
                sim.airsea.rh.set(era5_xr["rh"])
            sim.airsea.tcc.set(era5_xr["tcc"])
            sim.airsea.tp.set(era5_xr["tp"] / 3600.0)
            sim.airsea.u10.set(0.0)
            sim.airsea.v10.set(0.0)

        else:
            sim.airsea.t2m.set(15.0)
            sim.airsea.u10.set(0.0)
            sim.airsea.v10.set(0.0)
            sim.airsea.sp.set(101325.0)
            if humidity_measure == pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE:
                sim.airsea.d2m.set(14.0)    
            elif humidity_measure == pygetm.HumidityMeasure.RELATIVE_HUMIDITY:
                sim.airsea.rh.set(80.0)
    else:
        sim.airsea.taux.set(0.0)
        sim.airsea.tauy.set(0.0)
        sim.airsea.shf.set(100.0)
        sim.airsea.sp.set(101325.0)
        sim.airsea.pe.set(0)
        sim.airsea.swr.set(300.0)
    if sim.runtype == pygetm.RunType.BAROCLINIC:
        sim.radiation.set_jerlov_type(pygetm.Jerlov.Type_II)
        sim.radiation.A.set(0.7)
        sim.radiation.kc1.set(0.54)  # 1/g1 in gotm
        sim.radiation.kc2.set(3.23)
        sim.temp.set(11.6)
        sim.salt.set(30.2)
    else:
        if type(airsea) == pygetm.airsea.FluxesFromMeteo:
            sim.sst.set(sim.airsea.t2m)
        else:
            sim.sst.set(15.0)

    if boundaries:
        sim.logger.info("Getting 2D boundary data from TPX9")
        bdy_lon = sim.open_boundaries.lon
        bdy_lat = sim.open_boundaries.lat
        # sim.open_boundaries.z.set(0.0)
        # sim.open_boundaries.u.set(0.0)
        # sim.open_boundaries.v.set(0.0)
        sim.open_boundaries.z.set(tpxo.get(bdy_lon, bdy_lat, root=tpxo9_dir, variable="h"))
        sim.open_boundaries.u.set(
            tpxo.get(bdy_lon, bdy_lat, variable="u", root=tpxo9_dir)
        )
        sim.open_boundaries.v.set(
            tpxo.get(bdy_lon, bdy_lat, variable="v", root=tpxo9_dir)
        )
        if sim.runtype == pygetm.RunType.BAROCLINIC:
            sim.open_boundaries.sponge.tmrlx = True
            cmems = xr.open_dataset(cmems_file, decode_times=time_coder)
            # temp = xr.open_dataset(Path(cmems_file.format(var="temp", nemovar="thetao")), decode_times=time_coder)
            # salt = xr.open_dataset(Path(cmems_file.format(var="sal", nemovar="so")), decode_times=time_coder)
            # # find non-nan grid point closest to the boundary point and set the boundary temp to that value
            # lon_mesh, lat_mesh = np.meshgrid(temp.longitude.values, temp.latitude.values)
            # dist = np.sqrt((lon_mesh - bdy_lon.values)**2 + (lat_mesh - bdy_lat.values)**2).flatten()
            # # find first non-nan value in temp that is closest to the boundary point
            # temp_flat = temp.thetao.values[0,0,...].flatten()
            # valid_dist = np.where(~np.isnan(temp_flat), dist, np.inf)
            # closest_index = np.argmin(valid_dist)   
            # j, i = np.unravel_index(closest_index, temp.thetao.values[0,0,...].shape)
            # # extract and fill any nans with nearest depth value
            # temp_series = temp.isel(longitude=i, latitude=j).thetao.ffill(dim="depth")
            # # temp_seriesB = temp.isel(longitude=i, latitude=j-1).thetao.ffill(dim="depth")
            # # temp_series = xr.concat([temp_seriesA, temp_seriesB], dim="latitude")
            # # # reorder to time, depth latitude, longitude
            # # temp_series = temp_series.transpose("time", "depth", "latitude")
            # # # restore lat and lon coordinates to temp and salt so they can be used in the open boundary conditions
            # # temp = temp.assign_coords(longitude=temp.longitude.values, latitude=temp.latitude.values)
            # # extent temp dimension to include at least 1 for lat and lon
            # temp_series = temp_series.expand_dims({"bdy": [1]}, axis=1)

            # # temp_series = temp_series.expand_dims({"longitude": [temp_series.longitude.values],"latitude": [temp_series.latitude.values]}, axis=(2,3))
            # # temp_series.latitude.attrs = temp.latitude.attrs
            # # temp_series.longitude.attrs = temp.longitude.attrs
            # salt_series = salt.isel(longitude=i, latitude=j).so.ffill(dim="depth")
            # # salt_series.latitude.attrs = salt.latitude.attrs
            # # salt_series.longitude.attrs = salt.longitude.attrs
            # # salt_series = salt_series.expand_dims({"longitude": [salt_series.longitude.values], "latitude": [salt_series.latitude.values] }, axis=(2,3))

            # salt_series = salt_series.expand_dims({"bdy": [1]}, axis=1)
            # # drop latitude and longitude coordinates from temp and salt series so they can be used in the open boundary conditions
            # temp_series = temp_series.drop_vars(["latitude", "longitude"])
            # salt_series = salt_series.drop_vars(["latitude", "longitude"])
            sim.temp.open_boundaries.type= pygetm.SPONGE
            sim.salt.open_boundaries.type= pygetm.SPONGE
            sim.temp.open_boundaries.values.set(cmems['temperature'])#, on_grid = True)
            sim.salt.open_boundaries.values.set(cmems['salinity'])#, on_grid = True)

            # set open_boundary temp and salt to constant values
            # sim.temp.open_boundaries.values.set(13.6)
            # sim.salt.open_boundaries.values.set(35.)
    # set initial zt to 1m everywhere
    # sim.zt.set(1.0)
    if sim.fabm:
        sim.fabm.get_dependency("mole_fraction_of_carbon_dioxide_in_air").set(400.0)
        sim.fabm.get_dependency("mass_concentration_of_silt").set(0.0)
    sim.logger.info("Setting up output")

    output = sim.output_manager.add_netcdf_file(str(Path(args.output_data_dir) / (args.estuary_name +  "_" + output_prefix + "_2d.nc")), interval=output_interval2D, sync_interval=100)
    output.request("zt", "u1", "v1","u10", "v10", "sp", "swr","shf","t2m","d2m", grid=sim.T)
    output = sim.output_manager.add_netcdf_file(str(Path(args.output_data_dir) / (args.estuary_name + "_" + output_prefix + "_3d.nc")), interval=output_interval3D, sync_interval=50)
    output.request("uk", "vk", "tke", "num", "nuh", "eps", grid=sim.T)
    if sim.runtype == pygetm.pygetm.RunType.BAROCLINIC:
        output.request("temp", "salt", grid=sim.T)

    sim.start(
        starttime,
        timestep=timestep,
        split_factor=isplit,
        report=datetime.timedelta(hours=8), # report every 8 hours
        report_totals=datetime.timedelta(days=5), # report totals every 5 days
        # profile="tamar",

    )
    while sim.time < endtime:
        sim.advance(    check_finite=check_finite,)
    sim.finish()



    # plot the results for the 2D slice
    # open the netcdf file and plot the surface elevation at the final time step
    # plot 

    ds = xr.open_dataset(Path(args.output_data_dir) / (args.estuary_name + "_" + output_prefix + "_2d.nc"))
    # resample to min of plot_interval and output interval
    ds = ds.resample(time=min(plot_interval, output_interval2D)).nearest()
    # calculate distance along main channel with reference to the first node in the 2D slice
    x = ds.xt.values
    y = ds.yt.values
    distance = np.cumsum(np.sqrt(np.diff(x, axis=1)**2 + np.diff(y, axis=1)**2), axis=1)/1000
    ds = ds.assign_coords(distance=("distance", np.concatenate(([0],distance.squeeze()))))
    ds['H'] = (('yt', 'xt'), domain.H[1:2,1::2])
    plt.figure(figsize=(8, 6))
    plt.plot(ds.zt["time"], ds.zt[:, 0, 0], label="Surface Elevation river")
    plt.plot(ds.zt["time"], ds.zt[:, 0, -1], label="Surface Elevation open boundary")
    plt.title("Surface Elevation at RIVER NODE")
    plt.xlabel("Time")
    plt.ylabel("Surface Elevation (m)   ")
    plt.savefig(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_surface_elevation.png"), dpi=150)
    plt.close()
    # plot volume against distance along the main channel at the final time step
    # plt.figure(figsize=(8, 6))
    # plt.plot(ds.distance, gisdomain['element_volume_m3'][:-1]/10**9, label="Volume")
    # plt.title("Section volume Along 2D Slice")
    # plt.xlabel("Distance from Top of Estuary (km)")
    # plt.ylabel("Volume (km^3)")
    # plt.savefig(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_section_volume.png"), dpi=300)
    # plt.close()
    # # plot section area against distance along the main channel at the final time step
    # plt.figure(figsize=(8, 6))
    # plt.plot(ds.distance, gisdomain['real_intersection_area_m2'][:-1]/10**6, label="Section Area")
    # plt.title("Section Area Along 2D Slice")
    # plt.xlabel("Distance from Top of Estuary (km)")
    # plt.ylabel("Area (km^2)")
    # plt.savefig(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_section_area.png"), dpi=300)
    # # make an animation of zt against distance from top of estuary, using the nodes in the 2D slice
    # fig, ax = plt.subplots(figsize=(8, 6))
    # line, = ax.plot([], [], label="Surface Elevation")
    # # plot depth as a dashed line on a secondary y-axis
    # ax2 = ax.twinx()
    # ax2.plot(ds.distance, -ds.H[0, :], label="Depth", linestyle="--", color="gray")
    # ax.set_xlim(0, 40)
    # ax.set_ylim(-2.5, 2.5)
    # ax2.set_ylim(-20, 4)
    # time_text = ax.text(
    #     0.02, 0.95, "",
    #     transform=ax.transAxes,
    #     va="top"
    # )
    # ax.set_title("Surface Elevation Along 2D Slice")
    # ax.set_xlabel("Distance from Top of Estuary (km)")
    # ax.set_ylabel("Surface Elevation (m)")
    # ax2.set_ylabel("Depth (m)")
    # ax.legend()
    # def animate(i):
    #     line.set_data(ds.distance, ds.zt[i, 0, :])
    #     time_text.set_text(f"Time: {pd.to_datetime(ds.time[i].values)}")
    #     return line, time_text

    # ani = animation.FuncAnimation(fig, animate, frames=len(ds.time), interval=50, blit=True)
    # if show_plots:
    #     plt.show()
    # ani.save(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_surface_elevation_animation.gif"), writer="imagemagick", dpi=50)
    # del ani
    # fig, ax = plt.subplots(figsize=(8, 6))
    # line, = ax.plot([], [], label="Along axis velocity")
    # line2, = ax.plot([], [], label="Surface elevation")

    # # plot depth as a dashed line on a secondary y-axis
    # ax2 = ax.twinx()
    # ax2.plot(ds.distance, -ds.H[0, :], label="Depth", linestyle="--", color="gray")
    # ax.set_xlim(0, 40)
    # ax.set_ylim(-2., 2.)
    # ax2.set_ylim(-20, 4)
    # time_text = ax.text(
    #     0.02, 0.95, "",
    #     transform=ax.transAxes,
    #     va="top"
    # )
    # ax.set_title("Along Axis Velocity Along 2D Slice")
    # ax.set_xlabel("Distance from Top of Estuary (km)")
    # ax.set_ylabel("Along Axis Velocity (m/s)/Surface elevation (m)")
    # ax2.set_ylabel("Depth (m)")
    # ax.legend()

    # def animateU(i):
    #     line.set_data(ds.distance, ds.u1[i, 0, :])
    #     line2.set_data(ds.distance, ds.zt[i, 0, :])
    #     time_text.set_text(f"Time: {pd.to_datetime(ds.time[i].values)}")
    #     return line, line2, time_text

    # ani = animation.FuncAnimation(fig, animateU, frames=len(ds.time), interval=50, blit=True)
    # if show_plots:
    #     plt.show()
    # ani.save(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_along_axis_velocity_animation.gif"), writer="imagemagick", dpi=50)
    # # delete ani
    # del ani
    # load the 3d netcdf and plot an animation of the along axis velocity at the index closest to distance of 13.8 km from the top of the estuary, which is where the river is located
    ds3d = xr.open_dataset(Path(args.output_data_dir) / (args.estuary_name + "_" + output_prefix + "_3d.nc"))
    # Separate time-dependent coords from the dataset
    time_dep_coords = {k: v for k, v in ds3d.coords.items() 
                    if "time" in v.dims and k != "time"}
    # Resample main dataset
    ds3d_r = ds3d.resample(time=min(plot_interval, output_interval3D)).nearest()
    # Resample each time-dependent coord the same way
    new_coords = {}
    for k, v in time_dep_coords.items():
        new_coords[k] = v.resample(time=min(plot_interval, output_interval3D)).nearest()

    ds3d = ds3d_r.assign_coords(new_coords)

    distance2d = np.tile(ds.distance.values, (ds3d.zct.shape[1], 1))
    # find the index closest to distance of 13.8 km from the top of the estuary
    index = np.argmin(np.abs(ds.distance.values - 13.8))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # line, = ax.plot([], [], label="Along axis velocity")
    # line2, = ax.plot([], [], label="Along axis velocity (previous point)", linestyle="--")
    # line3, = ax.plot([], [], label="Along axis velocity (next point)", linestyle="--")
    # ax.set_ylim(0, 2)
    # ax.set_xlim(-1., 1.)
    # time_text = ax.text(
    #     0.02, 0.95, "",
    #     transform=ax.transAxes,
    #     va="top"
    # )
    # ax.set_title("Along Axis Velocity at steep change in surface elevation")
    # ax.set_ylabel("Height above bed (m)")
    # ax.set_xlabel("Along Axis Velocity (m/s)")
    # ax.legend(loc="upper right")

    # def animateU3d(i):
    #     line.set_data(ds3d.uk[i, :, 0, index], ds3d.zct[i, :,0, index])
    #     line2.set_data(ds3d.uk[i, :, 0, index-1], ds3d.zct[i, :,0, index-1])
    #     line3.set_data(ds3d.uk[i, :, 0, index+1], ds3d.zct[i, :,0, index+1])
    #     time_text.set_text(f"Time: {pd.to_datetime(ds3d.time[i].values)}")
    #     return line, line2, line3, time_text

    # ani = animation.FuncAnimation(fig, animateU3d, frames=len(ds3d.time), interval=100, blit=True)
    # if show_plots:
    #     plt.show()
    # ani.save(output_prefix + "_along_axis_velocity_vertical_profile_animation.gif", writer="imagemagick", dpi=50)
    # # delete ani
    # del ani

    # plot time series of t2m, d2m in one plot and swr and shf in another plot
    if type(airsea) == pygetm.airsea.FluxesFromMeteo:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
        axs[0, 0].plot(ds.time, ds.u10[:, 0, 0], label="Wind U10")
        axs[0, 0].plot(ds.time, ds.v10[:, 0, 0], label="Wind V10")
        axs[0, 0].set_title("Wind at 10m")
        axs[0, 0].set_ylabel("Wind speed (m/s)")
        axs[0, 0].legend()
        axs[0, 1].plot(ds.time, ds.t2m[:, 0, 0], label="Air Temperature")
        if humidity_measure == pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE:
            axs[0, 1].plot(ds.time, ds.d2m[:, 0, 0], label="Dew Point Temperature")
        axs[0, 1].set_title("Air and Dew Point Temperature at 2m")
        axs[0, 1].set_ylabel("Temperature (°C)")
        axs[0, 1].legend()
        axs[1, 0].plot(ds.time, ds.sp[:, 0, 0], label="Surface Pressure")
        axs[1, 0].set_title("Surface Pressure")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel("Pressure (Pa)")
        axs[1, 0].legend()
        axs[1, 1].plot(ds.time, ds.swr[:, 0, 0], label="Shortwave radiation")
        axs[1, 1].plot(ds.time, ds.shf[:, 0, 0], label="Sensible heat flux")
        axs[1, 1].set_title("Shortwave Radiation and Sensible Heat Flux")
        axs[1, 1].set_xlabel("Time")
        axs[1, 1].set_ylabel("W/m^2")
        axs[1, 1].legend()
        fig.savefig(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_meteo.png"), dpi=150)
        plt.close(fig)
        
    # # make an animation of a 2D pcolormesh transect along the 2D slice of the along axis velocity, using the nodes in the 2D slice
    animate_transect(
        distance2d,
        ds3d.zct[:, :, 0, :],
        ds3d.salt[:, :, 0, :],
        ds3d.time.values,
        "Salinity",
        units="g kg$^{-1}$",
        cmap=cmocean.cm.haline,
        output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_salinity.mp4")
    )
    plt.close('all')
    animate_transect(
        distance2d,
        ds3d.zct[:, :, 0, :],
        ds3d.temp[:, :, 0, :],
        ds3d.time.values,
        "Temperature",
        units="°C",
        cmap=cmocean.cm.thermal,
        output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_temperature.mp4")
    )
    plt.close('all')
    vmax = np.nanpercentile(np.abs(ds3d.uk), 99)

    animate_transect(
        distance2d,
        ds3d.zct[:, :, 0, :],
        ds3d.uk[:, :, 0, :],
        ds3d.time.values,
        "Along-axis velocity",
        units="m s$^{-1}$",
        cmap=cmocean.cm.balance,
        vmin=-vmax,
        vmax=vmax,
        output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_along_axis_velocity_transect.mp4")
    )
    distance2di = np.tile(ds.distance.values, (ds3d.zft.shape[1], 1))

    animate_transect(
        distance2di,
        ds3d.zft[:, :, 0, :],
        ds3d.tke[:, :, 0, :],
        ds3d.time.values,
        "Turbulent kinetic energy",
        units="m$^2$ s$^{-2}$",
        cmap=cmocean.cm.matter,
        output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_tke.mp4")
    )
    plt.close('all')

if __name__ == "__main__":
    main()