# Script to download CMEMS data for Vigo region using copernicusmarine API
# this needs the base conda environment rather than pygetm
import argparse
import glob
import os
import getpass
from pathlib import Path, WindowsPath
import pyproj
import numpy as np
import xarray as xr
import estimate_channel_section_volumes as ecs
import warnings
from typing import Literal
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
# os.putenv("COPERNICUSMARINE_DISABLE_SSL_CONTEXT","True")
# os.putenv("COPERNICUSMARINE_TRUST_ENV","False")
# os.putenv("COPERNICUSMARINE_TRUST_ENV","False")
# os.putenv("COPERNICUSMARINE_SERVICE_USERNAME","mbedington1")
# os.putenv("COPERNICUSMARINE_SERVICE_PASSWORD","asj#euT37")
import copernicusmarine
# copernicusmarine.login(
#     username = "mbedington1",
#     password = "asj#euT37",
#     configuration_file_directory=WindowsPath('C:\\Users\\rito\\Code\\pygetm_tests\\.copernicusmarine'),
#     overwrite_configuration_file= False,
#     skip_if_user_logged_in=True
# )
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download CMEMS data for Vigo region")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"/data/{getpass.getuser()}/Tamar_pyGETM/data"),
        help="Directory to save downloaded data",
    )
    # start date, end date and bbox for the data download
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for data download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for data download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
        help="Bounding box for data download",
    )
    # boundary point for extracting the closest non-nan value from the downloaded data
    parser.add_argument(
        "--bdy-lon",
        type=float,
        help="Longitude of boundary point for extracting closest non-nan value",
    )
    parser.add_argument(
        "--bdy-lat",
        type=float,
        help="Latitude of boundary point for extracting closest non-nan value",
    )
    return parser
def dd_to_dir(dd, is_lat):
    direction = ('N' if dd >= 0 else 'S') if is_lat else ('E' if dd >= 0 else 'W')
    return f"{abs(dd):.2f}{direction}"
def pygetm_varname_to_dataset_id(varname, dataset_id="MED"):
    """
    Map CMEMS variable names to dataset-specific id's to access. Examples are
    uo, vo for 3D current data in cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m
    zos for sea surface height in cmems_mod_med_phy-ssh_anfc_4.2km_PT15M-i
    thetao for temperature in cmems_mod_med_phy-tem_anfc_4.2km-3D_PT1H-m
    so for salinity in cmems_mod_med_phy-sal_anfc_4.2km-3D_PT1H-m
    bathymetry in cmems_mod_med_phy-bathymetry_4.2km-m

    Args:
        varname (str): Variable name to map. One of "zt", "temp", "salt", "u", "v", "U", "V".
        dataset_id (str, optional): Dataset ID to use for mapping. Defaults to "cmems_mod_med_phy-VAR_anfc_RES-3D_PT1H-m".
    """
    dct = {
        "zt": "cmems_mod_med_phy-ssh_anfc_4.2km_PT15M-i",
        "temp": "cmems_mod_med_phy-tem_anfc_4.2km-3D_PT1H-m",
        "salt": "cmems_mod_med_phy-sal_anfc_4.2km-3D_PT1H-m",
        "u": "cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m",
        "v": "cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m",
        "U": "cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m",
        "V": "cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m",
        "bathymetry": "cmems_mod_med_phy_anfc_4.2km_static",
    }
    if dataset_id == "IBI":
        dct = {
            "zt": "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i",
            "temp": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m",
            "salt": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m",
            "u": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m",
            "v": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m",
            "U": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m",
            "V": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m",
            "bathymetry": "cmems_mod_ibi_phy_anfc_0.027deg_static",
            "O3_c": "cmems_mod_ibi_bgc_anfc_0.027deg-3D_P1D-m",
            "N1_p": "cmems_mod_ibi_bgc_anfc_0.027deg-3D_P1D-m",
            "N3_n": "cmems_mod_ibi_bgc_anfc_0.027deg-3D_P1D-m",
            "N4_n": "cmems_mod_ibi_bgc_anfc_0.027deg-3D_P1D-m",
            "N5_s": "cmems_mod_ibi_bgc_anfc_0.027deg-3D_P1D-m",
            "O2_o": "cmems_mod_ibi_bgc_anfc_0.027deg-3D_P1D-m",
        }
    if dataset_id == "IBI_MULTIYEAR":
        dct = {
            "zt": "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m",
            "temp": "cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m", 
            "salt": "cmems_mod_ibi_phy-sal_my_0.027deg_P1D-m",
            "u1": "cmems_mod_ibi_phy-cur_my_0.027deg_PT1H-m",
            "v1": "cmems_mod_ibi_phy-cur_my_0.027deg_PT1H-m",
            "U": "cmems_mod_ibi_phy-cur_my_0.027deg_PT1H-m",
            "V": "cmems_mod_ibi_phy-cur_my_0.027deg_PT1H-m",
            "bathymetry": "cmems_mod_ibi_phy_my_0.027deg-3D_static",
            "O3_c": "cmems_mod_ibi_bgc-car_my_0.027deg_P1D-m",
            "N1_p": "cmems_mod_ibi_bgc-nut_my_0.027deg_P1D-m",
            "N3_n": "cmems_mod_ibi_bgc-nut_my_0.027deg_P1D-m",
            "N4_n": "cmems_mod_ibi_bgc-nut_my_0.027deg_P1D-m",
            "N5_s": "cmems_mod_ibi_bgc-nut_my_0.027deg_P1D-m",
            "O2_o": "cmems_mod_ibi_bgc-o2_my_0.027deg_P1D-m",

        }
    elif dataset_id == "NORTHWESTSHELF":
        dct = {
            "zt": "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i",
            "temp": "cmems_mod_nws_phy-tem_anfc_1.5km-3D_PT1H-i",
            "salt": "cmems_mod_nws_phy-sal_anfc_1.5km-3D_PT1H-i",
            "u": "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i",
            "v": "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i",
            "U": "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i",
            "V": "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i",
            "bathymetry": "cmems_mod_nws_phy_anfc_1.5km_static",
        }
    elif dataset_id == "BALTIC":
        dct = {
            "zt": "cmems_mod_bal_phy_anfc_PT15M-i",
            "temp": "cmems_mod_bal_phy_anfc_PT1H-i",
            "salt": "cmems_mod_bal_phy_anfc_PT1H-i",
            "u": "cmems_mod_bal_phy_anfc_PT1H-i",
            "v": "cmems_mod_bal_phy_anfc_PT1H-i",
            "U": "cmems_mod_bal_phy_anfc_PT1H-i",
            "V": "cmems_mod_bal_phy_anfc_PT1H-i",
            "bathymetry": "cmems_mod_bal_phy_anfc_static",
        }
    elif dataset_id == "MED_DAILY":
        dct = {
            "zt": "cmems_mod_med_phy-ssh_anfc_4.2km_PT15M-i",
            "temp": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
            "salt": "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
            "u": "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
            "v": "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
            "U": "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
            "V": "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
            "bathymetry": "cmems_mod_med_phy_anfc_4.2km_static",
        }
    elif dataset_id == "MED_OFFLINE":
        dct = {
            "zt": "cmems_mod_med_phy-ssh_anfc_4.2km_PT15M-i",
            "temp": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
            "salt": "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
            "bathymetry": "cmems_mod_med_phy_anfc_4.2km_static",
        }
    elif dataset_id == "MED":
        pass
    else:
        raise ValueError(
            f"Dataset ID {dataset_id} not recognized. Defaults to Mediterranean dataset."
        )
    return dct[varname]
def varname_to_short(
    varnames: list[str],
    dataset_id=None,
    reverse=False,
):
    if reverse:
        dct = {
            "zos": "zt",
            "thetao": "temp",
            "so": "salt",
            "vxo": "u",
            "vyo": "v",
            "uo": "uk",
            "vo": "vk",
            "U": "U",
            "V": "V",
            "ubar": "u1",
            "vbar": "v1",
            # FABM-ERSEM variables
            "dissic": "O3_c",
            "po4": "N1_p",
            "no3": "N3_n",
            "o2": "O2_o",
            "nh4": "N4_n",
            "si": "N5_s",
        }
    elif dataset_id == "ARCTIC":
        dct = {
            "zt": "zos",
            "temp": "thetao",
            "salt": "so",
            "u": "vxo",
            "v": "vyo",
            "U": "uxo",
            "V": "vyo",
        }
    elif dataset_id == "cmems_mod_bal_phy_anfc_PT15M-i":
        dct = {
            "zt": "sla",
            "temp": "thetao",
            "salt": "so",
            "u": "uo",
            "v": "vo",
            "U": "uo",  # will require postprocessing
            "V": "vo",
        }
    else:
        dct = {
            "zt": "zos",
            "temp": "thetao",
            "salt": "so",
            "uk": "uo",
            "vk": "vo",
            "U": "uo",  # will require postprocessing
            "V": "vo",  # will require postprocessing
            "u1": "ubar",
            "v1": "vbar",
            # for ersem variables
            "O3_c": "dissic",
            "N1_p": "po4",
            "N3_n": "no3",
            "O2_o": "o2",
            "N4_n": "nh4",
            "N5_s": "si",
        }

    translated = [dct[v] for v in varnames]
    return translated

def download_cmems_data(bbox: list[float], start_date: str, end_date: str, download_vars: list[str] = ["temp", "salt"], dataset_product: str = "IBI_MULTIYEAR"): 
    """Download CMEMS data for the specified bounding box and time range.
    Args:
        bbox: List of [min_lon, max_lon, min_lat, max_lat]
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        """
    for var in download_vars:
        dataset_id = pygetm_varname_to_dataset_id(var, dataset_id=dataset_product)
        short_varname = varname_to_short([var], dataset_id=dataset_id)[0]
        print(f"Downloading {var} data from {dataset_id} for bbox: {bbox} from {start_date} to {end_date}...")

        copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=[short_varname],
        minimum_longitude=bbox[0],
        maximum_longitude=bbox[1],
        minimum_latitude=bbox[2],
        maximum_latitude=bbox[3],
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_depth=0,
        maximum_depth=200,
        )



    # copernicusmarine.subset(
    #         dataset_id="cmems_mod_ibi_phy-sal_my_0.027deg_P1D-m",
    #         variables=["so"],
    #         minimum_longitude=bbox[0],
    #         maximum_longitude=bbox[1],
    #         minimum_latitude=bbox[2],
    #         maximum_latitude=bbox[3],
    #         start_datetime=start_date,
    #         end_datetime=end_date,
    #         minimum_depth=0,
    #         maximum_depth=200,
    #     )
    # if download_fabm:
    #     copernicusmarine.subset(
    #         dataset_id="cmems_mod_ibi_phy-fabm_my_0.027deg_P1D-m",
    #         variables=["fabm"],
    #         minimum_longitude=bbox[0],
    #         maximum_longitude=bbox[1],
    #         minimum_latitude=bbox[2],
    #         maximum_latitude=bbox[3],
    #         start_datetime=start_date,
    #         end_datetime=end_date,
    #         minimum_depth=0,
    #         maximum_depth=200,
    #     )
def extract_boundary_conditions(bbox: list[float], start_date: str, end_date: str, bdy_lon: float, bdy_lat: float, output_dir: Path, vars2proc: list[str] = ["temp", "salt"]):
    # Extract the closest non-nan point to Tamar and build boundary series for each requested variable.
    min_lon, max_lon, min_lat, max_lat = bbox
    min_lonstr = dd_to_dir(min_lon, is_lat=False)
    max_lonstr = dd_to_dir(max_lon, is_lat=False)
    min_latstr = dd_to_dir(min_lat, is_lat=True)
    max_latstr = dd_to_dir(max_lat, is_lat=True)
    bbox_str = f"{min_lonstr}-{max_lonstr}_{min_latstr}-{max_latstr}"
    start_date_norm = start_date.replace("/", "-")
    end_date_norm = end_date.replace("/", "-")

    datasets: dict[str, xr.Dataset] = {}
    short_names: dict[str, str] = {}

    for var in vars2proc:
        dataset_id = pygetm_varname_to_dataset_id(var, dataset_id="IBI_MULTIYEAR")
        nemovar = varname_to_short([var], dataset_id="IBI_MULTIYEAR")[0]
        short_names[var] = nemovar

        patterns: list[str] = []
        for root in (Path("./"), output_dir):
            patterns.extend(
                [
                    str(root / f"{dataset_id}_{nemovar}_{bbox_str}_*_{start_date_norm}_{end_date_norm}.nc"),
                    str(root / f"{dataset_id}_{nemovar}_{bbox_str}_*_{start_date_norm}-{end_date_norm}.nc"),
                    str(root / f"{dataset_id}_{nemovar}_*_{start_date_norm}_{end_date_norm}.nc"),
                    str(root / f"{dataset_id}_{nemovar}_*_{start_date_norm}-{end_date_norm}.nc"),
                ]
            )

        found_file = None
        for pattern in patterns:
            print(f"Looking for {var} file with pattern: {pattern}")
            matches = glob.glob(pattern)
            if matches:
                found_file = matches[0]
                break

        if found_file is None:
            raise FileNotFoundError(
                f"No file found for var '{var}' ({nemovar}) using dataset '{dataset_id}' "
                f"between {start_date_norm} and {end_date_norm}."
            )

        datasets[var] = xr.open_dataset(found_file, decode_times=time_coder)

    output_data = {}
    for var in vars2proc:
        print(f"Processing {var}...")
        ds = datasets[var]
        short_name = short_names[var]
        # Find non-nan grid point closest to the boundary point.
        lon_mesh, lat_mesh = np.meshgrid(ds.longitude.values, ds.latitude.values)
        dist = np.sqrt((lon_mesh - bdy_lon)**2 + (lat_mesh - bdy_lat)**2).flatten()
        ref_flat = ds[short_name].values[0, 0, ...].flatten()
        valid_dist = np.where(~np.isnan(ref_flat), dist, np.inf)
        closest_index = np.argmin(valid_dist)   

        ref_shape = ds[short_name].values[0, 0, ...].shape
        j, i = np.unravel_index(closest_index, ref_shape)

        series = ds.isel(longitude=i, latitude=j)[short_name]
        if "depth" in series.dims:
            series = series.ffill(dim="depth")
        series = series.expand_dims({"bdy": [1]}, axis=1)
        drop_coords = [coord for coord in ("latitude", "longitude") if coord in series.coords]
        if drop_coords:
            series = series.drop_vars(drop_coords)
        # rename the time dimension to time_var so that different variables with different time dimensions can be merged into a single dataset
        time_var = f"time_{var}"
        series = series.rename({"time": time_var})
        # print time range of valid data for this variable
        valid_times = series.dropna(dim=time_var, how="all")[time_var].values
        if valid_times.size > 0:
            print(f"Valid time range for {var}: {valid_times[0]} to {valid_times[-1]}")
        # Save with internal pygetm/FABM-ERSEM names (e.g. temp/salt/O3_c/N1_p).
        output_name = varname_to_short([short_name], reverse=True)[0]
        output_data[output_name] = series

    output_file = output_dir / f"tamar_boundary_conditions_{start_date}_{end_date}.nc"
    xr.Dataset(output_data).to_netcdf(output_file)

    for ds in datasets.values():
        ds.close()

    print(f"Boundary conditions saved to {output_file}")

def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bbox = args.bbox
    print(f"Downloading CMEMS data for bbox: {bbox} from {args.start_date} to {args.end_date}...")
    download_cmems_data(bbox, args.start_date, args.end_date)
    extract_boundary_conditions(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        bdy_lon=args.bdy_lon,
        bdy_lat=args.bdy_lat,
        output_dir=args.output_dir,
        vars2proc=["temp", "salt"],
    )

if __name__ == "__main__":
    main()
# copernicusmarine.subset(
#   dataset_id="cmems_mod_ibi_phy_my_0.083deg-2D_PT1H-m",
#   variables=["mlotst", "thetao", "ubar",  "vbar",  "zos"],
#   minimum_longitude=-9.2,
#   maximum_longitude=-8.6,
#   minimum_latitude=41.8,
#   maximum_latitude=42.5,
#   start_datetime="2021-09-01T00:00:00",
#   end_datetime="2021-10-01T23:00:00",
# )

# import copernicusmarine 
# import xarray as xr 
# import pyproj 
# import shutil from datetime 
# import datetime, timedelta 
# import glob

# # Dataset_ID IBI_MULTIYEAR_PHY_005_002
# dataset_id = "cmems_mod_ibi_phy_my_0.083deg-3D_P1D-m"
# # Geographical area
# lat_min, lat_max = 41.8, 42.5
# lon_min, lon_max = -9.2, -8.6
# # Depth (if needed)
# #depth = slice(0,10)
# # List of variables to extract
# variables_of_interest = ["so", "thetao", "uo", "vo", "zos"]  
# # Requested time period
# start_date = datetime(2021, 9, 1)
# end_date = datetime(2021, 10, 1)
# # Directory to save the subsetted data
# output_directory = ' C:\\Users\\rito\\Code\\pygetm_tests\\Vigo\\CMEMS' 
# # Directory where the script is executed
# script_directory = 'C:\\Users\\rito\\Code\\pygetm_tests\\Vigo' 

# # Function to download, extract, and save the data
# def process_date(date):
#     date_str = date.strftime('%Y%m%d')
#     subset_output_file = os.path.join(output_directory, f'{date_str}_subset.nc')
    
#     # If the subset for this date already exists, skip downloading
#     if os.path.exists(subset_output_file):
#         print(f"File {subset_output_file} already exists, skipping.")
#         return
    
#     print(f"Attempting to download data for {date_str}...")
    
#     # Download the data file
#     copernicusmarine.get(
#         dataset_id=dataset_id,
#         filter=f"*{date_str}*",
#         force_download=True
#     )
    
#     # Search for the downloaded .nc file in all subdirectories
#     downloaded_files = glob.glob(os.path.join(output_directory, f'**/*{date_str}_*.nc'), recursive=True)
    
#     if not downloaded_files:
#         print(f"No files downloaded for {date_str}, skipping.")
#         return
    
#     downloaded_file = downloaded_files[0]
#     print(f"Processing file: {downloaded_file}")
    
#     # Open the downloaded file
#     ds = xr.open_dataset(downloaded_file)

#     # Extract the variables of interest for the specified region
#     data_arrays = []
#     for var in variables_of_interest:
#         da = ds[var].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))  # Add depth=depth if needed
#         data_arrays.append(da)
    
#     # Merge the extracted DataArrays
#     final = xr.merge(data_arrays)
    
#     # Save the subset to a new file
#     final.to_netcdf(subset_output_file)
    
#     # Close the dataset before deleting the file
#     ds.close()
    
#     # Delete the downloaded file and its subdirectories after closing the dataset
#     shutil.rmtree(os.path.dirname(downloaded_file))

# # Function to merge the subsets
# def merge_subsets():
#     # Define the final name of the merged file
#     output_name = "subset_final.nc"
    
#     # Find all subset files
#     filenames = glob.glob(os.path.join(output_directory, '*_subset.nc'))
    
#     # Open all files into a single dataset
#     combined_data = xr.open_mfdataset(filenames, combine='by_coords')
    
#     # Save the merged dataset
#     combined_data.compute().to_netcdf(os.path.join(output_directory, output_name))
#     print(f"All subsets merged into {output_name}")

# # Determine the start date based on already downloaded files
# existing_files = glob.glob(os.path.join(output_directory, '*_subset.nc'))

# if existing_files:
#     # Find the latest created file to determine the resume date
#     latest_file = max(existing_files, key=os.path.getctime)
#     start_date = datetime.strptime(os.path.basename(latest_file)[:8], '%Y%m%d') + timedelta(days=1)

# # Loop through each day in the period
# current_date = start_date
# while current_date <= end_date:
#     try:
#         process_date(current_date)
#         print(f"Processed {current_date.strftime('%Y-%m-%d')}")
#     except Exception as e:
#         print(f"Failed to process {current_date.strftime('%Y-%m-%d')}: {e}")
#     current_date += timedelta(days=1)

# # Call the function to merge the subsets
# merge_subsets()

# # Import modules
# import copernicusmarine

# # Set parameters
# data_request = {
#    "dataset_id_sst_gap_l3s" : "cmems_obs-sst_atl_phy_nrt_l3s_P1D-m",
#    "longitude" : [-6.17, -5.09], 
#    "latitude" : [35.75, 36.29],
#    "time" : ["2023-01-01", "2023-01-31"],
#    "variables" : ["sea_surface_temperature"]
# }

# # Load xarray dataset
# sst_l3s = copernicusmarine.open_dataset(
#     dataset_id = data_request["dataset_id_sst_gap_l3s"],
#     minimum_longitude = data_request["longitude"][0],
#     maximum_longitude = data_request["longitude"][1],
#     minimum_latitude = data_request["latitude"][0],
#     maximum_latitude = data_request["latitude"][1],
#     start_datetime = data_request["time"][0],
#     end_datetime = data_request["time"][1],
#     variables = data_request["variables"]
# )
