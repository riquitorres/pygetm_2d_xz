# Script to download CMEMS data for Vigo region using copernicusmarine API
# this needs the base conda environment rather than pygetm
import argparse
import glob
import os
from pathlib import Path, WindowsPath
import pyproj
import numpy as np
import xarray as xr
import estimate_channel_section_volumes as ecs
import warnings
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

def download_cmems_data(bbox: list[float], start_date: str, end_date: str): 
    """Download CMEMS data for the specified bounding box and time range.
    Args:
        bbox: List of [min_lon, max_lon, min_lat, max_lat]
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        """
    copernicusmarine.subset(
      dataset_id="cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m",
      variables=["thetao"],
      minimum_longitude=bbox[0],
      maximum_longitude=bbox[1],
      minimum_latitude=bbox[2],
      maximum_latitude=bbox[3],
      start_datetime=start_date,
      end_datetime=end_date,
      minimum_depth=0,
      maximum_depth=200,
    )



    copernicusmarine.subset(
            dataset_id="cmems_mod_ibi_phy-sal_my_0.027deg_P1D-m",
            variables=["so"],
            minimum_longitude=bbox[0],
            maximum_longitude=bbox[1],
            minimum_latitude=bbox[2],
            maximum_latitude=bbox[3],
            start_datetime=start_date,
            end_datetime=end_date,
            minimum_depth=0,
            maximum_depth=200,
        )
def extract_boundary_conditions(bbox: list[float], start_date: str, end_date: str, bdy_lon: float, bdy_lat: float, output_dir: Path):
    cmems_file = "./cmems_mod_ibi_phy-{var}_my_0.027deg_P1D-m_{nemovar}_{bbox_str}_*_{start_date}_{end_date}.nc"    
    # extract the closest non-nan point to Tamar 
    min_lon, max_lon, min_lat, max_lat = bbox
    min_lonstr = dd_to_dir(min_lon, is_lat=False)
    max_lonstr = dd_to_dir(max_lon, is_lat=False)
    min_latstr = dd_to_dir(min_lat, is_lat=True)
    max_latstr = dd_to_dir(max_lat, is_lat=True)
    bbox_str = f"{min_lonstr}-{max_lonstr}_{min_latstr}-{max_latstr}"
    temp_files = str(Path('./') / cmems_file.format(var="temp", nemovar="thetao", bbox_str=bbox_str, start_date=start_date.replace("/", "-"), end_date=end_date.replace("/", "-")))
    salt_files = str(Path('./') / cmems_file.format(var="sal", nemovar="so", bbox_str=bbox_str, start_date=start_date.replace("/", "-"), end_date=end_date.replace("/", "-")))
    print(f"Looking for temp file with pattern: {temp_files}")
    if not glob.glob(temp_files):
        warnings.warn(f"No temp file found with pattern: {temp_files}")
        cmems_file = "./cmems_mod_ibi_phy-{var}_my_0.027deg_P1D-m_{nemovar}_*_{start_date}-{end_date}.nc"
        temp_files = str(Path('./') / cmems_file.format(var="temp", nemovar="thetao", bbox_str='*', start_date=start_date.replace("/", "-"), end_date=end_date.replace("/", "-")))
        salt_files = str(Path('./') / cmems_file.format(var="sal", nemovar="so", bbox_str='*', start_date=start_date.replace("/", "-"), end_date=end_date.replace("/", "-")))
        print(f"Looking for temp file with pattern: {temp_files}")
        temp_file = glob.glob(temp_files)[0]
        salt_file = glob.glob(salt_files)[0]
    else:
        temp_file = glob.glob(temp_files)[0]
        salt_file = glob.glob(salt_files)[0]
    temp = xr.open_dataset(temp_file, decode_times=time_coder)
    salt = xr.open_dataset(salt_file, decode_times=time_coder)
    # find non-nan grid point closest to the boundary point and set the boundary temp to that value
    lon_mesh, lat_mesh = np.meshgrid(temp.longitude.values, temp.latitude.values)
    dist = np.sqrt((lon_mesh - bdy_lon)**2 + (lat_mesh - bdy_lat)**2).flatten()
    # find first non-nan value in temp that is closest to the boundary point
    temp_flat = temp.thetao.values[0,0,...].flatten()
    valid_dist = np.where(~np.isnan(temp_flat), dist, np.inf)
    closest_index = np.argmin(valid_dist)   
    j, i = np.unravel_index(closest_index, temp.thetao.values[0,0,...].shape)
    # extract and fill any nans with nearest depth value
    temp_series = temp.isel(longitude=i, latitude=j).thetao.ffill(dim="depth")
    # extent temp dimension to include at least 1 for lat and lon
    temp_series = temp_series.expand_dims({"bdy": [1]}, axis=1)
    salt_series = salt.isel(longitude=i, latitude=j).so.ffill(dim="depth")
    salt_series = salt_series.expand_dims({"bdy": [1]}, axis=1)
    output_file = output_dir / f"tamar_boundary_conditions_{start_date}_{end_date}.nc"
    xr.Dataset({"temperature": temp_series, "salinity": salt_series}).to_netcdf(output_file)
    print(f"Boundary conditions saved to {output_file}")

def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bbox = args.bbox
    print(f"Downloading CMEMS data for bbox: {bbox} from {args.start_date} to {args.end_date}...")
    download_cmems_data(bbox, args.start_date, args.end_date)
    cmems_file = "./cmems_mod_ibi_phy-{var}_my_0.027deg_P1D-m_{nemovar}_{bbox_str}_*_{start_date}_{end_date}.nc"    
    # extract the closest non-nan point to Tamar 
    min_lon, max_lon, min_lat, max_lat = args.bbox
    min_lonstr = dd_to_dir(min_lon, is_lat=False)
    max_lonstr = dd_to_dir(max_lon, is_lat=False)
    min_latstr = dd_to_dir(min_lat, is_lat=True)
    max_latstr = dd_to_dir(max_lat, is_lat=True)
    bbox_str = f"{min_lonstr}-{max_lonstr}_{min_latstr}-{max_latstr}"
    temp_file = glob.glob(str(Path(args.output_dir) / cmems_file.format(var="temp", nemovar="thetao", bbox_str=bbox_str, start_date=args.start_date, end_date=args.end_date)))[0]
    salt_file = glob.glob(str(Path(args.output_dir) / cmems_file.format(var="sal", nemovar="so", bbox_str=bbox_str, start_date=args.start_date, end_date=args.end_date)))[0]
    temp = xr.open_dataset(temp_file, decode_times=time_coder)
    salt = xr.open_dataset(salt_file, decode_times=time_coder)
    # find non-nan grid point closest to the boundary point and set the boundary temp to that value
    lon_mesh, lat_mesh = np.meshgrid(temp.longitude.values, temp.latitude.values)
    dist = np.sqrt((lon_mesh - args.bdy_lon)**2 + (lat_mesh - args.bdy_lat)**2).flatten()
    # find first non-nan value in temp that is closest to the boundary point
    temp_flat = temp.thetao.values[0,0,...].flatten()
    valid_dist = np.where(~np.isnan(temp_flat), dist, np.inf)
    closest_index = np.argmin(valid_dist)   
    j, i = np.unravel_index(closest_index, temp.thetao.values[0,0,...].shape)
    # extract and fill any nans with nearest depth value
    temp_series = temp.isel(longitude=i, latitude=j).thetao.ffill(dim="depth")
    # extent temp dimension to include at least 1 for lat and lon
    temp_series = temp_series.expand_dims({"bdy": [1]}, axis=1)
    salt_series = salt.isel(longitude=i, latitude=j).so.ffill(dim="depth")
    salt_series = salt_series.expand_dims({"bdy": [1]}, axis=1)
    # drop latitude and longitude coordinates from temp and salt series so they can be used in the open boundary conditions
    temp_series = temp_series.drop_vars(["latitude", "longitude"])
    salt_series = salt_series.drop_vars(["latitude", "longitude"])
    # save the extracted boundary time series to a new netcdf file
    output_file = args.output_dir / f"tamar_boundary_conditions_{bbox_str}_{args.start_date}_{args.end_date}.nc"
    xr.Dataset({"temperature": temp_series, "salinity": salt_series}).to_netcdf(output_file)
    print(f"Boundary conditions saved to {output_file}")

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
