# pygetm_2d_xz

## Scripts to setup and run a 2D slice model of the Tamar

Clone this repository to get all you need to run 1 month of the Tamar 
Make sure you change directories defaults if you don't want to use all the commandline options

```bash

git clone git@github.com:riquitorres/pygetm_2d_xz.git
```


To get the full workflow you need to install pyfvcom in your pygetm conda environment (or a different one if you want)

```bash

git clone git@github.com:pmlmodelling/pyfvcom.git
git switch dev
```
To recreate the mesh definition files (nodes, depths and areas) and download cmems and era5 data run ...

```bash
python  setup_2D_model.py --mesh ../tamar_v0/tamar_v2_grd.dat  
```
To only download the forcing data use

```bash
python  -i download_forcing.py --estuary-name tamar --start-date 1994-01-01 --end-date 2025-12-31 --download-cmems-data --download-era5-data
```



To run the model and generate a few plots use

```bash
python -i run2Dslice.py --estuary-name tamar --start-date 2023-01-01 --end-date 2023-01-03  --fabm-file fabm-ersemv2004.yaml --cmems-file /data/rt/Tamar_pyGETM/data/tamar_boundary_conditions_1994-01-01_2025-12-31.nc 
```