"""
tamar_tef_along_axis.py
=======================
Total Exchange Flow (TEF) analysis along the main axis of a 2D x/z estuarine
slice model (e.g. pygetm curvilinear sigma-coordinate output).

For every x-column in the model the volume flux through the vertical face is
sorted into salinity bins and time-averaged over a user-specified interval
(default: monthly).  The bulk exchange quantities Q_in, Q_out, s_in, s_out are
then returned as functions of (x, averaging_period).

Assumptions about the input NetCDF
-----------------------------------
  u(time, z, x)       -- along-axis velocity [m s-1]; positive = seaward
  salt(time, z, x)    -- salinity [PSU]
  dz(time, z, x)      -- layer thickness [m]   (or reconstructed from sigma)
  B(x)                -- cross-section width / effective width [m]
                         (can be a 1-D variable or a scalar)
  time                -- CF-convention datetime coordinate

Adapt the variable names in the CONFIG section at the top to match your file.

Usage
-----
    python tamar_tef_along_axis.py \
        --input  /path/to/model_output.nc \
        --output /path/to/tef_results.nc  \
        --freq   MS                        # pandas offset alias: MS=month-start
                                           # other examples: W, QS, AS

Dependencies: numpy, xarray, pandas, scipy, pytef  (pip install pytef)
"""

import argparse
import csv
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pyTEF.calc import calc_bulk_values

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Edit these to match your NetCDF variable / dimension names
VAR = dict(
    time  = "time",
    x     = "xt",          # along-axis dimension (index or distance in m)
    z     = "z",          # vertical layer dimension (top-to-bottom or bottom-to-top)
    u     = "uk",          # along-axis velocity [m s-1]
    salt  = "salt",       # salinity  [PSU]
    dz    = "hn",         # layer thickness [m]; set to None to reconstruct below
    width = "B",          # cross-section width [m]; set to None for unit width
    eta   = "zt",        # free-surface height [m]; only needed if dz is None
    H     = "H",          # undisturbed water depth [m]; only needed if dz is None
    sigma = "sigma",      # sigma fractions (0..1); only needed if dz is None
)

N_SAL_BINS = 512          # number of salinity sorting bins
S_RANGE    = None         # (s_min, s_max) or None to auto-detect per averaging window
SIGN_U     = 1            # multiply u by this to make positive = INTO estuary (landward)
                          # set to -1 if positive u is seaward in your convention
# ───────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> xr.Dataset:
    """Open dataset with minimal decoding to keep memory manageable."""
    ds = xr.open_dataset(path, chunks={VAR["time"]: 24}).squeeze("yt", drop=True)
    # calculate dz
    if VAR["dz"] is None or VAR["dz"] not in ds:
        ds[VAR["dz"]] = ds['zft'].diff(dim='zi', label='lower').rename({'zi': VAR["z"]})
    return ds


def get_layer_thickness(ds: xr.Dataset) -> xr.DataArray:
    """
    Return dz(time, z, x) [m].
    Uses the dz variable if present; otherwise reconstructs from sigma coords.
    """
    if VAR["dz"] and VAR["dz"] in ds:
        return ds[VAR["dz"]]

    # Reconstruct for a sigma model:  dz_k = |d_sigma_k| * (H + eta)
    eta   = ds[VAR["eta"]]          # (time, x)
    H     = ds[VAR["H"]]            # (x,)
    sigma = ds[VAR["sigma"]]        # (z,)  -- interface or layer-centre values

    total_depth = H + eta           # (time, x)

    # Layer thickness from sigma differences (works for uniform sigma spacing)
    if sigma.sizes[VAR["z"]] == ds.sizes[VAR["z"]]:
        # sigma at layer centres → approximate dz as uniform
        nz = sigma.sizes[VAR["z"]]
        d_sigma = 1.0 / nz
    else:
        d_sigma = np.abs(np.diff(sigma.values))  # (nz,)

    dz = d_sigma * total_depth      # broadcast: (time, x) * scalar or (z,)
    return dz.transpose(VAR["time"], VAR["z"], VAR["x"])


def compute_volume_flux(ds: xr.Dataset, dz: xr.DataArray) -> xr.DataArray:
    """
    q_vol(time, z, x) = u * dz * B   [m3 s-1 per layer]
    Positive = landward (INTO estuary) after applying SIGN_U.
    """
    u = ds[VAR["u"]] * SIGN_U

    if VAR["width"] and VAR["width"] in ds:
        B = ds[VAR["width"]]    # (x,) or scalar
    else:
        B = 1.0                 # unit-width slice

    q_vol = u * dz * B          # (time, z, x)
    return q_vol


def build_salinity_bins(s_min: float, s_max: float) -> np.ndarray:
    """Return N_SAL_BINS+1 bin edges covering [s_min, s_max]."""
    return np.linspace(s_min, s_max, N_SAL_BINS + 1)


def sort_flux_into_salinity_bins(
    q_vol: np.ndarray,    # (time, z)
    salt:  np.ndarray,    # (time, z)
    bin_edges: np.ndarray # (N+1,)
) -> np.ndarray:
    """
    Bin-sort volume flux into salinity classes for a single x-column.

    Returns
    -------
    q_s : (time, N_SAL_BINS)  volume flux per salinity bin [m3 s-1 per salinity unit]
    """
    nt, nz = q_vol.shape
    N = len(bin_edges) - 1
    q_s = np.zeros((nt, N), dtype=np.float64)

    bin_width = bin_edges[1] - bin_edges[0]

    for ti in range(nt):
        s_col  = salt[ti]   # (nz,)
        q_col  = q_vol[ti]  # (nz,)

        # Map each layer to a bin index
        idx = np.searchsorted(bin_edges[1:-1], s_col)  # (nz,) in [0, N-1]
        np.add.at(q_s[ti], idx, q_col)

    # Normalise to flux per salinity unit (consistent with pyTEF convention)
    q_s /= bin_width
    return q_s


def time_average_q(
    q_s:    np.ndarray,    # (time, N_bins)
    times:  pd.DatetimeIndex,
    freq:   str
) -> tuple[np.ndarray, list]:
    """
    Average q_s over non-overlapping windows defined by freq.

    Returns
    -------
    q_avg  : (n_periods, N_bins)
    periods: list of period-start Timestamps
    """
    df = pd.DataFrame(q_s, index=times)
    grouped = df.resample(freq)
    q_avg   = grouped.mean().values          # (n_periods, N_bins)
    periods = list(grouped.mean().index)
    return q_avg, periods


def compute_Q_profile(q_avg: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Integrate q from s=s_max down to each bin interface to get Q(S).
    Q(S) = ∫_{S}^{S_max} q(s') ds'   → inflow is Q > 0 at high S
    Shape: (n_periods, N_bins+1)  at bin interfaces
    """
    ds_bin = bin_edges[1] - bin_edges[0]
    # cumsum from high salinity to low salinity (flip, cumsum, flip back)
    Q = np.flip(
        np.cumsum(np.flip(q_avg, axis=-1) * ds_bin, axis=-1),
        axis=-1
    )
    # Append a zero at the high-salinity end (Q(S_max) = 0 by definition)
    Q = np.concatenate([Q, np.zeros((Q.shape[0], 1))], axis=-1)
    return Q


def extract_bulk_values(
    Q:         np.ndarray,   # (n_periods, N_bins+1)
    bin_edges: np.ndarray    # (N_bins+1,)
) -> dict:
    """
    Apply pyTEF dividing-salinity method to extract bulk Q_in, Q_out, s_in, s_out.

    Returns dict of arrays shaped (n_periods,).
    Only the first inflow/outflow layer is returned (handles two-layer exchange).
    For multi-layer exchange the raw Q profile should be inspected directly.
    """
    n_periods = Q.shape[0]
    Qin  = np.full(n_periods, np.nan)
    Qout = np.full(n_periods, np.nan)
    sin  = np.full(n_periods, np.nan)
    sout = np.full(n_periods, np.nan)

    for ti in range(n_periods):
        Q_prof = Q[ti]
        if np.all(np.isnan(Q_prof)) or np.nanmax(np.abs(Q_prof)) == 0:
            continue
        try:
            result = calc_bulk_values(
                coord=xr.DataArray(bin_edges, dims=["s"]),
                Q=xr.DataArray(Q_prof, dims=["s"])
            )
            # result contains Qin, Qout as arrays (multiple layers possible)
            qi = np.atleast_1d(result.Qin.values)
            qo = np.atleast_1d(result.Qout.values)
            si = np.atleast_1d(result.sin.values)  if hasattr(result, "sin")  else np.array([np.nan])
            so = np.atleast_1d(result.sout.values) if hasattr(result, "sout") else np.array([np.nan])
            Qin[ti]  = qi[0] if len(qi) > 0 else np.nan
            Qout[ti] = qo[0] if len(qo) > 0 else np.nan
            sin[ti]  = si[0] if len(si) > 0 else np.nan
            sout[ti] = so[0] if len(so) > 0 else np.nan
        except Exception:
            pass

    return dict(Qin=Qin, Qout=Qout, sin=sin, sout=sout)


def run_tef(input_path: str, output_path: str, freq: str, depth_file: str):
    print(f"Opening {input_path}")
    ds = load_data(input_path)
    # remove yt dimension if present (not needed for 2D slice)
    if "yt" in ds.dims:
        ds = ds.drop_vars("yt")
    times  = pd.DatetimeIndex(ds[VAR["time"]].values)
    x_vals = ds[VAR["x"]].values
    nx     = len(x_vals)

    dz    = get_layer_thickness(ds)
    # read width of section
    if VAR["width"] and VAR["width"] in ds:
        B = ds[VAR["width"]]    # (x,) or scalar
    else:
        depths = []
        with open(depth_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                section_id = int(row["section_id"])
                depths.append([section_id, float(row['real_mean_depth_m']), float(row["element_area_m2"])])
        B = xr.DataArray([d[2] for d in depths], dims=[VAR["x"]])  # effective width from element area
        ds[VAR["width"]] = B[1:]  # assign to dataset, skipping first section if needed

    q_vol = compute_volume_flux(ds, dz)   # (time, z, x)
    salt  = ds[VAR["salt"]]               # (time, z, x)

    # Global salinity range (so bins are consistent across x and time)
    if S_RANGE is None:
        print("Scanning salinity range … ", end="", flush=True)
        s_min = float(salt.min().compute())
        s_max = float(salt.max().compute())
        # Add small margins
        s_min = max(0.0, s_min - 0.5)
        s_max = s_max + 0.5
        print(f"S ∈ [{s_min:.2f}, {s_max:.2f}]")
    else:
        s_min, s_max = S_RANGE

    bin_edges = build_salinity_bins(s_min, s_max)
    bin_ctrs  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Determine output time axis from a dummy resample
    dummy = pd.Series(0, index=times).resample(freq).mean()
    period_times = dummy.index
    n_periods    = len(period_times)

    # Output arrays  (x, period)
    Qin_out  = np.full((nx, n_periods), np.nan)
    Qout_out = np.full((nx, n_periods), np.nan)
    sin_out  = np.full((nx, n_periods), np.nan)
    sout_out = np.full((nx, n_periods), np.nan)
    # Store full Q(s) profile for diagnostics: (x, period, N_bins+1)
    Q_profiles = np.full((nx, n_periods, N_SAL_BINS + 1), np.nan)

    print(f"Processing {nx} x-columns at '{freq}' averaging …")

    for xi in range(nx):
        if xi % max(1, nx // 20) == 0:
            print(f"  x = {xi}/{nx}")

        # Load this column into memory
        q_col   = q_vol.isel({VAR["x"]: xi}).values   # (time, z)
        s_col   = salt.isel({VAR["x"]: xi}).values     # (time, z)

        # Handle any NaN/masked layers (e.g. above bathymetry)
        valid = np.isfinite(q_col) & np.isfinite(s_col)
        q_col[~valid] = 0.0
        s_col[~valid] = 0.0

        # Sort into salinity bins
        q_s = sort_flux_into_salinity_bins(q_col, s_col, bin_edges)  # (time, N)

        # Time-average
        q_avg, _ = time_average_q(q_s, times, freq)  # (n_periods, N)

        # Integrate to get Q profile
        Q_prof = compute_Q_profile(q_avg, bin_edges)  # (n_periods, N+1)
        Q_profiles[xi] = Q_prof

        # Bulk values via pyTEF dividing-salinity
        bulk = extract_bulk_values(Q_prof, bin_edges)
        Qin_out[xi]  = bulk["Qin"]
        Qout_out[xi] = bulk["Qout"]
        sin_out[xi]  = bulk["sin"]
        sout_out[xi] = bulk["sout"]

    # ── Save output ────────────────────────────────────────────────────────────
    print(f"Writing {output_path}")

    coords = {
        "x":      (["x"],      x_vals,             {"long_name": "Along-axis distance", "units": "m"}),
        "time":   (["time"],   period_times.values, {"long_name": f"Period start ({freq})"}),
        "s_bin":  (["s_bin"],  bin_ctrs,            {"long_name": "Salinity bin centre", "units": "PSU"}),
        "s_iface":(["s_iface"],bin_edges,            {"long_name": "Salinity bin interface", "units": "PSU"}),
    }

    ds_out = xr.Dataset(
        {
            "Qin":  (["x", "time"], Qin_out,
                     {"long_name": "TEF inflow volume flux", "units": "m3 s-1",
                      "comment": "Positive = landward (into estuary)"}),
            "Qout": (["x", "time"], Qout_out,
                     {"long_name": "TEF outflow volume flux", "units": "m3 s-1",
                      "comment": "Negative = seaward"}),
            "sin":  (["x", "time"], sin_out,
                     {"long_name": "TEF inflow salinity", "units": "PSU"}),
            "sout": (["x", "time"], sout_out,
                     {"long_name": "TEF outflow salinity", "units": "PSU"}),
            "Q_profile": (["x", "time", "s_iface"], Q_profiles,
                     {"long_name": "Integrated transport Q(S)",
                      "units": "m3 s-1",
                      "comment": "Q(S) = integral of q from S to S_max; "
                                 "shape (x, time, N_sal_bins+1)"}),
        },
        coords=coords,
        attrs={
            "source":      input_path,
            "freq":        freq,
            "N_sal_bins":  N_SAL_BINS,
            "s_min":       s_min,
            "s_max":       s_max,
            "sign_u_conv": f"u multiplied by {SIGN_U} → positive = landward",
            "method":      "TEF dividing-salinity (MacCready 2011; Lorenz et al. 2019)",
        }
    )

    ds_out.to_netcdf(output_path)
    print("Done.")
    return ds_out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TEF analysis along the estuary axis of a 2D x/z slice model."
    )
    p.add_argument("--input",  required=True, help="Path to model NetCDF output")
    p.add_argument("--output", required=True, help="Path for TEF results NetCDF")
    p.add_argument("--depth_file", required=True, help="Path to CSV file containing depth information")
    p.add_argument("--freq",   default="MS",
                   help="Pandas resample frequency alias (default: MS = month-start). "
                        "Examples: W (weekly), QS (quarterly), AS (annual), "
                        "30D (30-day windows), 12h (12-hourly — tidal average)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tef(args.input, args.output, args.freq, args.depth_file)
