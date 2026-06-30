# run script to setup and run a 2D slice pyGETM model of the Tamar estuary

import argparse
import csv
import datetime
import logging
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
def setup_logging(
    log_file: str, 
) -> logging.Logger:
    """
    Set up logging to console and file.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    """
    handlers: list[logging.Handler] = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    handlers.append(ch)
    # Create file handler
    fh = logging.FileHandler(log_file)
    # fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    handlers.append(fh)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logger = logging.getLogger()
    # logger.info("pygetm %s", pygetm.get_version())

    return logger

class MySimulation(pygetm.Simulation):
    _MIXING_STAGE_ORDER = {
        "pre_state_update": 0,
        "micro_momentum_2d": 1,
        "micro_surface_elevation": 2,
        "macro_fabm_sources": 3,
        "macro_freshwater_inputs": 4,
        "macro_update_depth": 5,
        "macro_momentum_3d": 6,
        "macro_vertical_mixing": 7,
        "macro_tracer_transport": 8,
        "post_state_update": 9,
        "post_forcing_update": 10,
    }

    def __init__(
        self,
        *args,
        initial,
        enable_mixing_diagnostics=False,
        mixing_log_interval=1,
        mixing_tracers=("salt",),
        mixing_include_micro=False,
        **kwargs,
    ):
        self.initial = initial
        self.enable_mixing_diagnostics = enable_mixing_diagnostics
        self.mixing_log_interval = max(1, int(mixing_log_interval))
        self.mixing_tracers = tuple(mixing_tracers)
        self.mixing_include_micro = mixing_include_micro
        self._mixing_macro_counter = 0
        self._mixing_records = []
        super().__init__(*args, **kwargs)

    def _get_tracer_object(self, tracer_name):
        if not hasattr(self, tracer_name):
            return None
        return getattr(self, tracer_name)

    def _tracer_moment(self, tracer_name):
        tracer = self._get_tracer_object(tracer_name)
        if tracer is None:
            return {
                "mean": np.nan,
                "variance": np.nan,
                "second_moment": np.nan,
                "volume": np.nan,
                "content": np.nan,
                "variance_content": np.nan,
            }

        values = getattr(tracer, "all_values", None)
        if values is None:
            return {
                "mean": np.nan,
                "variance": np.nan,
                "second_moment": np.nan,
                "volume": np.nan,
                "content": np.nan,
                "variance_content": np.nan,
            }

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return {
                "mean": np.nan,
                "variance": np.nan,
                "second_moment": np.nan,
                "volume": np.nan,
                "content": np.nan,
                "variance_content": np.nan,
            }

        # Prefer volume-weighted moments where possible.
        area = getattr(tracer.grid, "_area", None)
        hn = getattr(getattr(tracer.grid, "hn", None), "all_values", None)
        if area is not None and hn is not None:
            try:
                vol = hn * area
                mask = finite_mask & np.isfinite(vol) & (vol > 0.0)
                if np.any(mask):
                    w = vol[mask]
                    x = values[mask]
                    total_volume = np.sum(w)
                    mean = np.sum(w * x) / total_volume
                    second_moment = np.sum(w * x**2) / total_volume
                    variance = second_moment - mean**2
                    content = np.sum(w * x)
                    variance_content = total_volume * variance
                    return {
                        "mean": float(mean),
                        "variance": float(variance),
                        "second_moment": float(second_moment),
                        "volume": float(total_volume),
                        "content": float(content),
                        "variance_content": float(variance_content),
                    }
            except Exception:
                pass

        x = values[finite_mask]
        mean = float(np.mean(x))
        variance = float(np.var(x))
        second_moment = float(np.mean(x**2))
        sample_count = float(x.size)
        return {
            "mean": mean,
            "variance": variance,
            "second_moment": second_moment,
            "volume": sample_count,
            "content": float(np.sum(x)),
            "variance_content": float(sample_count * variance),
        }

    def _record_mixing_stage(self, stage, macro_active):
        if not self.enable_mixing_diagnostics:
            return
        if (not macro_active) and (not self.mixing_include_micro):
            return
        if macro_active and self._mixing_macro_counter % self.mixing_log_interval != 0:
            return

        for tracer_name in self.mixing_tracers:
            stats = self._tracer_moment(tracer_name)
            self._mixing_records.append(
                {
                    "time": self.time,
                    "istep": int(self.istep),
                    "macro_step": int(self._mixing_macro_counter),
                    "macro_active": bool(macro_active),
                    "stage": stage,
                    "tracer": tracer_name,
                    "mean": stats["mean"],
                    "variance": stats["variance"],
                    "second_moment": stats["second_moment"],
                    "volume": stats["volume"],
                    "content": stats["content"],
                    "variance_content": stats["variance_content"],
                }
            )

    def _build_budget_tables(self, records_df):
        df = records_df.copy()
        if df.empty:
            return {
                "budget": pd.DataFrame(),
                "stage_summary": pd.DataFrame(),
                "tidal_summary": pd.DataFrame(),
                "spring_neap_summary": pd.DataFrame(),
            }

        df["stage_order"] = df["stage"].map(self._MIXING_STAGE_ORDER).fillna(999)
        df = df.sort_values(["tracer", "macro_step", "istep", "stage_order"])

        grouped = df.groupby(["tracer", "macro_step"], sort=False)
        df["prev_stage"] = grouped["stage"].shift(1)
        df["prev_variance"] = grouped["variance"].shift(1)
        df["prev_variance_content"] = grouped["variance_content"].shift(1)
        df["ref_variance"] = grouped["variance"].transform("first")
        df["ref_variance_content"] = grouped["variance_content"].transform("first")
        df["delta_variance"] = df["variance"] - grouped["variance"].shift(1)
        df["delta_second_moment"] = (
            df["second_moment"] - grouped["second_moment"].shift(1)
        )
        df["delta_content"] = df["content"] - grouped["content"].shift(1)
        df["delta_volume"] = df["volume"] - grouped["volume"].shift(1)
        df["delta_variance_content"] = (
            df["variance_content"] - grouped["variance_content"].shift(1)
        )
        budget = df[df["prev_stage"].notna()].copy()

        eps = 1e-12
        budget["delta_variance_norm_prev"] = np.where(
            np.abs(budget["prev_variance"]) > eps,
            budget["delta_variance"] / budget["prev_variance"],
            np.nan,
        )
        budget["delta_variance_norm_ref"] = np.where(
            np.abs(budget["ref_variance"]) > eps,
            budget["delta_variance"] / budget["ref_variance"],
            np.nan,
        )
        budget["delta_variance_content_norm_ref"] = np.where(
            np.abs(budget["ref_variance_content"]) > eps,
            budget["delta_variance_content"] / budget["ref_variance_content"],
            np.nan,
        )

        budget["cum_delta_variance"] = budget.groupby(
            ["tracer", "macro_step"], sort=False
        )["delta_variance"].cumsum()
        budget["cum_delta_variance_norm_ref"] = np.where(
            np.abs(budget["ref_variance"]) > eps,
            budget["cum_delta_variance"] / budget["ref_variance"],
            np.nan,
        )

        step_totals = (
            df.groupby(["tracer", "macro_step"], as_index=False)
            .agg(
                step_initial_variance=("variance", "first"),
                step_final_variance=("variance", "last"),
                step_initial_variance_content=("variance_content", "first"),
                step_final_variance_content=("variance_content", "last"),
            )
            .assign(
                step_total_delta_variance=lambda x: x["step_final_variance"]
                - x["step_initial_variance"],
                step_total_delta_variance_content=lambda x: x[
                    "step_final_variance_content"
                ]
                - x["step_initial_variance_content"],
            )
        )
        budget = budget.merge(step_totals, on=["tracer", "macro_step"], how="left")
        budget["stage_fraction_of_step_variance_change"] = np.where(
            np.abs(budget["step_total_delta_variance"]) > eps,
            budget["delta_variance"] / budget["step_total_delta_variance"],
            np.nan,
        )
        budget["stage_fraction_of_step_variance_content_change"] = np.where(
            np.abs(budget["step_total_delta_variance_content"]) > eps,
            budget["delta_variance_content"]
            / budget["step_total_delta_variance_content"],
            np.nan,
        )

        stage_summary = budget.groupby(["tracer", "stage"], as_index=False).agg(
            mean_delta_variance=("delta_variance", "mean"),
            sum_delta_variance=("delta_variance", "sum"),
            mean_delta_variance_content=("delta_variance_content", "mean"),
            sum_delta_variance_content=("delta_variance_content", "sum"),
            mean_delta_variance_norm_prev=("delta_variance_norm_prev", "mean"),
            mean_delta_variance_norm_ref=("delta_variance_norm_ref", "mean"),
            sum_delta_variance_norm_ref=("delta_variance_norm_ref", "sum"),
            mean_delta_variance_content_norm_ref=(
                "delta_variance_content_norm_ref",
                "mean",
            ),
            sum_delta_variance_content_norm_ref=(
                "delta_variance_content_norm_ref",
                "sum",
            ),
            mean_stage_fraction_of_step_variance_change=(
                "stage_fraction_of_step_variance_change",
                "mean",
            ),
            mean_stage_fraction_of_step_variance_content_change=(
                "stage_fraction_of_step_variance_content_change",
                "mean",
            ),
            samples=("delta_variance", "size"),
        )

        tidal_summary = pd.DataFrame()
        spring_neap_summary = pd.DataFrame()
        budget["time"] = pd.to_datetime(budget["time"], errors="coerce")
        timed = budget[budget["time"].notna()].copy()
        if not timed.empty:
            t0 = timed["time"].min()
            hours = (timed["time"] - t0).dt.total_seconds() / 3600.0
            tidal_period_hours = 12.4206012
            spring_neap_window_hours = 14.765 * 24.0
            timed["tidal_cycle"] = np.floor(hours / tidal_period_hours).astype(int)
            timed["spring_neap_window"] = np.floor(
                hours / spring_neap_window_hours
            ).astype(int)

            tidal_summary = timed.groupby(
                ["tracer", "stage", "tidal_cycle"], as_index=False
            ).agg(
                mean_delta_variance=("delta_variance", "mean"),
                sum_delta_variance=("delta_variance", "sum"),
                mean_delta_variance_content=("delta_variance_content", "mean"),
                sum_delta_variance_content=("delta_variance_content", "sum"),
                mean_delta_variance_norm_ref=("delta_variance_norm_ref", "mean"),
                sum_delta_variance_norm_ref=("delta_variance_norm_ref", "sum"),
                mean_delta_variance_content_norm_ref=(
                    "delta_variance_content_norm_ref",
                    "mean",
                ),
                sum_delta_variance_content_norm_ref=(
                    "delta_variance_content_norm_ref",
                    "sum",
                ),
                samples=("delta_variance", "size"),
            )

            spring_neap_summary = timed.groupby(
                ["tracer", "stage", "spring_neap_window"], as_index=False
            ).agg(
                mean_delta_variance=("delta_variance", "mean"),
                sum_delta_variance=("delta_variance", "sum"),
                mean_delta_variance_content=("delta_variance_content", "mean"),
                sum_delta_variance_content=("delta_variance_content", "sum"),
                mean_delta_variance_norm_ref=("delta_variance_norm_ref", "mean"),
                sum_delta_variance_norm_ref=("delta_variance_norm_ref", "sum"),
                mean_delta_variance_content_norm_ref=(
                    "delta_variance_content_norm_ref",
                    "mean",
                ),
                sum_delta_variance_content_norm_ref=(
                    "delta_variance_content_norm_ref",
                    "sum",
                ),
                samples=("delta_variance", "size"),
            )

        return {
            "budget": budget,
            "stage_summary": stage_summary,
            "tidal_summary": tidal_summary,
            "spring_neap_summary": spring_neap_summary,
        }

    def write_mixing_diagnostics(self, output_path):
        if not self._mixing_records:
            self.logger.warning("No mixing diagnostics recorded; skipping write")
            return
        output_path = Path(output_path)
        records_df = pd.DataFrame(self._mixing_records)
        records_df.to_csv(output_path, index=False)

        tables = self._build_budget_tables(records_df)
        budget_path = output_path.with_name(output_path.stem + "_budget.csv")
        stage_path = output_path.with_name(output_path.stem + "_stage_summary.csv")
        tidal_path = output_path.with_name(output_path.stem + "_tidal_summary.csv")
        spring_neap_path = output_path.with_name(
            output_path.stem + "_spring_neap_summary.csv"
        )

        tables["budget"].to_csv(budget_path, index=False)
        tables["stage_summary"].to_csv(stage_path, index=False)
        if not tables["tidal_summary"].empty:
            tables["tidal_summary"].to_csv(tidal_path, index=False)
        if not tables["spring_neap_summary"].empty:
            tables["spring_neap_summary"].to_csv(spring_neap_path, index=False)

        self.logger.info("Wrote mixing diagnostics to %s", output_path)
        self.logger.info("Wrote stagewise budget table to %s", budget_path)
        self.logger.info("Wrote stage summary table to %s", stage_path)
        if not tables["tidal_summary"].empty:
            self.logger.info("Wrote tidal-cycle summary table to %s", tidal_path)
        if not tables["spring_neap_summary"].empty:
            self.logger.info(
                "Wrote spring-neap summary table to %s", spring_neap_path
            )

    def _advance_state_split(self, macro_active):
        self.momentum.advance_depth_integrated(
            self.timestep, self.tausx, self.tausy, self.dpdx, self.dpdy
        )
        self._record_mixing_stage("micro_momentum_2d", macro_active)

        self.advance_surface_elevation(
            self.timestep, self.momentum.U, self.momentum.V, self.fwf
        )
        self.T.z.halo_updaters[pygetm.parallel.Neighbor.ALL].start()

        self._int_river_flow += self.rivers.flow * self.timestep

        self.T.z.halo_updaters[pygetm.parallel.Neighbor.ALL].finish()
        self._record_mixing_stage("micro_surface_elevation", macro_active)

        if self.runtype > pygetm.RunType.BAROTROPIC_2D and macro_active:
            if self.fabm:
                self.fabm.advance(self.macrotimestep)
                self._record_mixing_stage("macro_fabm_sources", macro_active)

            self.add_freshwater_inputs(self.macrotimestep)
            self._record_mixing_stage("macro_freshwater_inputs", macro_active)

            self.update_depth(_3d=True, timestep=self.macrotimestep)
            self._record_mixing_stage("macro_update_depth", macro_active)

            self.momentum.advance(
                self.macrotimestep,
                self.split_factor,
                self.tausxo,
                self.tausyo,
                self.dpdxo,
                self.dpdyo,
                self.internal_pressure.idpdx,
                self.internal_pressure.idpdy,
                self.vertical_mixing.num,
            )
            self._record_mixing_stage("macro_momentum_3d", macro_active)

            self.vertical_mixing.advance(
                self.macrotimestep,
                self.ustar_s,
                self.momentum.ustar_b,
                self.z0s,
                self.T.z0b,
                self.NN,
                self.momentum.SS,
            )
            self._record_mixing_stage("macro_vertical_mixing", macro_active)

            self.tracers.advance(
                self.macrotimestep,
                self.momentum.uk,
                self.momentum.vk,
                self.momentum.ww,
                self.vertical_mixing.nuh,
            )
            self._record_mixing_stage("macro_tracer_transport", macro_active)

            if self.runtype == pygetm.RunType.BAROCLINIC and self.delay_slow_ip:
                self.internal_pressure.idpdx.all_values.sum(
                    axis=0, out=self.momentum.SxB.all_values
                )
                self.internal_pressure.idpdy.all_values.sum(
                    axis=0, out=self.momentum.SyB.all_values
                )

    def advance_split_with_mixing(self, check_finite=False):
        macro_updated = self.istep % self.split_factor == 0
        self.output_manager.prepare_save(macro=macro_updated)

        self.time += self.timedelta
        self.istep += 1
        macro_active = self.istep % self.split_factor == 0
        if macro_active:
            self._mixing_macro_counter += 1

        if self.report != 0 and self.istep % self.report == 0:
            self.logger.info(self.time)

        self._record_mixing_stage("pre_state_update", macro_active)
        self._advance_state_split(macro_active)
        self._record_mixing_stage("post_state_update", macro_active)

        self.input_manager.update(self.time, macro=macro_active)
        self._update_forcing_and_diagnostics(macro_active)
        self._record_mixing_stage("post_forcing_update", macro_active)

        self.output_manager.save(self.timestep * self.istep, self.istep, self.time)

        if check_finite:
            if not self.check_finite(macro_active):
                raise Exception("Non-finite values found")

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
            # self.airsea.sp.all_values = 101325.0 * ramp + self.airsea.sp.all_values * (1 - ramp)
            self.airsea.pe.all_values *= ramp
            self.rivers.flow *= ramp
        super()._update_forcing_and_diagnostics(macro_active)
class NumericalMixingMixin:
    """
    Drop-in mixin to add online numerical mixing diagnostics to any
    pygetm Simulation subclass.

    Inject BEFORE pygetm.simulation.Simulation in the MRO:

        class MyRun(NumericalMixingMixin, pygetm.simulation.Simulation):
            pass
    """
    def __init__(
        self,
        *args,
        initial,
        **kwargs,
    ):
        self.initial = initial
        super().__init__(*args, **kwargs)

    def add_nummix_tracer(self, output_interval_steps: int):
        """
        Register the salt_sq tracer with the model's TracerCollection.
        Call this AFTER the Simulation is constructed but BEFORE sim.start().

        The tracer is initialised to salt² so it is thermodynamically
        consistent from t=0.

        output_interval_steps: number of model timesteps per output write,
        e.g. 3600 for hourly output at a 1-second timestep,
            60   for hourly output at a 60-second timestep.

        """
        # Access the live salt Array from the simulation
        salt = self.salt  # pygetm.core.Array on the T-grid
        print(salt.all_values.shape)  # (nz, ny, nx)
        # Allocate backing numpy array for salt_sq, initialised to s²
        sq_data = salt.all_values ** 2  # shape (nz, ny,  nx) for a 2D slice
        print(sq_data.shape)  # (nz, ny, nx)
        # Add to the tracer collection — same advection scheme as salt
        self._salt_sq: Tracer = self.tracers.add(
            name="salt_sq",
            data=sq_data,
            long_name="Salinity squared",
            units="PSU2",
            fill_value=pygetm.constants.FILL_VALUE,
            rivers_follow_target_cell=True,   # s²_river = s²_cell (no injection)
            precipitation_follows_target_cell=False,
            molecular_diffusivity=1.1e-9,
            attrs=dict(standard_name="sea_water_absolute_salinity")
        )

        # Diagnostic arrays (same shape as salt, on T-grid)
        nz, ny, nx = salt.values.shape
        print("[nummix] allocating chi_num and chi_phy arrays of shape ", nz, ny, nx)
        self._chi_num_acc = np.zeros((nz, ny, nx), dtype=np.float64)
        self._chi_phy_acc = np.zeros((nz, ny, nx), dtype=np.float64)
        self._nummix_count = 0           # timestep counter for averaging

        # Store reference to underlying numpy views for speed
        self._salt_values   = salt.values          # live view into pygetm array
        self._sq_values     = self._salt_sq.values  # live view into salt_sq

        # Scratch arrays (avoid repeated allocation in the hot loop)
        self._s_before  = np.empty_like(self._salt_values)
        self._sq_before = np.empty_like(self._sq_values)

        # chi_num and chi_phy as pygetm Arrays for output manager
        grid = salt.grid
        print("[nummix] registering chi_num and chi_phy arrays for output... on grid ", grid.nx, grid.ny, grid.nz)
        self._chi_num = grid.array(
            z = pygetm.constants.CENTERS,
            name="chi_num",
            long_name="Numerical salinity variance dissipation rate",
            units="PSU2 s-1",
            fill_value=0.0,
            attrs={"positive": "down",
                   "comment": "Klingbeil et al. (2014) method; time-averaged"}
        )
        print("[nummix] chi_num registered with shape ", self._chi_num.shape)
        self._chi_phy = grid.array(
            z = pygetm.constants.CENTERS,
            name="chi_phy",
            long_name="Physical salinity variance dissipation rate",
            units="PSU2 s-1",
            fill_value=0.0,
            attrs={"comment": "2*nuh*(ds/dz)^2; time-averaged"}
        )
        print("[nummix] chi_phy registered with shape ", self._chi_phy.shape)
        self._output_interval_steps = output_interval_steps
        self._last_out_step = -1
        self._nummix_ready = True
        print("[nummix] salt_sq tracer registered; chi_num and chi_phy will "
              "be accumulated each timestep.")

    def add_nummix_output(self, output_file):
        """
        Register chi_num and chi_phy with an existing OutputFile so they are
        written at the same interval as the rest of the output.

        Call AFTER add_nummix_tracer() and AFTER creating your output file.
        """
        if not getattr(self, "_nummix_ready", False):
            raise RuntimeError("Call add_nummix_tracer() first.")
        output_file.request(self._chi_num, time_average=True)
        output_file.request(self._chi_phy, time_average=True)
        output_file.request("salt_sq")   # optional — s² field itself
        print("[nummix] chi_num, chi_phy, salt_sq added to output.")

    # ── Override the core timestepping method ─────────────────────────────────
    def _advance_state(self, macro_active: bool):
        # Update transports U and V from time=-1/2 to +1/2, using surface stresses and
        # pressure gradients defined at time=0
        # Inputs and outputs are on U and V grids. Stresses and pressure gradients have
        # already been updated by the call to _update_forcing_and_diagnostics at the end
        # of the previous time step.
        self.momentum.advance_depth_integrated(
            self.timestep, self.tausx, self.tausy, self.dpdx, self.dpdy
        )

        # Update surface elevation on T grid from time=0 to time=1 using transports
        # U and V at time=1/2 and freshwater fluxes at time=0. This also updates halos
        # so that depths and thicknesses can be computed everywhere without further
        # halo exchange
        self.advance_surface_elevation(
            self.timestep, self.momentum.U, self.momentum.V, self.fwf
        )
        self.T.z.halo_updaters[pygetm.parallel.Neighbor.ALL].start()

        # Track cumulative river inflow (m3) over the current macrotimestep
        self._int_river_flow += self.rivers.flow * self.timestep

        self.T.z.halo_updaters[pygetm.parallel.Neighbor.ALL].finish()

        if self.runtype > pygetm.RunType.BAROTROPIC_2D and macro_active:
            # Use previous source terms for biogeochemistry (valid for the start of the
            # current macrotimestep) to update tracers. This should be done before the
            # tracer concentrations change due to transport or rivers, as the source
            # terms are only valid for the current tracer concentrations.
            if self.fabm:
                self.fabm.advance(self.macrotimestep)

            # Update layer thicknesses and tracer concentrations to account for
            # precipitation, evaporation and river inflow between start and end of the
            # current macrotimestep.
            self.add_freshwater_inputs(self.macrotimestep)

            # Update water depth D and layer thicknesses hn on all grids.
            # On the T grid, these will be consistent with surface elevation
            # at the end of the microtimestep, that is, with the result of
            # the call to advance_surface_elevation called above.
            # Water depth and thicknesses on U/V/X grids will be
            # 1/2 MACROtimestep behind.
            # On the T grid, the previous value of surface elevation and
            # thicknesses will be stored in variables zio and ho, respectively.
            # These will thus be a full macrotimestep behind, but do account
            # for freshwater input over the past macrotimestep, as that was
            # added to surface elevation and thicknesses by the call to
            # add_freshwater_inputs above.
            self.update_depth(_3d=True, timestep=self.macrotimestep)

            # Update momentum from time=-1/2 to 1/2 of the macrotimestep, using forcing
            # defined at time=0. For this purpose, surface stresses (tausxo, tausyo)
            # and surface pressure gradients (dpdxo, dpdyo) at the end of the previous
            # macrotimestep were saved
            # Internal pressure idpdx and idpdy were calculated at the end of the
            # previous macrotimestep and are therefore ready as-is.
            self.momentum.advance(
                self.macrotimestep,
                self.split_factor,
                self.tausxo,
                self.tausyo,
                self.dpdxo,
                self.dpdyo,
                self.internal_pressure.idpdx,
                self.internal_pressure.idpdy,
                self.vertical_mixing.num,
            )

            # Update turbulent quantities (T grid - interfaces) from time=0 to
            # time=1 (macrotimestep), using surface/buoyancy-related forcing
            # (ustar_s, z0s, NN) at time=0, and bottom/velocity-related forcing
            # (ustar_b, z0b, SS) at time=1/2
            # self.T.z0b.all_values[1:, 1:] = 0.5 * (np.maximum(self.U.z0b.all_values[1:, 1:], self.U.z0b.all_values[1:, :-1]) + np.maximum(self.V.z0b.all_values[:-1, 1:], self.V.z0b.all_values[1:, :-1]))
            self.vertical_mixing.advance(
                self.macrotimestep,
                self.ustar_s,
                self.momentum.ustar_b,
                self.z0s,
                self.T.z0b,
                self.NN,
                self.momentum.SS,
            )

            # Advect and diffuse tracers. Source terms are optionally handled too,
            # as part of the diffusion update.
            self._advance_tracers(
            )

            # If we have to delay slow (2D depth-integrated) terms for internal pressure
            # by one macrotimestep, calculate them now, at the end of state update of the
            # macrotimestep, and just before the new 3D internal pressure is calculated.
            if self.runtype == pygetm.RunType.BAROCLINIC and self.delay_slow_ip:
                self.internal_pressure.idpdx.all_values.sum(
                    axis=0, out=self.momentum.SxB.all_values
                )
                self.internal_pressure.idpdy.all_values.sum(
                    axis=0, out=self.momentum.SyB.all_values
                )

    def _advance_tracers(self, ):
        """
        Intercept pygetm's internal tracer advance to sandwich the
        numerical mixing calculation around it.

        This assumes pygetm.simulation.Simulation calls self._advance_tracers()
        (or self.tracers.advance()) somewhere in its timestep.  If your pygetm
        version uses a different internal name, adjust accordingly — check
        pygetm/simulation.py for the call site.
        """
        if not getattr(self, "_nummix_ready", False):
            # Fallback: mixin not activated, behave normally
            self.tracers.advance(
                self.macrotimestep,
                self.momentum.uk,
                self.momentum.vk,
                self.momentum.ww,
                self.vertical_mixing.nuh,
)
            return

        # ── 1. Snapshot s and s² BEFORE advance ───────────────────────────
        np.copyto(self._s_before,  self._salt_values)
        np.copyto(self._sq_before, self._sq_values)

        # ── 2. Synchronise salt_sq open boundaries to s_OB² ───────────────
        #    This prevents boundary conditions from injecting spurious variance.
        #    (salt OBs are already set by pygetm before this call.)
        self._sync_sq_open_boundaries()

        # ── 3. Advance ALL tracers (including salt_sq) through advection ───
        self.tracers.advance( 
            self.macrotimestep,
            self.momentum.uk,
            self.momentum.vk,
            self.momentum.ww,
            self.vertical_mixing.nuh,
        )

        # ── 4. Compute chi_num ────────────────────────────────────────────
        #    After advance:
        #      self._salt_values  = s_after  (advected by pygetm)
        #      self._sq_values    = (s²)_after  (same scheme applied to s²)
        #    Theoretical s² if advection were perfect:
        #      sq_theoretical = s_after²
        #    Numerical mixing = variance lost by advection scheme:
        #      chi_num = (sq_theoretical - sq_advected) / dt  [PSU² s⁻¹]

        sq_theoretical = self._salt_values ** 2
        chi_num_step   = (sq_theoretical - self._sq_values) / self.macrotimestep

        # Clip to zero: chi_num should be non-negative by construction.
        # Small negatives can appear from floating-point round-off.
        np.clip(chi_num_step, 0.0, None, out=chi_num_step)

        # ── 5. Compute chi_phy ────────────────────────────────────────────
        chi_phy_step = self._compute_chi_phy()

        # ── 6. Accumulate for time-averaging ─────────────────────────────
        self._chi_num_acc  += chi_num_step
        self._chi_phy_acc  += chi_phy_step
        self._nummix_count += 1
        # flush and reset at every output interval
        if self.istep % self._output_interval_steps == 0:
            # self.logger.info("[nummix] flushing chi_num and chi_phy at istep %d", self.istep)
            self._chi_num.values[...] = self._chi_num_acc / max(self._nummix_count, 1)
            self._chi_phy.values[...] = self._chi_phy_acc / max(self._nummix_count, 1)
            self._chi_num_acc[...]  = 0.0
            self._chi_phy_acc[...]  = 0.0
            self._nummix_count      = 0
        else:
            # self.logger.info("[nummix] accumulating chi_num and chi_phy at istep %d", self.istep)
            self._chi_num.values[...] = self._chi_num_acc / max(self._nummix_count, 1)
            self._chi_phy.values[...] = self._chi_phy_acc / max(self._nummix_count, 1)
    def _sync_sq_open_boundaries(self):
        """
        Set salt_sq open boundary values to s_OB² so that the boundary
        condition for s² is thermodynamically consistent with that for s.
        Only needed if your domain has open boundaries.
        """
        try:
            # salt OB values have already been set by pygetm at this point
            salt_ob  = self.salt.open_boundaries
            sq_ob    = self._salt_sq.open_boundaries
            if salt_ob is not None and sq_ob is not None:
                # .values is the numpy array of boundary values
                sq_ob.all_values[...] = salt_ob.all_values ** 2
        except AttributeError:
            pass  # No open boundaries in this setup

    def _compute_chi_phy(self) -> np.ndarray:
        """
        Physical salinity variance dissipation rate:
            chi_phy = 2 * nuh * (ds/dz)²
        on the T-grid cell centres.

        nuh  is the vertical eddy diffusivity at layer interfaces [m² s⁻¹]
        ds/dz is approximated by finite differences at the same interfaces,
        then averaged to cell centres.

        Returns array of shape (nz, nx) in PSU² s⁻¹.
        """
        try:
            nuh_iface = self.vertical_mixing.nuh.values   # (nz+1, nx) at interfaces
        except AttributeError:
            # turbulence not yet started or attribute name differs
            return np.zeros_like(self._salt_values)

        s   = self._salt_values   # (nz, ny, nx)  at cell centres
        nz, ny, nx = s.shape

        # Layer thicknesses dz (nz, nx) — use the live grid thickness
        try:
            dz = self.T.hn.values   # (nz+1, ny, nx), positive downward
        except AttributeError:
            # Fallback: equal spacing (rough, but avoids crash)
            dz = np.full_like(s, self.T.H.mean() / nz)

        # ds/dz at interfaces k+1/2 (nz-1 interior interfaces):
        #   (s[k+1] - s[k]) / (0.5*(dz[k+1] + dz[k]))
        dz_iface   = 0.5 * (dz[:-1] + dz[1:])          # (nz-1, nx)
        ds_dz_int  = (s[1:] - s[:-1]) / np.where(dz_iface > 0, dz_iface, 1e-10)

        # Variance dissipation at interior interfaces (nz-1, nx)
        chi_iface = 2.0 * nuh_iface[1:-1] * ds_dz_int ** 2

        # Average to cell centres
        chi_phy = np.zeros_like(s)
        chi_phy[:-1] += 0.5 * chi_iface
        chi_phy[1:]  += 0.5 * chi_iface

        return chi_phy  # PSU² s⁻¹

    # ── Output flushing: called by pygetm's output manager ────────────────────

    def _flush_nummix_output(self):
        """
        Time-average the accumulated chi fields and copy into the pygetm
        Array objects so the output manager can write them.

        pygetm calls output-manager callbacks after advance; if your version
        does not call this automatically you can wire it in:

            sim.output_manager.register_callback(sim._flush_nummix_output)

        or call it manually at the end of your time loop.
        """
        if not getattr(self, "_nummix_ready", False):
            return
        if self._nummix_count > 0:
            self._chi_num.values[...] = self._chi_num_acc / self._nummix_count
            self._chi_phy.values[...] = self._chi_phy_acc / self._nummix_count
            # Reset accumulators
            self._chi_num_acc[...]  = 0.0
            self._chi_phy_acc[...]  = 0.0
            self._nummix_count      = 0
    def _reset_nummix_accumulators(self):
        """Call after each output write to start a fresh averaging window."""
        if not getattr(self, "_nummix_ready", False):
            return
        self._chi_num_acc[...]  = 0.0
        self._chi_phy_acc[...]  = 0.0
        self._nummix_count      = 0
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
            # self.airsea.sp.all_values = 101325.0 * ramp + self.airsea.sp.all_values * (1 - ramp)
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
    parser.add_argument(
        "--split-advance-for-mixing",
        action="store_true",
        help="Use explicit operator-split advance and log per-stage mixing diagnostics",
    )
    parser.add_argument(
        "--mixing-log-interval",
        type=int,
        default=1,
        help="Record diagnostics every N macro-steps when split advance is enabled",
    )
    parser.add_argument(
        "--mixing-tracers",
        type=str,
        default="salt",
        help="Comma-separated tracer names for mixing diagnostics (e.g. salt,temp)",
    )
    parser.add_argument(
        "--mixing-include-micro",
        action="store_true",
        help="Also record diagnostics at micro-step stages",
    )
    parser.add_argument(
        "--mixing-diagnostics-file",
        type=Path,
        default=None,
        help="Optional CSV output path for stagewise mixing diagnostics",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logging(args.output_data_dir / (args.estuary_name + "_run.log"))

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
    output_prefix = "adaptive_v0"
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
    # if plot period interval longer than 1 month reduce the plotting frequency to every 12 hours
    if (endtime - starttime).days > 30:
        plot_interval = datetime.timedelta(hours=12)
    else:
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
                chmin=1.0,
                vfilter=0.10,
                hfilter=0.10,
                cNN=2,
                drho=0.75,
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

    mixing_tracers = tuple(
        tracer.strip() for tracer in args.mixing_tracers.split(",") if tracer.strip()
    )
        # 2. Subclass Simulation to inject the mixin
    class TamarRun(NumericalMixingMixin, pygetm.simulation.Simulation):
        pass

    sim = TamarRun(
            domain,
            runtype=run_type,
        vertical_coordinates=vertical_coordinates,
        airsea=airsea,
        gotm=Path(".") / "gotm.yaml",
        Dcrit=0.5,
        Dmin=0.1,
        initial=True, # to apply ramping to open boundary conditions at the start of the simulation, if using restart file set to False
        fabm=args.fabm_file,
        logger=logger,
        )

    # sim = MySimulation(
    #         domain,
    #         runtype=run_type,
    #     vertical_coordinates=vertical_coordinates,
    #     airsea=airsea,
    #     gotm=Path(".") / "gotm.yaml",
    #     Dcrit=0.5,
    #     Dmin=0.1,
    #     initial=True, # to apply ramping to open boundary conditions at the start of the simulation, if using restart file set to False
    #     fabm=args.fabm_file,
    #     logger=logger,
    #     enable_mixing_diagnostics=args.split_advance_for_mixing,
    #     mixing_log_interval=args.mixing_log_interval,
    #     mixing_tracers=mixing_tracers,
    #     mixing_include_micro=args.mixing_include_micro,
    #     )
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
                # river["temp"].set(10.0)
            if sim.fabm:
                # set river fabm tracer variables 
                vars2process = ["N3_n", "N4_n", "N1_p", "N5_s",  "O3_c"] #"O3_TA",
                for var in vars2process:
                    if river_config["emorid"][var] in tamar.data_vars:
                        river[var].set(tamar[river_config["emorid"][var]])
                    else:
                        sim.logger.warning(f"Variable {var} not found in river data file")
                        # river[var].set(0.0)
                # set river fabm tracer variables to follow target cell instead of the default 0 
                vars2follow = ["O2_o", "O3_bioalk" ] #"O3_TA",
                for var in vars2follow:
                    river[var].follow_target_cell = True
    # setup airsea to constant values too
    if type(airsea) == pygetm.airsea.FluxesFromMeteo:
        if era_5_file:
            print("Using ERA5 atmospheric forcing from file: directory", era_5_file.parent, "filename", era_5_file.name)        
            # check if era5 filename has a range of years in it (len is longer than era5_1994.nc) and if so, use the wildcard to match all files in the range
            if len(era_5_file.name) < 14:
                era5_path = era_5_file.parent
                # Winds from era5 offshore are too large for the estuary... we should scale them down a bit... maybe compare them with Penlee winds from the observatory or WRF outputs from PML
                sim.airsea.u10.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "u10"))
                sim.airsea.v10.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "v10"))
                sim.airsea.sp.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "sp"))
                sim.airsea.t2m.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "t2m")) # igotm already converts to degC
                if humidity_measure == pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE:
                    sim.airsea.d2m.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "d2m")) # igotm already converts to degC
                elif humidity_measure == pygetm.HumidityMeasure.RELATIVE_HUMIDITY:
                    sim.airsea.rh.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "rh"))
                sim.airsea.tcc.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "tcc"))
                sim.airsea.tp.set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "tp") / 3600.0)
                # for river in sim.rivers.values():
                #     river["temp"].set(pygetm.input.from_nc(str(era5_path / "era5_????.nc"), "t2m") )
            else:
                era5_xr = xr.open_dataset(era_5_file, decode_times=time_coder)
                # Winds from era5 offshore are too large for the estuary... we should scale them down a bit... maybe compare them with Penlee winds from the observatory or WRF outputs from PML
                # scale down the winds by a factor of 10 for now
                sim.airsea.u10.set(era5_xr["u10"]*0.5)
                sim.airsea.v10.set(era5_xr["v10"]*0.5)
                sim.airsea.sp.set(era5_xr["sp"])
                sim.airsea.t2m.set(era5_xr["t2m"]) # igotm already converts to degC
                if humidity_measure == pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE:
                    sim.airsea.d2m.set(era5_xr["d2m"]) # igotm already converts to degC
                elif humidity_measure == pygetm.HumidityMeasure.RELATIVE_HUMIDITY:
                    sim.airsea.rh.set(era5_xr["rh"])
                sim.airsea.tcc.set(era5_xr["tcc"])
                sim.airsea.tp.set(era5_xr["tp"] / 3600.0)
                # sim.airsea.u10.set(0.0)
                # sim.airsea.v10.set(0.0)
                # for river in sim.rivers.values():
                #     river["temp"].set(era5_xr["t2m"])
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
            # file has already been post-process to have only one point in the boundary, so we can just use it directly
            cmems = xr.open_dataset(cmems_file, decode_times=time_coder)
            sim.temp.open_boundaries.type= pygetm.SPONGE
            sim.salt.open_boundaries.type= pygetm.SPONGE
            sim.temp.open_boundaries.values.set(cmems['temp'])#, on_grid = True)
            sim.salt.open_boundaries.values.set(cmems['salt'])#, on_grid = True)
            if sim.fabm:
                for varname in ["N3_n", "P1_c", "N4_n","N1_p", "N5_s"]: #"O2_o","O3_c",
                    if varname in cmems.data_vars:
                        sim[varname].open_boundaries.type= pygetm.SPONGE
                        sim[varname].open_boundaries.values.set(pygetm.input.from_nc(cmems_file, varname), on_grid=True)#, on_grid = True)
                    else:
                        sim.logger.warning(f"Variable {varname} not found in CMEMS data file")
            # set open_boundary temp and salt to constant values
            # sim.temp.open_boundaries.values.set(13.6)
            # sim.salt.open_boundaries.values.set(35.)
    # set initial zt to 1m everywhere
    # sim.zt.set(1.0)
    if sim.fabm:
        sim.fabm.get_dependency("mole_fraction_of_carbon_dioxide_in_air").set(400.0)
        sim.fabm.get_dependency("mass_concentration_of_silt").set(0.0)
    sim.logger.info("Setting up output")
    sim.output_manager.add_restart(str(Path(args.output_data_dir) / (args.estuary_name + "_" + output_prefix + "_restart.nc")), interval=datetime.timedelta(days=30), sync_interval=100)
    output = sim.output_manager.add_netcdf_file(str(Path(args.output_data_dir) / (args.estuary_name +  "_" + output_prefix + "_2d.nc")), interval=output_interval2D, sync_interval=100)
    output.request("zt", "u1", "v1","u10", "v10", "sp", "swr","shf","t2m","d2m", grid=sim.T)
    output = sim.output_manager.add_netcdf_file(str(Path(args.output_data_dir) / (args.estuary_name + "_" + output_prefix + "_3d.nc")), interval=output_interval3D, sync_interval=50)
    output.request("uk", "vk", "tke", "num", "nuh", "eps", grid=sim.T)
    if sim.runtype == pygetm.pygetm.RunType.BAROCLINIC:
        output.request("temp", "salt", "hnt", grid=sim.T)
    if sim.fabm:
        output.request("N3_n", "P1_c", "B1_c","O2_o","O3_c","N4_n","N1_p", "N5_s", "O3_TA", "O3_pCO2", "Z4_c", grid=sim.T)

    # 3. Register the salt_sq tracer BEFORE sim.start()
    print("output interval for nummix tracer:", int(output_interval3D / datetime.timedelta(seconds=timestep)))
    sim.add_nummix_tracer(output_interval_steps=int(output_interval3D / datetime.timedelta(seconds=timestep)))
    
    sim.add_nummix_output(output)          # adds chi_num, chi_phy, salt_sq
    print(sim.vertical_mixing.nuh)
    sim.logger.info("Starting simulation")
    sim.start(
        starttime,
        timestep=timestep,
        split_factor=isplit,
        report=datetime.timedelta(hours=8), # report every 8 hours
        report_totals=datetime.timedelta(days=5), # report totals every 5 days
        # profile="tamar",

    )
    while sim.time < endtime:
        if args.split_advance_for_mixing:
            sim.advance_split_with_mixing(check_finite=check_finite)
        else:
            sim.advance(check_finite=check_finite)
    sim.finish()

    if args.split_advance_for_mixing:
        if args.mixing_diagnostics_file is None:
            mixing_path = (
                Path(args.output_data_dir)
                / f"{args.estuary_name}_{output_prefix}_mixing_diagnostics.csv"
            )
        else:
            mixing_path = args.mixing_diagnostics_file
        sim.write_mixing_diagnostics(mixing_path)



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
    # plt.figure(figsize=(8, 6))
    # plt.plot(ds.zt["time"], ds.zt[:, 0, 0], label="Surface Elevation river")
    # plt.plot(ds.zt["time"], ds.zt[:, 0, -1], label="Surface Elevation open boundary")
    # plt.title("Surface Elevation at RIVER NODE")
    # plt.xlabel("Time")
    # plt.ylabel("Surface Elevation (m)   ")
    # plt.savefig(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_surface_elevation.png"), dpi=150)
    # plt.close()
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
    # index = np.argmin(np.abs(ds.distance.values - 13.8))
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
    # if type(airsea) == pygetm.airsea.FluxesFromMeteo:
    #     fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    #     axs[0, 0].plot(ds.time, ds.u10[:, 0, 0], label="Wind U10")
    #     axs[0, 0].plot(ds.time, ds.v10[:, 0, 0], label="Wind V10")
    #     axs[0, 0].set_title("Wind at 10m")
    #     axs[0, 0].set_ylabel("Wind speed (m/s)")
    #     axs[0, 0].legend()
    #     axs[0, 1].plot(ds.time, ds.t2m[:, 0, 0], label="Air Temperature")
    #     if humidity_measure == pygetm.HumidityMeasure.DEW_POINT_TEMPERATURE:
    #         axs[0, 1].plot(ds.time, ds.d2m[:, 0, 0], label="Dew Point Temperature")
    #     axs[0, 1].set_title("Air and Dew Point Temperature at 2m")
    #     axs[0, 1].set_ylabel("Temperature (°C)")
    #     axs[0, 1].legend()
    #     axs[1, 0].plot(ds.time, ds.sp[:, 0, 0], label="Surface Pressure")
    #     axs[1, 0].set_title("Surface Pressure")
    #     axs[1, 0].set_xlabel("Time")
    #     axs[1, 0].set_ylabel("Pressure (Pa)")
    #     axs[1, 0].legend()
    #     axs[1, 1].plot(ds.time, ds.swr[:, 0, 0], label="Shortwave radiation")
    #     axs[1, 1].plot(ds.time, ds.shf[:, 0, 0], label="Sensible heat flux")
    #     axs[1, 1].set_title("Shortwave Radiation and Sensible Heat Flux")
    #     axs[1, 1].set_xlabel("Time")
    #     axs[1, 1].set_ylabel("W/m^2")
    #     axs[1, 1].legend()
    #     fig.savefig(Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_meteo.png"), dpi=150)
    #     plt.close(fig)
        
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
    # animate_transect(
    #     distance2d,
    #     ds3d.zct[:, :, 0, :],
    #     ds3d.temp[:, :, 0, :],
    #     ds3d.time.values,
    #     "Temperature",
    #     units="°C",
    #     cmap=cmocean.cm.thermal,
    #     output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_temperature.mp4")
    # )
    # plt.close('all')
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
    plt.close('all')

    distance2di = np.tile(ds.distance.values, (ds3d.zft.shape[1], 1))

    animate_transect(
        distance2di,
        ds3d.zft[:, :, 0, :],
        np.log(ds3d.tke[:, :, 0, :]),
        ds3d.time.values,
        "Turbulent kinetic energy",
        units="m$^2$ s$^{-2}$",
        cmap=cmocean.cm.matter,
        output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_tke.mp4")
    )


    plt.close('all')
    if sim._nummix_ready:
        ds3d.chi_phy.min().values

        animate_transect(
            distance2d,
            ds3d.zct[:, :, 0, :],
            ds3d.salt_sq[:, :, 0, :],
            ds3d.time.values,
            "Salt squared",
            units="g$^2$ kg$^{-2}$",
            cmap=cmocean.cm.matter,
            output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_salt_squared_transect.mp4")
        )
        plt.close('all')
        animate_transect(
            distance2d,
            ds3d.zct[:, :, 0, :],
            ds3d.chi_num[:, :, 0, :],
            ds3d.time.values,
            "Numerical mixing",
            units="m$^2$ s$^{-1}$",
            cmap=cmocean.cm.matter,
            output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_numerical_mixing_transect.mp4")
        )
        plt.close('all')
        animate_transect(
            distance2d,
            ds3d.zct[:, :, 0, :],
            ds3d.chi_phy[:, :, 0, :],
            ds3d.time.values,
            "Physical mixing",
            units="m$^2$ s$^{-1}$",
            cmap=cmocean.cm.matter,
            output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_physical_mixing_transect.mp4")
        )
        plt.close('all')
    if sim.fabm:
        animate_transect(
            distance2d,
            ds3d.zct[:, :, 0, :],
            ds3d.N3_n[:, :, 0, :],
            ds3d.time.values,
            "Nitrate",
            units="mmol N/m^3",
            cmap=cmocean.cm.matter,
            output= Path(args.output_fig_dir) / (args.estuary_name + "_" + output_prefix + "_nitrate_transect.mp4")
        )
    plt.close('all')

# close ds
    ds.close()
    ds3d.close()

if __name__ == "__main__":
    main()
