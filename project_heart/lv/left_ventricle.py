from .modules import LV_FiberEstimator, LVClinicalMetricsComputations, LV3DMetricsPlotter
import numpy as np
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.enum_utils import check_enum
from project_heart.utils.assertions import assert_iterable

import logging
logging.basicConfig()
logger = logging.getLogger('LV')

from copy import copy

import pandas as pd


class LV(LV_FiberEstimator, LVClinicalMetricsComputations, LV3DMetricsPlotter):
    def __init__(self, log_level=logging.INFO, *args, **kwargs):
        super(LV, self).__init__(log_level=log_level, *args, **kwargs)

        logger.setLevel(log_level)

    # ===============================
    # Basic Metrics
    # ===============================

    def xyz(self, **kwargs) -> np.ndarray:
        """Return pointer to nodal position [xyz] state array.

        Raises:
            RuntimeError: If 'xyz' could not be extracted or automatically computed based on current state's data. 

        Returns:
            np.ndarray (n_timesteps, 3): Pointer to 'xyz' states array.
        """
        if self.states.check_key(self.STATES.XYZ):
            return self.states.get(self.STATES.XYZ, **kwargs)
        else:
            try:
                return self.compute_xyz_from_displacement(**kwargs)
            except:
                raise RuntimeError(
                    "Could not retrieve 'xyz' data from states. Did you compute it? Check if states has 'displacement' data.")

    # ===============================
    # Geo Metrics (formely geochars)
    # ===============================

    # --- Metrics that do not need spks

    def volume(self, **kwargs):
        if self.states.check_key(self.STATES.VOLUME):
            return self.states.get(self.STATES.VOLUME)
        else:
            try:
                return self.compute_volume_based_on_nodeset(**kwargs)
            except:
                raise RuntimeError(
                    "Could not retrieve 'xyz' data from states. Did you compute it? Check if states has 'displacement' data.")

    def longitudinal_distances(self, nodesets:set=None, t: float = None, recompute=False, **kwargs) -> float:
        # check if ls was computed
        if not self.states.check_key(self.STATES.LONG_DISTS) or recompute:
            self.compute_longitudinal_distance(nodesets=nodesets, **kwargs)
        return self.states.get(self.STATES.LONG_DISTS, t=t)

    # --- Metrics that do require spks

    def radial_distance(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.RADIAL_DISTANCE) or recompute:
            self.compute_radial_distance(spks, **kwargs)
        return self.states.get(self.STATES.RADIAL_DISTANCE, t=t) 
    
    def radial_length(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.RADIAL_LENGTH) or recompute:
            self.compute_radial_length(spks, **kwargs)
        return self.states.get(self.STATES.RADIAL_LENGTH, t=t) 
    
    def thickness(self, endo_spks, epi_spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.THICKNESS) or recompute:
            self.compute_thickness(endo_spks, epi_spks, **kwargs)
        return self.states.get(self.STATES.THICKNESS, t=t) 
    
    def longitudinal_length(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.LONG_LENGTH) or recompute:
            self.compute_longitudinal_length(spks, **kwargs)
        return self.states.get(self.STATES.LONG_LENGTH, t=t) 

    def global_longitudinal_length(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.GLOBAL_LONG_LENGTH) or recompute:
            self.compute_longitudinal_length(spks, as_global=True, **kwargs)
        return self.states.get(self.STATES.GLOBAL_LONG_LENGTH, t=t) 
    
    def circumferential_length(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.CIRC_LENGTH) or recompute:
            self.compute_circumferential_length(spks, **kwargs)
        return self.states.get(self.STATES.CIRC_LENGTH, t=t)
    
    def global_circumferential_length(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.GLOBAL_CIRCUMFERENTIAL_LENGTH) or recompute:
            self.compute_circumferential_length(spks, as_global=True, **kwargs)
        return self.states.get(self.STATES.GLOBAL_CIRCUMFERENTIAL_LENGTH, t=t)
    
    def rotation(self, spks, t=None,  recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.ROTATION) or recompute:
            self.compute_rotation(spks, **kwargs)
        return self.states.get(self.STATES.ROTATION, t=t)
        # key = self.STATES.ROTATION
        # self._apply_generic_spk_metric_schematics(
        #     spks, key,
        #     self.compute_spk_rotation,
        #     t_ed=None, geochar=False, **kwargs)
        # return self.states.get(key, t=t) 

    def extract_geometrics(self, metrics, recompute=False, dtype=np.float64, **kwargs):
        
        import pandas as pd
        from collections import deque
        
        def assert_spks_arg(key):
            assert "spks" in metrics[key], "Metric '{}' requires 'spks' argument.".format(key)
        def assert_endo_epi_spks_arg(key):
            assert "endo_spks" in metrics[key], "Metric '{}' requires 'endo_spks' argument.".format(key)
            assert "epi_spks" in metrics[key], "Metric '{}' requires 'epi_spks' argument.".format(key)
        def assert_apex_base_spks_arg(key):
            assert "apex_spks" in metrics[key], "Metric '{}' requires 'apex_spks' argument.".format(key)
            assert "base_spks" in metrics[key], "Metric '{}' requires 'base_spks' argument.".format(key)
        def execute_w_spks(fun, key):
            assert_spks_arg(key)
            args = copy(metrics[key])
            spks = args["spks"]
            args.pop("spks")
            fun(spks, recompute=recompute, **args)
        def execute_w_endo_epi_spks(fun, key):
            assert_endo_epi_spks_arg(key)
            args = copy(metrics[key])
            endo_spks = args["endo_spks"]
            epi_spks = args["epi_spks"]
            args.pop("endo_spks")
            args.pop("epi_spks")
            fun(endo_spks, epi_spks, recompute=recompute, **args)
        def execute_w_apex_base_spks(fun, key):
            assert_apex_base_spks_arg(key)
            args = copy(metrics[key])
            apex_spks = args["apex_spks"]
            base_spks = args["base_spks"]
            args.pop("apex_spks")
            args.pop("base_spks")
            fun(apex_spks, base_spks, recompute=recompute, **args)
        def resolve_add_info(df,info,all_dfs):
            if info is not None:
                all_dfs.append(info)
            else:
                all_dfs.append(df)

        valkeys = self.STATES
        
        all_dfs = deque([])
        
        # compute apex and base over timesteps
        key = self.STATES.APEX_BASE_OVER_TIMESTEPS.value
        if key in metrics:
            logger.info("Computing {}.".format(key))
            execute_w_apex_base_spks(self.compute_base_apex_ref_over_timesteps, key)
        # volume
        key = valkeys.VOLUME.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            self.volume(**metrics[key])
            df, _ = self.get_metric_as_df(key, search_spk_info=False)
            all_dfs.append(df)
        # longitudinal distances
        key = valkeys.LONG_DISTS.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            self.longitudinal_distances(**metrics[key])
            df, info = self.get_metric_as_df(key, 
                search_spk_info=True, 
                search_suffix={self.REGIONS.ENDO, self.REGIONS.EPI},
                merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # radius 1
        key = valkeys.RADIAL_DISTANCE.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.radial_distance, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # radius 2
        key = valkeys.RADIAL_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.radial_length, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # thickness
        key = valkeys.WALL_THICKNESS.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_endo_epi_spks(self.thickness, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # longitudinal length
        key = valkeys.LONGITUDINAL_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.longitudinal_length, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # global longitudinal length
        key = valkeys.GLOBAL_LONGITUDINAL_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.global_longitudinal_length, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # circumferential length
        key = valkeys.CIRCUMFERENTIAL_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.circumferential_length, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # global circumferential length
        key = valkeys.GLOBAL_CIRCUMFERENTIAL_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.global_circumferential_length, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # rotation
        key = valkeys.ROTATION.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.rotation, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        
        merged = pd.concat(all_dfs, axis=1)
        merged = merged.loc[:,~merged.columns.duplicated()]
        return merged.astype(dtype)


    # ===============================
    # Clinical Metrics
    # ===============================

    # --- Metrics that do not need spks

    def ejection_fraction(self,
                          t_es: float = None,
                          t_ed: float = 0.0,
                          **kwargs) -> float:
        """Calculates the change in volume between end-systole and end-diatole.
            ES and ED are timesteps where the ES and ED occur.

        Args:
            t_es (float or None, optional): End-systole timestep. Defaults to None.
            t_ed (float, optional): End-diastole timestep. Defaults to 0.0.

        Returns:
            float (if t_es is specified): Ejection fraction (%) at t_es.
            np.ndarray (n_timesteps, 1): Volumetric fraction array.
        """
        # check if vf was computed
        if not self.states.check_key(self.STATES.VF):
            self.compute_volumetric_fraction(t_ed=t_ed, **kwargs)

        return self.states.get(self.STATES.VF, t=t_es)

    def longitudinal_shortening(self,
                                t_es: float = None,
                                t_ed: float = 0.0,
                                **kwargs) -> float:

        # check if ls was computed
        if not self.states.check_key(self.STATES.LS):
            self.compute_longitudinal_shortening(t_ed=t_ed, **kwargs)

        return self.states.get(self.STATES.LS, t=t_es)

    # --- Metrics that do require spks
    
    def radial_shortening(self,spks, t_es: float = None, t_ed: float = 0.0, recompute=False, **kwargs) -> float:
        # make sure we have computed required geometric data for all spks
        self.radius(spks, t=None, recompute=recompute, **kwargs)
        # check if metric needs to be computed
        if not self.states.check_key(self.STATES.RADIAL_SHORTENING) or recompute:
            return self.compute_radial_shortening(spks, t_ed=t_ed, **kwargs)
        return self.states.get(self.STATES.RADIAL_SHORTENING, t=t_es)
    
    def wall_thickening(self, endo_spks, epi_spks, t_es: float = None, t_ed: float = 0.0, recompute=False, **kwargs) -> float:
        # make sure we have computed required geometric data for all spks
        self.thickness(endo_spks, epi_spks, t=None, recompute=recompute, **kwargs)
        # check if metric needs to be computed
        if not self.states.check_key(self.STATES.THICKENING) or recompute:
            return self.compute_thicknening(endo_spks, epi_spks, t_ed=t_ed, **kwargs)
        return self.states.get(self.STATES.THICKENING, t=t_es)

    def longitudinal_strain(self, spks, t_es: float = None, t_ed: float = 0.0, recompute=False, **kwargs) -> float:
        # make sure we have computed required geometric data for all spks
        self.longitudinal_length(spks, t=None, recompute=recompute, **kwargs)
        # check if metric needs to be computed
        if not self.states.check_key(self.STATES.LONGITUDINAL_STRAIN) or recompute:
            return self.compute_longitudinal_strain(spks, t_ed=t_ed, **kwargs)
        return self.states.get(self.STATES.LONGITUDINAL_STRAIN, t=t_es)
    
    def circumferential_strain(self, spks, t_es: float = None, t_ed: float = 0.0, recompute=False, **kwargs) -> float:
        # make sure we have computed required geometric data for all spks
        self.circumferential_length(spks, t=None, recompute=recompute, **kwargs)
        # check if metric needs to be computed
        if not self.states.check_key(self.STATES.CIRCUMFERENTIAL_STRAIN) or recompute:
            return self.compute_circumferential_strain(spks, t_ed=t_ed, **kwargs)
        return self.states.get(self.STATES.CIRCUMFERENTIAL_STRAIN, t=t_es)

    def twist(self, apex_spks, base_spks, t_es: float = None, t_ed: float = 0.0, recompute=False, **kwargs) -> float:
        # make sure we have computed required geometric data for all spks
        self.rotation(apex_spks, t=None, recompute=recompute, **kwargs)
        self.rotation(base_spks, t=None, recompute=recompute, **kwargs)
        if not self.states.check_key(self.STATES.TWIST) or recompute:
            self.compute_twist(apex_spks, base_spks, t_ed=t_ed, **kwargs)
        return self.states.get(self.STATES.TWIST, t=t_es)

    def torsion(self, apex_spks, base_spks, t_es: float = None, t_ed: float = 0.0, recompute=False, **kwargs) -> float:
        # make sure we have computed required geometric data for all spks
        self.rotation(apex_spks, t=None, recompute=recompute, **kwargs)
        self.rotation(base_spks, t=None, recompute=recompute, **kwargs)
        if not self.states.check_key(self.STATES.TORSION) or recompute:
            self.compute_torsion(apex_spks, base_spks, t_ed=t_ed, **kwargs)
        return self.states.get(self.STATES.TORSION, t=t_es)

    # ===============================
    # Utils
    # ===============================       

    def get_metric_as_df(self, metric, 
            search_spk_info=True, 
            search_suffix:set=None,
            merged_info=False
            ):
        # check for arguments
        assert_iterable(search_suffix)
        
        import pandas as pd
        metric = self.check_enum(metric)
        if not self.states.check_key(metric):
            raise ValueError(
                "Metric '{}' not found in states." \
                "Did you compute it?".format(metric))

        ts = self.states.timesteps
        re = self.states.get(metric) # get main metric
        df = pd.DataFrame({"timesteps": ts, metric: re})
        info = None
        if search_spk_info:
            logger.debug("Initiate search for spk info.")
            try:
                spks = self.states.get_spks_from_data(metric)
                info = dict(group=df.copy(), name=df.copy(), subset=df.copy(), group_name=df.copy())
                info_cats = dict(name=set(), group=set(), subset=set(), group_name=set())
                for spk in spks:
                    info_cats[self.SPK_SETS.NAME.value].add(spk.name)
                    info_cats[self.SPK_SETS.GROUP.value].add(spk.group)
                    info_cats[self.SPK_SETS.SUBSET.value].add(spk.subset)
                    info_cats[self.SPK_SETS.GROUP_NAME.value].add("{}_{}".format(spk.group, spk.name))
                for cat, values in info_cats.items():
                    for suffix in values:
                        logger.debug("Searching for spk data with suffix '{}'".format(suffix))
                        try:
                            check_val = "{}_{}".format(metric, suffix)
                            data = self.states.get(check_val)
                            info[cat][check_val] = data
                        except KeyError:
                            logger.debug("Speckle suffix '{}' not found for metric '{}': {}".format(suffix, metric, check_val))
            except:
                logger.debug("Spks-data relationship not found for metric {}. "
                "Check 'set_data_spk_rel' or 'add_spk_data'.".format(metric))
        if search_suffix is not None:
            logger.debug("Initiate search for suffix info.")
            if info is None:
                info = dict(suffix=df.copy())
            else:
                info.update(suffix=df.copy())
            for key in search_suffix:
                name, value = check_enum(key)
                name = name.lower()
                logger.debug("Searching for suffix '{}' with value '{}'".format(name, value))
                try:
                    check_val = "{}_{}".format(metric, value)
                    save_val = "{}_{}".format(metric, name)
                    data = self.states.get(check_val)
                    info["suffix"][save_val] = data
                except KeyError:
                    logger.debug("Suffix '{}' not found for metric '{}': {}".format(key, metric, check_val))     
        
        if not merged_info or info is None:
            return df, info
        else:
            merged = pd.concat(info.values(), axis=1)
            merged = merged.loc[:,~merged.columns.duplicated()]
            return df, merged
    

    # ===============================
    # Plots
    # ===============================   

    def plot_metric(self, 
        metric: str, 
        kind="line", 
        from_ts:float=0.0, 
        to_ts:float=-1,
        plot_infos:set=None,
        search_suffix:set=None,
        plot_infos_only:bool=False,
        plot_infos_only_on_info_plots:bool=False,
        **kwargs):
        # Check input arguments
        assert_iterable(plot_infos, accept_none=True)
        assert_iterable(search_suffix, accept_none=True)
        # set function values
        metric = self.check_enum(metric)
        search_spk_info = True if plot_infos is not None else False
        # get metric data
        df, info = self.get_metric_as_df(metric, 
                    search_spk_info=search_spk_info, 
                    search_suffix=search_suffix)
        # decide what timesteps to plot
        if from_ts > 0:
            df = df.loc[df["timesteps"] >= from_ts]
        if to_ts > 0:
            df = df.loc[df["timesteps"] <= to_ts]
        # make plot
        default_args = dict(grid=True, figsize=(15,7))
        default_args.update(kwargs)
        if not plot_infos_only:
            df.plot(x="timesteps", y=metric, kind=kind, **default_args)
        # resolve info plots
        if plot_infos is None and search_suffix is not None:
            plot_infos = ["suffix"]
        elif plot_infos is not None and search_suffix is not None:
            plot_infos.append("suffix")
        # plot additional data if requested
        if info is not None and len(plot_infos) > 0:
            for key in plot_infos:
                if key in info:
                    df = info[key]
                    y = list(df.columns)
                    y.remove("timesteps")
                    if plot_infos_only_on_info_plots and len(y) > 1:
                        y.remove(metric)
                    if from_ts > 0:
                        df = df.loc[df["timesteps"] >= from_ts]
                    if to_ts > 0:
                        df = df.loc[df["timesteps"] <= to_ts]
                    df.plot(x="timesteps", y=y, kind=kind, **default_args)
                else:
                    raise ValueError("info '{}' not found. "
                    "Options are: {}".format(key, list(info.keys())))
    
    def plot_speckles_radial_distances(self, spk_args, t=0, approach="moving_vector", window_size=None, **kwargs):
        from project_heart.utils.spatial_utils import project_pt_on_line
        
        if window_size is None:
            window_size = (600,400)

        # plot speckles
        plotter = self.plot_speckles(spk_args, cmap="tab20", categories=True, re=True, t=t, **kwargs)
        
        if approach=="fixed_vector":
            apex_ts = self.states.get(self.STATES.APEX_REF, t=0)
            base_ts = self.states.get(self.STATES.BASE_REF, t=0)
        else:
            apex_ts = self.states.get(self.STATES.APEX_REF, t=t)
            base_ts = self.states.get(self.STATES.BASE_REF, t=t)
        
        spk_deque = self._resolve_spk_args(spk_args)
        line_lengths = []
        for spk in spk_deque:
            spk_xyz = self.states.get(self.STATES.XYZ, mask=spk.ids, t=t)

            for pt in spk_xyz:
                p_pt = project_pt_on_line(pt, apex_ts, base_ts)
                line_lengths.append(np.linalg.norm(pt - p_pt))
                plotter.add_lines(np.vstack((pt, p_pt)), color="magenta")
            
            plotter.add_lines(np.vstack((apex_ts, base_ts)), color="cyan")
            plotter.add_points(np.vstack((apex_ts, base_ts)), color="purple", point_size=200)

        plotter.show(window_size=window_size)
        
        return np.mean(line_lengths)
       
    def plot_speckles_radial_lengths(self, spk_args, t=0, approach="moving_centers", window_size=None, **kwargs):
        
        if window_size is None:
            window_size = (600,400)

        # plot speckles
        plotter = self.plot_speckles(spk_args, cmap="tab20", categories=True, re=True, t=t, **kwargs)
        
        if approach=="fixed_centers":
            apex_ts = self.states.get(self.STATES.APEX_REF, t=0)
            base_ts = self.states.get(self.STATES.BASE_REF, t=0)
        else:
            apex_ts = self.states.get(self.STATES.APEX_REF, t=t)
            base_ts = self.states.get(self.STATES.BASE_REF, t=t)
        
        spk_deque = self._resolve_spk_args(spk_args)
        line_lengths = []
        for spk in spk_deque:
            spk_xyz = self.states.get(self.STATES.XYZ, mask=spk.ids, t=t)
            if approach=="fixed_centers":
                spk_la_center = self.get_speckles_la_centers(spk_args, t=0)
            else:
                spk_la_center = self.get_speckles_la_centers(spk_args, t=t)

            for pt in spk_xyz:
                line_lengths.append(np.linalg.norm(pt - spk_la_center))
                plotter.add_lines(np.vstack((pt, spk_la_center)), color="magenta")
            
            plotter.add_lines(np.vstack((apex_ts, base_ts)), color="cyan")
            plotter.add_points(np.vstack((apex_ts, base_ts)), color="purple", point_size=200)

        plotter.show(window_size=window_size)
        
        return np.mean(line_lengths)
    
    def plot_speckles_wall_thickness_rd(self, endo_spk_args, epi_spk_args, t=0, approach="moving_vector", window_size=None, **kwargs):
        
        from project_heart.utils.spatial_utils import project_pt_on_line
        from project_heart.utils.vector_utils import unit_vector
        
        
        if window_size is None:
            window_size = (600,400)

        # plot speckles
        plotter = self.plot_speckles(endo_spk_args, cmap="tab20", categories=True, re=True, t=t, **kwargs)
        plotter = self.plot_speckles(epi_spk_args, cmap="tab20", categories=True, re=True, t=t, plotter=plotter, **kwargs)
        
        
        if approach=="fixed_vector":
            apex_ts = self.states.get(self.STATES.APEX_REF, t=0)
            base_ts = self.states.get(self.STATES.BASE_REF, t=0)
        else:
            apex_ts = self.states.get(self.STATES.APEX_REF, t=t)
            base_ts = self.states.get(self.STATES.BASE_REF, t=t)
        
        endo_spk_deque = self._resolve_spk_args(endo_spk_args)
        epi_spk_deque = self._resolve_spk_args(epi_spk_args)
        
        line_lengths = []
        for endo_spk, epi_spk in zip(endo_spk_deque, epi_spk_deque):
            endo_spk_xyz = self.states.get(self.STATES.XYZ, mask=endo_spk.ids, t=t)
            epi_spk_xyz = self.states.get(self.STATES.XYZ, mask=epi_spk.ids, t=t)
            
            for endo_pt, epi_pt in zip(endo_spk_xyz, epi_spk_xyz):
                endo_p_pt = project_pt_on_line(endo_pt, apex_ts, base_ts)
                epi_p_pt = project_pt_on_line(epi_pt, apex_ts, base_ts)
                
                endo_vec = endo_p_pt - endo_pt
                epi_vec = epi_p_pt - epi_pt
                                
                endo_len = np.linalg.norm(endo_vec)
                epi_len = np.linalg.norm(epi_vec)
                
                thickness = epi_len - endo_len
                
                plotter.add_lines(np.vstack((endo_pt, endo_p_pt)), color="orange")
                plotter.add_lines(np.vstack((epi_pt, epi_p_pt)), color="lightgreen")
                plotter.add_lines(np.vstack((epi_pt, epi_pt+(thickness*unit_vector(epi_vec)) )), color="magenta")
                          
                line_lengths.append(thickness)
            
            plotter.add_lines(np.vstack((apex_ts, base_ts)), color="cyan")
            plotter.add_points(np.vstack((apex_ts, base_ts)), color="purple", point_size=200)

        plotter.show(window_size=window_size)
        
        return np.mean(line_lengths)
    
    def plot_speckles_wall_thickness_rl(self, endo_spk_args, epi_spk_args, t=0, approach="moving_centers", window_size=None, **kwargs):
        
        from project_heart.utils.spatial_utils import project_pt_on_line
        from project_heart.utils.vector_utils import unit_vector
        
        
        if window_size is None:
            window_size = (600,400)

        # plot speckles
        plotter = self.plot_speckles(endo_spk_args, cmap="tab20", categories=True, re=True, t=t, **kwargs)
        plotter = self.plot_speckles(epi_spk_args, cmap="tab20", categories=True, re=True, t=t, plotter=plotter, **kwargs)
        
        
        if approach=="fixed_vector":
            apex_ts = self.states.get(self.STATES.APEX_REF, t=0)
            base_ts = self.states.get(self.STATES.BASE_REF, t=0)
        else:
            apex_ts = self.states.get(self.STATES.APEX_REF, t=t)
            base_ts = self.states.get(self.STATES.BASE_REF, t=t)
        
        endo_spk_deque = self._resolve_spk_args(endo_spk_args)
        epi_spk_deque = self._resolve_spk_args(epi_spk_args)
        
        line_lengths = []
        for endo_spk, epi_spk in zip(endo_spk_deque, epi_spk_deque):
            endo_spk_xyz = self.states.get(self.STATES.XYZ, mask=endo_spk.ids, t=t)
            epi_spk_xyz = self.states.get(self.STATES.XYZ, mask=epi_spk.ids, t=t)
            
            if approach=="fixed_centers":
                endo_spk_la_center = self.get_speckles_la_centers(endo_spk_args, t=0)
                epi_spk_la_center = self.get_speckles_la_centers(epi_spk_args, t=0)
            else:
                endo_spk_la_center = self.get_speckles_la_centers(endo_spk_args, t=t)
                epi_spk_la_center = self.get_speckles_la_centers(endo_spk_args, t=t)
                                            
            for endo_pt, epi_pt in zip(endo_spk_xyz, epi_spk_xyz):
                epi_p_pt = project_pt_on_line(epi_pt, apex_ts, base_ts)
                
                endo_vec = endo_spk_la_center - endo_pt
                epi_vec = epi_spk_la_center - epi_pt
                                
                endo_len = np.linalg.norm(endo_vec)
                epi_len = np.linalg.norm(epi_vec)
                
                thickness = epi_len - endo_len
                
                plotter.add_lines(np.vstack((endo_pt, endo_spk_la_center)), color="orange")
                plotter.add_lines(np.vstack((epi_pt, epi_spk_la_center)), color="lightgreen")
                
                straight_vec = epi_p_pt - epi_pt
                vec_dir = np.mean([epi_vec[0], straight_vec], axis=0)
                
                plotter.add_lines(np.vstack((epi_pt, epi_pt+(thickness*unit_vector(vec_dir)) )), color="magenta")
                          
                line_lengths.append(thickness)
            
            plotter.add_lines(np.vstack((apex_ts, base_ts)), color="cyan")
            plotter.add_points(np.vstack((apex_ts, base_ts)), color="purple", point_size=200)

        plotter.show(window_size=window_size)
        
        return np.mean(line_lengths)
    
    # ===============================
    # exports to FEA solvers
    # ===============================   

    def to_feb_template(self, template_path, filepath, 
            mat=1, bcmat=2, log_level=logging.INFO):
        logger.setLevel(log_level)
        logger.warn("This method is not yet finalized.")
        from febio_python.feb import FEBio_feb
        from project_heart.enums import LV_RIM

        # ----------------------
        # Read template file
        logger.debug("Loading template: {}".format(template_path))
        feb = FEBio_feb.from_file(template_path)

        # ----------------------
        # add nodes and elements
        logger.debug("Checking for Nodes and Elements...")
        nodes = feb.geometry().find("Nodes")
        if nodes is None:
            logger.debug("No nodes found. Adding 'LV' nodes.")
            feb.add_nodes([
                {
                    "name": "LV", 
                    "nodes": np.round(self.nodes(), 5),
                }
            ])
            logger.debug("No nodes found. Adding 'LV' elements.")
            for elements in self.cells().values():
                feb.add_elements([
                    {
                    "name": "LV", 
                    "mat": str(mat),
                    "elems": elements + 1
                    }
                ])
        else:
            logger.debug("Some nodes found. Appending 'LV' nodes.")
            if nodes.find("LV") is None:
                feb.add_nodes([
                    {
                        "name": "LV", 
                        "nodes": np.round(self.nodes(), 5),
                    }
                ])
            logger.debug("Some nodes found. Appending 'LV' elements.")
            for elements in self.cells().values():
                feb.add_elements([
                    {
                    "name": "LV", 
                    "mat": str(mat),
                    "elems": elements + 1
                    }
                ])
        
        # ----------------------
        # add nodesets
        logger.debug("Checking for Nodesets...")
        feb.add_nodesets(self.get_nodesets_from_enum(self.REGIONS))
        
        # ----------------------
        # Add Surfaces
        logger.debug("Checking for Surfaces...")
        surfs_to_add = {}
        for sname, surfoi in self._surfaces_oi.items():
            try:
                sname = self.REGIONS(sname).name
            except:
                logger.debug("Could not determine name for surface "
                " of interest. Will use surface id (int) as string value")
                sname = str(sname)
            logger.debug("Adding surface_oi  '{}'".format(sname))
            surfs_to_add[sname] = np.vstack(surfoi) + 1
        if len(surfs_to_add) > 0:
            feb.add_surfaces(surfs_to_add)

        # ----------------------
        # bcs
        logger.debug("Checking for Boundary conditions...")
        node_offset = self.mesh.n_points
        for bcname, vcvals in self._bcs.items():
            logger.debug("Adding BC  '{}'".format(bcname))
            bcnodes = np.round(vcvals[1]["RIM_NODES"], 5)
            feb.add_nodes([
                {"name": bcname, 
                "nodes": bcnodes
                },
            ], initial_el_id=node_offset+1)
            feb.add_elements([
                {
                    "name": bcname, 
                    "type": "quad4",
                    "mat": str(bcmat),
                    "elems": vcvals[1][LV_RIM.ELEMENTS.value] + node_offset + 1 # adjust element ids
                }
            ], initial_el_id=node_offset+1)
            node_offset += len(bcnodes)
        
        # ----------------------
        # Add Discrete sets
        logger.debug("Checking for dicrete sets...")
        adjusted_discrete_sets = {}
        to_adj = self.mesh.n_points
        for key, values in self._discrete_sets.items():
            logger.debug("Discrete set to add: '{}'".format(key))
            adj_vals = np.copy(values) + 1
            adj_vals[:, 1] += to_adj
            adjusted_discrete_sets[key] = adj_vals
            to_adj += len(self.get_bc(key)[1]["RIM_NODES"]) # account for nodes
            
        if len(adjusted_discrete_sets) > 0:
            feb.add_discretesets(adjusted_discrete_sets)
            import xml.etree.ElementTree as ET
            discrete = feb.discrete()
            for key in adjusted_discrete_sets.keys():
                subel = ET.SubElement(discrete, "discrete")
                subel.set("discrete_set", key)
                subel.set("dmat","1")

        # ----------------------
        # Add fibers
        try:
            logger.debug("Checking for fibers...")
            feb.add_meshdata([
            {
                "elem_set": "LV", 
                "var": "mat_axis",
                "elems": {
                "a": self.get(self.CONTAINERS.MESH_CELL_DATA, self.FIBERS.F0),
                "d": self.get(self.CONTAINERS.MESH_CELL_DATA, self.FIBERS.S0),
                }
                }
            ])
            logger.debug("Fiber data successfully added.")
        except KeyError:
            logger.info("No Fiber data found. Add mesh data will be ignored.")
        
        logger.debug("Writing file...")
        feb.write(filepath)
        logger.debug("File written successfully.")
    
    