from .modules import LV_FiberEstimator, LV_Speckles, LVBaseMetricsComputations
import numpy as np
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.enum_utils import check_enum
from project_heart.utils.assertions import assert_iterable

import logging
logging.basicConfig()
logger = logging.getLogger('LV')

from copy import copy

import pandas as pd


class LV(LV_FiberEstimator, LVBaseMetricsComputations):
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

    def circumferential_length(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.CIRC_LENGTH) or recompute:
            self.compute_circumferential_length(spks, **kwargs)
        return self.states.get(self.STATES.CIRC_LENGTH, t=t)
    
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

    def extract_geometrics(self, metrics, dtype=np.float64, **kwargs):
        
        import pandas as pd
        from collections import deque
        
        def assert_spks_arg(key):
            assert "spks" in metrics[key], "Metric '{}' requires 'spks' argument.".format(key)
        def assert_endo_epi_spks_arg(key):
            assert "endo_spks" in metrics[key], "Metric '{}' requires 'endo_spks' argument.".format(key)
            assert "epi_spks" in metrics[key], "Metric '{}' requires 'epi_spks' argument.".format(key)
        def execute_w_spks(fun, key):
            assert_spks_arg(key)
            args = copy(metrics[key])
            spks = args["spks"]
            args.pop("spks")
            fun(spks, **args)
        def execute_w_endo_epi_spks(fun, key):
            assert_endo_epi_spks_arg(key)
            args = copy(metrics[key])
            endo_spks = args["endo_spks"]
            epi_spks = args["epi_spks"]
            args.pop("endo_spks")
            args.pop("epi_spks")
            fun(endo_spks, epi_spks, **args)
        def resolve_add_info(df,info,all_dfs):
            if info is not None:
                all_dfs.append(info)
            else:
                all_dfs.append(df)

        valkeys = self.STATES
        
        all_dfs = deque([])

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
                search_spk_info=False, 
                search_suffix={self.REGIONS.ENDO, self.REGIONS.EPI},
                merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # radius
        key = valkeys.RADIUS.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.radius, key)
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
        key = valkeys.LONG_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.longitudinal_length, key)
            df, info = self.get_metric_as_df(key, search_spk_info=True, merged_info=True)
            resolve_add_info(df, info, all_dfs)
        # circumferential length
        key = valkeys.CIRC_LENGTH.value
        if key in metrics:
            logger.info("Extracting {}.".format(key))
            execute_w_spks(self.circumferential_length, key)
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
                info = dict(group=df.copy(), name=df.copy(), group_name=df.copy())
                info_cats = dict(name=set(), group=set(), group_name=set())
                for spk in spks:
                    info_cats[self.SPK_SETS.NAME.value].add(spk.name)
                    info_cats[self.SPK_SETS.GROUP.value].add(spk.group)
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
        from_ts=0.0, 
        to_ts=-1,
        plot_infos:set=None,
        search_suffix:set=None,
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
                    if from_ts > 0:
                        df = df.loc[df["timesteps"] >= from_ts]
                    if to_ts > 0:
                        df = df.loc[df["timesteps"] <= to_ts]
                    df.plot(x="timesteps", y=y, kind=kind, **default_args)
                else:
                    raise ValueError("info '{}' not found. "
                    "Options are: {}".format(key, list(info.keys())))
    
    def plot_longitudinal_line(self, 
                               re=False, 
                               plot_kwargs=None, 
                               window_size=None, 
                               plotter=None,
                               **kwargs):
        if window_size is None:
            window_size = (600,400)
        # set plot_kwargs as empty dict if not specified
        if plot_kwargs is None:
            plot_kwargs = dict()
        # set default plot args
        if plotter is None:
            d_plotkwargs = dict(
                
                    style='wireframe', 
                    color="gray", 
                    opacity=0.6,
                    vnodes=[
                        (
                            self.VIRTUAL_NODES.BASE, 
                            {
                                "color": "red",
                                "point_size": 600.0,
                            }
                        ),
                        (
                            self.VIRTUAL_NODES.APEX, 
                            {
                                "color": "orange",
                                "point_size": 600.0,
                            }
                        ),
                        ],
                    window_size=window_size
                )
            d_plotkwargs.update(plot_kwargs)
            # plot mesh and virtual nodes
            plotter = self.plot(re=True, **d_plotkwargs)
        # create long line for plot
        from project_heart.utils import lines_from_points
        apex = self.get_virtual_node(self.VIRTUAL_NODES.APEX)
        base = self.get_virtual_node(self.VIRTUAL_NODES.BASE)
        line = lines_from_points((apex, base))
        # set default args for long line and update if user provided
        # new args.
        d_kwargs = dict(color="cyan")
        d_kwargs.update(kwargs)
        # add line mesh
        plotter.add_mesh(line, **d_kwargs)
        # if requested, return plotter
        if re:
            return plotter
        # if not requested, show plot
        else:
            plotter.show(window_size=window_size)
        
    def plot_speckles(self, 
                        spk_args, 
                        t=0, 
                        k_bins=False,
                        add_centers=False,
                        add_k_centers=False,
                        add_la_centers=False,
                        plot_kwargs=None, 
                        centers_kwargs=None,
                        la_centers_kwargs=None,
                        k_centers_kwargs=None,
                        k_centers_as_line=False,
                        k_center_filters=None,
                        window_size=None, 
                        re=False,
                        plotter=None,
                        **kwargs):
        
        
        # Set default values
        if window_size is None:
            window_size = (600,400)
        if plot_kwargs is None:
            plot_kwargs = dict()
        if centers_kwargs is None:
            centers_kwargs = dict()
        if k_centers_kwargs is None:
            k_centers_kwargs = dict()
        if la_centers_kwargs is None:
            la_centers_kwargs = dict()
        if k_center_filters is None:
            k_center_filters = dict()
        
        if plotter is None:
            d_plotkwargs = dict(
                    style='points', 
                    color="gray", 
                    opacity=0.7,
                    window_size=window_size
                )
            d_plotkwargs.update(plot_kwargs)
            plotter = self.plot("mesh", t=t, re=True, **d_plotkwargs)
        
        # get spk data
        spk_deque = self._resolve_spk_args(spk_args) 
        spk_pts = self.get_speckles_xyz(spk_deque, t=t) # spk locations
        
        # resolve default args based on specifications
        d_kwargs = dict()
        if not add_centers and not add_k_centers:
            d_kwargs.update(
                dict(
                point_size=275,
                cmap="tab20"
                )
            )
        else:
            d_kwargs.update(
                dict(
                point_size=200,
                opacity=0.60,
                cmap="tab20"
                )
            )
        d_kwargs.update(kwargs)
        # resolze spke color palette
        if k_bins:
            klabels = spk_deque.binarize_k_clusters()
            kl_ids = spk_deque.enumerate_ids()
            bins = np.zeros(len(klabels))
            bins[kl_ids] = klabels + 1
        else:
            bins = spk_deque.binarize()
        # add speckle plot        
        plotter.add_points(spk_pts, scalars=bins, **d_kwargs)
        
        # add additional info
        if add_centers:
            centers = self.get_speckles_centers(spk_deque, t=t)
            if not add_k_centers:
                d_centers_args = dict(point_size=300,color="red")
            else:
                d_centers_args = dict(point_size=300,color="orange",opacity=0.90)
            d_centers_args.update(centers_kwargs)
            plotter.add_points(centers, **d_centers_args)
        if add_k_centers:
            k_centers = self.get_speckles_k_centers(spk_deque, t=t, **k_center_filters)
            # if not k_centers_as_line:
            d_k_centers_args = dict(point_size=300,color="blue")
            d_k_centers_args.update(k_centers_kwargs)
            plotter.add_points(k_centers, **d_k_centers_args)
        # else:
            d_k_centers_args = dict(color="magenta", width=10)
            d_k_centers_args.update(k_centers_kwargs)
            plotter.add_lines(k_centers, **d_k_centers_args)
        if add_la_centers:
            la_centers = self.get_speckles_la_centers(spk_deque, t=t)
            d_la_k_centers_args = dict(point_size=300,color="green")
            d_la_k_centers_args.update(la_centers_kwargs)
            plotter.add_points(la_centers, **d_la_k_centers_args)
        
        # resolve plotter return
        if re:
            return plotter
        else:
            plotter.show(window_size=window_size)
        
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
    
    def plot_longitudinal_distance(self, nodesets, 
                                   t=0, 
                                   nodeset_colors=None,
                                   approach="estimate_apex_base",
                                   plotter=None, 
                                   window_size=None, 
                                   plot_kwargs=None,
                                   **kwargs):
        # Set default values
        if window_size is None:
            window_size = (600,400)
        if plot_kwargs is None:
            plot_kwargs = dict()
        
        if nodeset_colors is None:
            nodeset_colors = ["blue", "green", "brown"]
        
        if plotter is None:
            d_plotkwargs = dict(
                    style='points', 
                    color="gray", 
                    opacity=0.4,
                    window_size=window_size,
                    pretty=False,
                )
            d_plotkwargs.update(plot_kwargs)
            plotter = self.plot("mesh", t=t, re=True, **d_plotkwargs)
            
            
        dists = []
        bases = []
        apexes = []
        for i, nodeset in enumerate(nodesets):
            # get node positions from nodeset at specified state
            xyz = self.states.get(self.STATES.XYZ,mask=self.get_nodeset(nodeset), t=t)
 
            if approach == "extremes":
                zs = xyz[:,2]
                min_idx = np.argmin(zs) 
                max_idx = np.argmax(zs)
                es_base = xyz[max_idx]
                es_apex = xyz[min_idx]
                dist = abs(zs[max_idx]) + abs(zs[min_idx])
                es_base[1] = 0.5*(i+1)
                es_base[0] = 0.0
                es_apex[1] = 0.5*(i+1)
                es_apex[0] = 0.0
                
            elif approach == "estimate_apex_base":
                (es_base, es_apex) = self.est_apex_and_base_refs_iteratively(xyz, **kwargs)["long_line"]
                dist = np.linalg.norm(es_base - es_apex)
            else:
                raise ValueError("Unknown approach. Avaiable approaches are: "
                                "'extremes' and 'estimate_apex_base'. Received: '{}'."
                                "Please, check documentation for further details."
                                .format(approach))
                
            plotter.add_lines(np.vstack((es_apex, es_base)), width=20, color=nodeset_colors[i])
            bases.append(es_base)
            apexes.append(es_apex)
            dists.append(dist)
        
        if approach == "extremes":            
            mean_apex = np.mean(apexes, axis=0)
            mean_base = np.mean(bases, axis=0)
            mean_base[0] = 0.0
            mean_base[1] = 0.0
            mean_apex[0] = 0.0
            mean_apex[1] = 0.0
            
            
        elif approach == "estimate_apex_base":
            mean_apex = np.mean(apexes, axis=0)
            mean_base = np.mean(bases, axis=0)
             
        plotter.add_lines(np.vstack((mean_apex, mean_base)), color="magenta")
        plotter.enable_anti_aliasing()
        
        plotter.show(window_size=window_size)
        return np.mean(dists)
                
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
    
    