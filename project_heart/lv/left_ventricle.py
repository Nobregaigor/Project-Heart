from .modules import LV_FiberEstimator, LV_Speckles, LVBaseMetricsComputations
import numpy as np
# from project_heart.enums import CONTAINER, STATES, LV_SURFS

import logging
logging.basicConfig()
logger = logging.getLogger('LV')


class LV(LV_FiberEstimator, LVBaseMetricsComputations):
    def __init__(self, *args, **kwargs):
        super(LV, self).__init__(*args, **kwargs)

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

    def longitudinal_distances(self,
                                t_es: float = None,
                                t_ed: float = 0.0,
                                **kwargs) -> float:

        # check if ls was computed
        if not self.states.check_key(self.STATES.LONG_DISTS):
            self.compute_longitudinal_distance(**kwargs)

        return self.states.get(self.STATES.LONG_DISTS, t=t_es)

    # --- Metrics that do require spks

    def radius(self, spks, t=None, recompute=False, **kwargs):
        if not self.states.check_key(self.STATES.RADIUS) or recompute:
            self.compute_radius(spks, **kwargs)
        return self.states.get(self.STATES.RADIUS, t=t) 
    
    
    def thickness(self, endo_spks, epi_spks, t=None, **kwargs):
        if not self.states.check_key(self.STATES.THICKNESS) or recompute:
            self.compute_thickness(endo_spks, epi_spks, **kwargs)
        return self.states.get(self.STATES.THICKNESS, t=t) 
        return self.states.get(self.STATES.THICKNESS, t=t) 
    
    def longitudinal_length(self, spks, t=None, **kwargs):
        key = self.STATES.LONG_LENGTH
        self._apply_generic_spk_metric_schematics(
            spks, key,
            self.compute_spk_longitudinal_length,
            t_ed=None, geochar=False, **kwargs)
        return self.states.get(key, t=t) 

    def circumferential_length(self, spks, t=None, **kwargs):
        key = self.STATES.CIRC_LENGTH
        self._apply_generic_spk_metric_schematics(
            spks, key,
            self.compute_spk_circumferential_length,
            t_ed=None, geochar=False, **kwargs)
        return self.states.get(key, t=t) 
    
    def rotation(self, spks, t=None, **kwargs):
        key = self.STATES.ROTATION
        self._apply_generic_spk_metric_schematics(
            spks, key,
            self.compute_spk_rotation,
            t_ed=None, geochar=False, **kwargs)
        return self.states.get(key, t=t) 

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
    
    def radial_shortening(self,spks,t_es: float = None, t_ed: float = 0.0,**kwargs) -> float:
        key = self.STATES.RADIAL_SHORTENING
        self._apply_generic_spk_metric_schematics(
            spks, key,
            self.compute_spk_radial_shortening,
            t_ed=t_ed, geochar=True, **kwargs)
        return self.states.get(key, t=t_es)
    
    def wall_thickening(self, endo_spks, epi_spks, t_es: float = None, t_ed: float = 0.0,**kwargs) -> float:
        key = self.STATES.WALL_THICKENING
        self._apply_generic_spk_metric_schematics(
            endo_spks, key,
            self.compute_spk_thickening,
            t_ed=t_ed, geochar=True, spks_2=epi_spks,**kwargs)
        return self.states.get(key, t=t_es) 

    def longitudinal_strain(self,spks,t_es: float = None, t_ed: float = 0.0,**kwargs) -> float:
        key = self.STATES.LONG_STRAIN
        self._apply_generic_spk_metric_schematics(
            spks, key,
            self.compute_spk_longitudinal_strain,
            t_ed=t_ed, geochar=True, **kwargs)
        return self.states.get(key, t=t_es)
    
    def circumferential_strain(self,spks,t_es: float = None, t_ed: float = 0.0,**kwargs) -> float:
        key = self.STATES.CIRC_STRAIN
        self._apply_generic_spk_metric_schematics(
            spks, key,
            self.compute_spk_circumferential_strain,
            t_ed=t_ed, geochar=True, **kwargs)
        return self.states.get(key, t=t_es)

    def twist(self, apex_spks, base_spks, t_es: float = None, t_ed: float = 0.0,**kwargs) -> float:
        key = self.STATES.TWIST
        self._apply_generic_spk_metric_schematics(
            apex_spks, key,
            self.compute_spk_twist,
            t_ed=t_ed, geochar=True, spks_2=base_spks,**kwargs)
        return self.states.get(key, t=t_es)

    def torsion(self, apex_spks, base_spks, t_es: float = None, t_ed: float = 0.0,**kwargs) -> float:
        key = self.STATES.TORSION
        self._apply_generic_spk_metric_schematics(
            apex_spks, key,
            self.compute_spk_torsion,
            t_ed=t_ed, geochar=True, spks_2=base_spks,**kwargs)
        return self.states.get(key, t=t_es) 

    # ===============================
    # Utils
    # ===============================       

    def get_metric_as_df(self, metric, search_info=True):
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
        if search_info:
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
                        try:
                            check_val = "{}_{}".format(metric, suffix)
                            data = self.states.get(check_val)
                            info[cat][check_val] = data
                        except KeyError:
                            print("suffix '{}' not found for metric '{}': {}".format(suffix, metric, check_val))
            except:
                print("Spks-data relationship not found for metric {}. "
                "Check 'set_data_spk_rel' or 'add_spk_data'.".format(metric))
        return df, info

    # ===============================
    # Plots
    # ===============================   

    def plot_metric(self, 
        metric: str, 
        kind="line", 
        from_ts=0.0, 
        to_ts=-1,
        plot_infos:list=[],
        **kwargs):
        metric = self.check_enum(metric)
        df, info = self.get_metric_as_df(metric, search_info=True)
        if from_ts > 0:
            df = df.loc[df["timesteps"] >= from_ts]
        if to_ts > 0:
            df = df.loc[df["timesteps"] <= to_ts]
        
        default_args = dict(grid=True, figsize=(15,7))
        default_args.update(kwargs)

        df.plot(x="timesteps", y=metric, kind=kind, **default_args)
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
    
    
    # def radial_shortening(self,
    #                       spks,
    #                       t_es: float = None,
    #                       t_ed: float = 0.0,
    #                       reduce_additional_info=True,
    #                       recompute=False,
    #                       **kwargs) -> float:

    #     # check if RS was computed
    #     if not self.states.check_key(self.STATES.RS) or recompute:
    #         self._compute_metric_from_spks(spks,
    #                                        self.compute_spk_radial_shortening,
    #                                        self.STATES.RS,
    #                                        t_ed=t_ed,
    #                                        **kwargs)
    #         # compute additional info 
    #         if reduce_additional_info:
    #             # reduce metric
    #             self._reduce_metric_from_spks(spks, self.STATES.RS, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(spks, self.STATES.RS, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(spks, self.STATES.RS, self.SPK_SETS.GROUP_NAME)
            
    #         # save spks used to compute this metric
    #         self.states.set_data_spk_rel(spks, self.STATES.RS)
    #         self.states.set_data_spk_rel(spks, self.metric_geochar_map[self.STATES.RS.value])

    #     return self.states.get(self.STATES.RS, t=t_es)

    # def wall_thickening(self,
    #                     endo_spks,
    #                     epi_spks,
    #                     t_es: float = None,
    #                     t_ed: float = 0.0,
    #                     reduce_additional_info=True,
    #                     recompute=False,
    #                     **kwargs) -> float:
    #     # check if wt was computed
    #     if not self.states.check_key(self.STATES.WT) or recompute:
    #         self._compute_metric_from_coupled_spks(
    #             endo_spks,
    #             epi_spks,
    #             self.compute_spk_thickening,
    #             self.STATES.WT,
    #             t_ed=t_ed,
    #             **kwargs)
    #         # compute additional info 
    #         if reduce_additional_info:
    #             self._reduce_metric_from_spks(endo_spks, self.STATES.WT, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(endo_spks, self.STATES.WT, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(endo_spks, self.STATES.WT, self.SPK_SETS.GROUP_NAME)
    #             self._reduce_metric_from_spks(epi_spks, self.STATES.WT, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(epi_spks, self.STATES.WT, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(epi_spks, self.STATES.WT, self.SPK_SETS.GROUP_NAME)
            
    #         # save spks used to compute this metric
    #         endo_spks = self._resolve_spk_args(endo_spks)
    #         epi_spks = self._resolve_spk_args(epi_spks)
    #         endo_spks.extend(epi_spks)
    #         self.states.set_data_spk_rel(endo_spks, self.STATES.WT)
    #         self.states.set_data_spk_rel(endo_spks, self.metric_geochar_map[self.STATES.WT.value])
        
    #     return self.states.get(self.STATES.WT, t=t_es)

    # def longitudinal_strain(self,
    #                         spks,
    #                         t_es: float = None,
    #                         t_ed: float = 0.0,
    #                         reduce_additional_info=True,
    #                         recompute=False,
    #                         **kwargs) -> float:
    #     # check if SL was computed
    #     if not self.states.check_key(self.STATES.SL) or recompute:
    #         self._compute_metric_from_spks(spks,
    #                                        self.compute_spk_longitudinal_strain,
    #                                        self.STATES.SL,
    #                                        t_ed=t_ed,
    #                                        **kwargs)
    #         if reduce_additional_info:
    #             self._reduce_metric_from_spks(spks, self.STATES.SL, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(spks, self.STATES.SL, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(spks, self.STATES.SL, self.SPK_SETS.GROUP_NAME)
            
    #         # save spks used to compute this metric
    #         self.states.set_data_spk_rel(spks, self.STATES.SL)
    #         self.states.set_data_spk_rel(spks, self.metric_geochar_map[self.STATES.SL.value])

    #     return self.states.get(self.STATES.SL, t=t_es)

    # def circumferential_strain(self,
    #                            spks,
    #                            t_es: float = None,
    #                            t_ed: float = 0.0,
    #                            reduce_additional_info=True,
    #                            recompute=False,
    #                            **kwargs) -> float:
    #     # check if SC was computed
    #     if not self.states.check_key(self.STATES.SC) or recompute:
    #         self._compute_metric_from_spks(spks,
    #                                        self.compute_spk_circumferential_strain,
    #                                        self.STATES.SC,
    #                                        t_ed=t_ed,
    #                                        **kwargs)

    #         if reduce_additional_info:
    #             self._reduce_metric_from_spks(spks, self.STATES.SC, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(spks, self.STATES.SC, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(spks, self.STATES.SC, self.SPK_SETS.GROUP_NAME)
            
    #         # save spks used to compute this metric
    #         self.states.set_data_spk_rel(spks, self.STATES.SC)
    #         self.states.set_data_spk_rel(spks, self.metric_geochar_map[self.STATES.SC.value])

    #     return self.states.get(self.STATES.SC, t=t_es)

    # def rotation(self,
    #              spks,
    #              t_es: float = None,
    #              t_ed: float = 0.0,
    #              reduce_additional_info=True,
    #              recompute=False,
    #              **kwargs) -> float:
    #     # check if RO was computed
    #     if not self.states.check_key(self.STATES.RO) or recompute:
    #         self._compute_metric_from_spks(spks,
    #                                        self.compute_spk_rotation,
    #                                        self.STATES.RO,
    #                                        t_ed=t_ed,
    #                                        **kwargs)

    #         if reduce_additional_info:
    #             self._reduce_metric_from_spks(spks, self.STATES.RO, self.SPK_SETS.GROUP, geochar=True)
    #             self._reduce_metric_from_spks(spks, self.STATES.RO, self.SPK_SETS.NAME, geochar=True)
    #             self._reduce_metric_from_spks(spks, self.STATES.RO, self.SPK_SETS.GROUP_NAME, geochar=True)
            
    #         # save spks used to compute this metric
    #         self.states.set_data_spk_rel(spks, self.STATES.RO)

    #     return self.states.get(self.STATES.RO, t=t_es)

    # def twist(self,
    #           apex_spks,
    #           base_spks,
    #           t_es: float = None,
    #           t_ed: float = 0.0,
    #           reduce_additional_info=True,
    #           recompute=False,
    #           **kwargs) -> float:
    #     # check if tw was computed
    #     if not self.states.check_key(self.STATES.TW) or recompute:
    #         self._compute_metric_from_coupled_spks(
    #             apex_spks,
    #             base_spks,
    #             self.compute_spk_twist,
    #             self.STATES.TW,
    #             t_ed=t_ed,
    #             **kwargs)

    #         if reduce_additional_info:
    #             self._reduce_metric_from_spks(apex_spks, self.STATES.TW, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(apex_spks, self.STATES.TW, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(apex_spks, self.STATES.TW, self.SPK_SETS.GROUP_NAME)
    #             self._reduce_metric_from_spks(base_spks, self.STATES.TW, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(base_spks, self.STATES.TW, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(base_spks, self.STATES.TW, self.SPK_SETS.GROUP_NAME)
            
    #         # save spks used to compute this metric
    #         apex_spks = self._resolve_spk_args(apex_spks)
    #         base_spks = self._resolve_spk_args(base_spks)
    #         apex_spks.extend(base_spks)
    #         self.states.set_data_spk_rel(apex_spks, self.STATES.TW)
    #         self.states.set_data_spk_rel(apex_spks, self.metric_geochar_map[self.STATES.TW.value])

    #     return self.states.get(self.STATES.TW, t=t_es)

    # def torsion(self,
    #             apex_spks,
    #             base_spks,
    #             t_es: float = None,
    #             t_ed: float = 0.0,
    #             reduce_additional_info=True,
    #             recompute=False,
    #             **kwargs) -> float:
    #     # check if TO was computed
    #     if not self.states.check_key(self.STATES.TO) or recompute:
    #         self._compute_metric_from_coupled_spks(
    #             apex_spks,
    #             base_spks,
    #             self.compute_spk_torsion,
    #             self.STATES.TO,
    #             t_ed=t_ed,
    #             **kwargs)

    #         if reduce_additional_info:
    #             self._reduce_metric_from_spks(apex_spks, self.STATES.TO, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(apex_spks, self.STATES.TO, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(apex_spks, self.STATES.TO, self.SPK_SETS.GROUP_NAME)
    #             self._reduce_metric_from_spks(base_spks, self.STATES.TO, self.SPK_SETS.GROUP)
    #             self._reduce_metric_from_spks(base_spks, self.STATES.TO, self.SPK_SETS.NAME)
    #             self._reduce_metric_from_spks(base_spks, self.STATES.TO, self.SPK_SETS.GROUP_NAME)
        
    #         # save spks used to compute this metric
    #         apex_spks = self._resolve_spk_args(apex_spks)
    #         base_spks = self._resolve_spk_args(base_spks)
    #         apex_spks.extend(base_spks)
    #         self.states.set_data_spk_rel(apex_spks, self.STATES.TO)
    #         self.states.set_data_spk_rel(apex_spks, self.metric_geochar_map[self.STATES.TO.value])

    #     return self.states.get(self.STATES.TO, t=t_es)

    