
import numpy as np
from .lv_metrics_geometrics_computations import LVGeometricsComputations
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.spatial_utils import centroid, radius
from project_heart.modules.speckles.speckle import Speckle, SpeckeDeque
from project_heart.utils.spatial_utils import compute_longitudinal_length, compute_circumferential_length, compute_length_by_clustering
from project_heart.utils.vector_utils import angle_between
from project_heart.utils.enum_utils import add_to_enum
import logging

logger = logging.getLogger('LV.BaseMetricsComputations')

from collections import deque


class LVClinicalMetricsComputations(LVGeometricsComputations):
    def __init__(self, log_level=logging.INFO, *args, **kwargs):
        super(LVClinicalMetricsComputations, self).__init__(log_level=log_level, *args, **kwargs)
        # self.metric_geochar_map = {
        #     self.STATES.LONGITUDINAL_SHORTENING.value: self.STATES.LONGITUDINAL_DISTANCE.value,
        #     self.STATES.RADIAL_SHORTENING.value: self.STATES.RADIUS.value,
        #     self.STATES.WALL_THICKENING.value: self.STATES.THICKNESS.value,
        #     self.STATES.LONG_STRAIN.value: self.STATES.LONG_LENGTH.value,
        #     self.STATES.CIRC_STRAIN.value: self.STATES.CIRC_LENGTH.value,
        #     self.STATES.TWIST.value: self.STATES.ROTATION.value,
        #     self.STATES.TORSION.value: self.STATES.ROTATION.value,
        #     self.STATES.ROTATION.value: self.STATES.ROTATION.value # required just for spk computation
        # }
        # self.compute_metrics_plot_infos = compute_metrics_plot_infos
        # self.metric

        logger.setLevel(log_level)

    # =============================
    # Fundamental computations
    # =============================

    # ---- Nodal position over timesteps (xyz data)

    def compute_volumetric_fraction(self,  t_ed: float = 0.0, **kwargs):
        if not self.states.check_key(self.STATES.VOLUME):
            try:
                self.compute_volume_based_on_nodeset(**kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute volumes. Please, either verify required data or add state data for 'VOLUME' manually.")
        vol1 = self.states.get(self.STATES.VOLUME)
        vol2 = self.states.get(self.STATES.VOLUME, t=t_ed)
        # compute % shortening
        vf = (vol2 - vol1) / (vol2 + self.EPSILON) * 100.0
        self.states.add(self.STATES.VF, vf)  # save to states
        return self.states.get(self.STATES.VF)  # return pointer

    # ---- Longitudinal shortening

    def compute_longitudinal_shortening(self, t_ed: float = 0.0, **kwargs) -> np.ndarray:
        if not self.states.check_key(self.STATES.LONG_DISTS):
            try:
                self.compute_longitudinal_distance(**kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute londitudinal distances. Please, either verify required data or add state data for 'LONG_DISTS' manually.")

        d2 = self.states.get(self.STATES.LONG_DISTS)
        d1 = self.states.get(self.STATES.LONG_DISTS, t=t_ed)
        # compute % shortening
        ls = (d1 - d2) / (d1 + self.EPSILON) * 100.0
        self.states.add(self.STATES.LS, ls)  # save to states
        return self.states.get(self.STATES.LS)  # return pointer

    def compute_radial_shortening(self, spks, t_ed=0.0, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.RADIAL_SHORTENING
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for all spks
        res = [self._compute_spk_relative_error(spk, 
                    self.STATES.RADIAL_DISTANCE, key,
                    t_ed=t_ed, reduce_by=reduce_by, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean data across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(spks, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)
    
    def compute_radial_strain(self, spks, t_ed=0.0, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.RADIAL_SHORTENING
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for all spks
        res = [self._compute_spk_relative_error(spk, 
                    self.STATES.RADIAL_LENGTH, key,
                    t_ed=t_ed, reduce_by=reduce_by, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean data across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(spks, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    def compute_thicknening(self, endo_spks, epi_spks, t_ed=0.0, reduce_by={"name"}, **kwargs):
        # set key for this function
        key = self.STATES.WALL_THICKENING
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        endo_spks = self._resolve_spk_args(endo_spks)
        epi_spks = self._resolve_spk_args(epi_spks)
        # compute metric for all spks
        res = [self._compute_spk_relative_error(spk, 
                    self.STATES.WALL_THICKNESS, key,
                    t_ed=t_ed, reduce_by=reduce_by, 
                    switch_es=True, **kwargs) for spk in endo_spks]
        _ = [self._compute_spk_relative_error(spk, 
                    self.STATES.WALL_THICKNESS, key,
                    t_ed=t_ed, reduce_by=reduce_by, 
                    switch_es=True, **kwargs) for spk in epi_spks]
        # reduce metric (here we compute the mean data across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(endo_spks, key)
        self.states.set_data_spk_rel(epi_spks, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(endo_spks, key, **kwargs)
            self._reduce_metric_by_group(epi_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(endo_spks, key, **kwargs)
            self._reduce_metric_by_name(epi_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(endo_spks, key, **kwargs)
            self._reduce_metric_by_name(epi_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(endo_spks, key, **kwargs)
            self._reduce_metric_by_group_and_name(epi_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    def compute_longitudinal_strain(self, spks, t_ed=0.0, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.LONGITUDINAL_STRAIN
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for all spks
        res = [self._compute_spk_relative_error(spk, 
                    self.STATES.LONG_LENGTH, key,
                    t_ed=t_ed, reduce_by=reduce_by, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean data across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(spks, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)
 
    def compute_circumferential_strain(self, spks, t_ed=0.0, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.CIRCUMFERENTIAL_STRAIN
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for all spks
        res = [self._compute_spk_relative_error(spk, 
                    self.STATES.CIRC_LENGTH, key,
                    t_ed=t_ed, reduce_by=reduce_by, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean data across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(spks, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    def compute_twist(self,  apex_spk, base_spk, t_ed: float = 0.0, reduce_by={"name"}, **kwargs):
        # set key for this function
        key = self.STATES.TWIST
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        apex_spk = self._resolve_spk_args(apex_spk)
        base_spk = self._resolve_spk_args(base_spk)
        # compute metric
        res = [self.compute_spk_twist(apex_s, base_s, **kwargs) for (apex_s, base_s) in zip(apex_spk, base_spk)]
        # reduce metric (here we compute the mean radius across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(apex_spk, key)
        self.states.set_data_spk_rel(base_spk, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(apex_spk, key, **kwargs)
            self._reduce_metric_by_group(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(apex_spk, key, **kwargs)
            self._reduce_metric_by_name(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(apex_spk, key, **kwargs)
            self._reduce_metric_by_subset(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(apex_spk, key, **kwargs)
            self._reduce_metric_by_group_and_name(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))

    def compute_torsion(self, apex_spk, base_spk, t_ed: float = 0.0, reduce_by={"name"}, **kwargs):
        # set key for this function
        key = self.STATES.TORSION
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        apex_spk = self._resolve_spk_args(apex_spk)
        base_spk = self._resolve_spk_args(base_spk)
        # compute metric
        res = [self.compute_spk_torsion(apex_s, base_s, **kwargs) for (apex_s, base_s) in zip(apex_spk, base_spk)]
        # reduce metric (here we compute the mean radius across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(apex_spk, key)
        self.states.set_data_spk_rel(base_spk, key)
        # Break down computation by each 'set of spks'
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(apex_spk, key, **kwargs)
            self._reduce_metric_by_group(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(apex_spk, key, **kwargs)
            self._reduce_metric_by_name(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(apex_spk, key, **kwargs)
            self._reduce_metric_by_subset(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(apex_spk, key, **kwargs)
            self._reduce_metric_by_group_and_name(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))


        # check for valid arguments
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        assert isinstance(t_ed, (int, float)), "t_ed must be a valid number. It refers to time at end-systole"
        # check if reference geo metric was computed
        if not self.states.check_spk_key(spk, geo_key):
            raise RuntimeError(
                "{} data not found for spk '{}'."
                "This geometric metric is required to"
                "compute '{}'".format(geo_key, spk, cm_key))
        # get data at end-systole and end-diastole.
        # end-systole will be an array (it is, essentially, all timesteps)
        d_es = self.states.get_spk_data(spk, geo_key)
        d_ed = self.states.get_spk_data(spk, geo_key, t=t_ed)
        # compute relative error
        if not switch_es:
            spk_res = self._compute_relative_error(d_ed, d_es)
        else:
            spk_res = self._compute_relative_error(d_es, d_ed)
        # save data to states and return pointer
        self.states.add_spk_data(spk, cm_key, spk_res)  
        return self.states.get_spk_data(spk, cm_key)