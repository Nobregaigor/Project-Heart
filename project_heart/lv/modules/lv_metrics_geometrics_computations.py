
import numpy as np
from .lv_speckles import LV_Speckles
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.spatial_utils import centroid, radius
from project_heart.modules.speckles.speckle import Speckle, SpeckeDeque
from project_heart.utils.spatial_utils import compute_longitudinal_length, compute_circumferential_length, compute_length_by_clustering
from project_heart.utils.vector_utils import angle_between
from project_heart.utils.enum_utils import add_to_enum

from project_heart.utils.extended_classes import ExtendedDict

import logging

logger = logging.getLogger('LV.BaseMetricsComputations')

from collections import deque



class ExplainableMetric():
    def __init__(self):
        self.key = None       # str/enum
        self.approach = None  # str
        
        # generic speckles
        self.speckles = None 
        
        # speckles at base or apex
        self.base_speckles = None
        self.apex_speckles = None
        
        # speckles at endo or epi
        self.endo_speckles = None
        self.epi_speckles = None
        
        # wheter metric used default approach to compute apex/base
        self.used_reference_apex_base = True
        self.apex = None     # reference to apex
        self.base = None     # reference to base


class LVGeometricsComputations(LV_Speckles):
    def __init__(self, log_level=logging.INFO, *args, **kwargs):
        super(LVGeometricsComputations, self).__init__(log_level=log_level, *args, **kwargs)
        self.EPSILON = 1e-10
        self.explainable_metrics = ExtendedDict() # stores methods and approach used for metric computations
        logger.setLevel(log_level)

    # =============================
    # Fundamental computations
    # =============================

    # ---- Nodal position over timesteps (xyz data)

    def compute_xyz_from_displacement(self) -> np.ndarray:
        """Computes position data (xyz) for each node for each timestep.
           Requires 'displacement' state data.

        Raises:
            RuntimeError: If 'displacement' is not found within states data. 

        Returns:
            np.ndarray (n_timesteps, 3): Pointer to 'xyz' states array.
        """

        # check if key exists in states
        if not self.states.check_key(self.STATES.DISPLACEMENT):
            raise RuntimeError(
                "'Displacement data not found in states. Did you add it to states?")
        disp = self.states.get(self.STATES.DISPLACEMENT)
        pos = disp + self.nodes()  # adds initial position to all states
        self.states.add(self.STATES.XYZ, pos)  # save to states
        return self.states.get(self.STATES.XYZ)  # return pointer

    # ----------------
    # ---- Volume ----

    def compute_volume_based_on_nodeset(self,
                                        nodeset: str = None,
                                        dtype: np.dtype = np.float64
                                        ) -> np.ndarray:
        # try to import required module
        try:
            from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
        except ImportError:
            raise ImportError(
                "scipy.spatial is required for volume computation.")
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get node ids (index array) from nodeset
        nodeset = self.REGIONS.ENDO if nodeset is None else nodeset
        nodeids = self.get_nodeset(nodeset)
        # apply nodeids mask for position data
        positions = self.states.get(self.STATES.XYZ, mask=nodeids)
        vols = np.zeros(len(positions), dtype=dtype)
        for i, xyz in enumerate(positions):
            vols[i] = ConvexHull(xyz, qhull_options="Qt Qx Qv Q4 Q14").volume
        self.states.add(self.STATES.VOLUME, vols)  # save to states
        return self.states.get(self.STATES.VOLUME)  # return pointer

    # =============================
    #   Geometric computations
    # =============================
    
    # ---- Apex and Base ref over timesteps
    
    def compute_base_apex_ref_over_timesteps(self, apex_spk, base_spk, recompute=True, log_level=logging.INFO):
        
        # recompute is not used (here for temporary placeholder)
        
        log = logger.getChild("compute_base_apex_ref_over_timesteps")
        log.setLevel(log_level)
        log.info("Computing apex and base virtual nodes over timesteps")
        log.debug("Using apex spk: {}".format(apex_spk))
        log.debug("Using base spk: {}".format(base_spk))

        apex_spk = self._resolve_spk_args(apex_spk)
        base_spk = self._resolve_spk_args(base_spk)
        
        assert len(apex_spk) == 1, "Only one speckle is allowed for apex computation."
        assert len(base_spk) == 1, "Only one speckle is allowed for base computation."
        
        apex_spk = apex_spk[0]
        base_spk = base_spk[0]
                
        assert self.check_spk(
            apex_spk), "apex_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            base_spk), "base_spk must be a valid 'Speckle' object."
        
        # check if xyz was computed; If not, try to compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
            
        xyz_apex = self.states.get(self.STATES.XYZ, mask=apex_spk.ids)
        xyz_base = self.states.get(self.STATES.XYZ, mask=base_spk.ids)
        
        apex = [centroid(xyz) for xyz in xyz_apex]
        base = [centroid(xyz) for xyz in xyz_base]
        
        log.debug("apex: \n{}".format(apex))
        log.debug("base: \n{}".format(base))
        
        self.states.add(self.STATES.BASE_REF, base)  # save to states
        self.states.add(self.STATES.APEX_REF, apex)  # save to states

        # return pointers
        return (self.states.get(self.STATES.BASE_REF), self.states.get(self.STATES.APEX_REF))
          
    # ---- Speckle centers at longitudinal axis

    def compute_spk_la_centers_over_timesteps(self, spk, log_level=logging.INFO, **kwargs):
        from project_heart.utils.spatial_utils import project_pt_on_line
        # check for speckle input
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle center for spk: '{}'".format(spk))
        # check if apex and base references were computed
        if not self.states.check_key(self.STATES.BASE_REF) or \
            not self.states.check_key(self.STATES.APEX_REF):
            raise RuntimeError("States not found for BASE_REF or APEX_REF."
                               "Please, either set them manually or try using "
                               "'compute_base_apex_ref_over_timesteps'.")
        # compute spk centers over timesteps based on 'k' height
        # from project_heart.utils.spatial_utils import get_p_along_line
        apex_ts = self.states.get(self.STATES.APEX_REF)
        base_ts = self.states.get(self.STATES.BASE_REF)
        # spk_center = np.mean(self.get_speckles_xyz(spk), 1)
        xyz_ts = self.states.get(self.STATES.XYZ, mask=spk.ids)
        # spk_center = np.asarray([centroid(xyz) for xyz in xyz_ts], dtype=np.float64)
        spk_center = np.mean(xyz_ts, axis=0)
        spk_res = np.zeros((len(xyz_ts),3), dtype=np.float64)
        for i, (sc, at, bt) in enumerate(zip(spk_center, apex_ts, base_ts)):
            spk_res[i] = project_pt_on_line(sc, at, bt)
        logger.debug("\n-apex:'{}\n-base:'{}'\n-centers:'{}'".
                     format(np.vstack(apex_ts), np.vstack(base_ts), spk_res))
        self.states.add_spk_data(spk, self.STATES.LA_CENTERS, spk_res)  # save to states
        return self.states.get_spk_data(spk, self.STATES.LA_CENTERS) # return pointer
    
    def compute_la_centers_over_timesteps(self, spks, log_level=logging.INFO, **kwargs):
        log = logger.getChild("compute_la_centers_over_timesteps")
        log.setLevel(log_level)
        log.debug("Computing spekles la centers over timesteps.")
        spks = self._resolve_spk_args(spks)
        res = [self.compute_spk_la_centers_over_timesteps(spk, log_level=log_level, **kwargs) for spk in spks]
        
    
    # -------------------------------
    # ---- Longitudinal distance ----

    def compute_longitudinal_distances_between_speckles(self, apex_spk, base_spk, 
                                                        approach="centroid",
                                                        use_axis_aligment=False,
                                                        log_level=logging.INFO, 
                                                        dtype=np.float64, **kwargs):
        key = self.STATES.LONGITUDINAL_DISTANCE
        logger.info("Computing '{}' with approach '{}' with axis aligment set to '{}'."
                    .format(key, approach, use_axis_aligment))        
        if approach == "along_longitudinal_axis":
            # check if speckle centers were computed. If not, compute them.
            # -- check for apex LA centers
            if not self.states.check_spk_key(apex_spk, self.STATES.LA_CENTERS):
                self.compute_spk_la_centers_over_timesteps(apex_spk, log_level=log_level)
            # -- check for base LA centers
            if not self.states.check_spk_key(base_spk, self.STATES.LA_CENTERS):
                self.compute_spk_la_centers_over_timesteps(base_spk, log_level=log_level)
            # get centers data
            apex_centers = self.states.get_spk_data(apex_spk, self.STATES.LA_CENTERS)
            base_centers = self.states.get_spk_data(base_spk, self.STATES.LA_CENTERS)
        elif approach == "centroid":
            apex_centers = np.array([centroid(xyz) for xyz in self.get_speckles_xyz(apex_spk)], dtype=dtype)
            base_centers = np.array([centroid(xyz) for xyz in self.get_speckles_xyz(base_spk)], dtype=dtype)
        elif approach == "mean":
            apex_centers = np.mean(self.get_speckles_xyz(apex_spk), 1)
            base_centers = np.mean(self.get_speckles_xyz(base_spk), 1)
        else:
            raise ValueError("Invalid approach. Options are: 'along_longitudinal_axis'. 'centroid' or 'mean'. "
                             "Received: {}".format(approach))
        # compute distance
        if not use_axis_aligment:
            from project_heart.utils.spatial_utils import distance
            spk_res = np.array([distance(a_c, b_c) for a_c, b_c in zip(apex_centers, base_centers)], 
                                dtype=dtype)
        else:
            spk_res = np.array([abs(b_c[2] - a_c[2]) for a_c, b_c in zip(apex_centers, base_centers)], 
                                dtype=dtype)
        
        # record explainable_metrics
        exm = ExplainableMetric()
        exm.key = key
        exm.approach = approach
        exm.apex_speckles = apex_spk
        exm.base_speckles = base_spk
        exm.used_reference_apex_base=False
        exm.apex = apex_centers
        exm.base = base_centers
        self.explainable_metrics[self.states.get_spk_state_key(apex_spk, key)] = exm
        self.explainable_metrics[self.states.get_spk_state_key(base_spk, key)] = exm
        
        
        self.states.add_spk_data(apex_spk, key, spk_res)  # save to states
        self.states.add_spk_data(base_spk, key, spk_res)  # save to states
        return self.states.get_spk_data(apex_spk, key) # return pointer

    def compute_longitudinal_distance(self, apex_spks, base_spks, reduce_by=None,
                                      dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
        # set key for this function
        key = self.STATES.LONGITUDINAL_DISTANCE
        logger.info("Computing metric '{}'.".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        apex_spks = self._resolve_spk_args(apex_spks)
        base_spks = self._resolve_spk_args(base_spks)
        # compute metric
        res = [self.compute_longitudinal_distances_between_speckles(apex_s, base_s, **kwargs) for (apex_s, base_s) in zip(apex_spks, base_spks)]
        # reduce metric (here we compute the mean radius across entire LV)
        self._reduce_metric_and_save(res, key, **kwargs)
        # set metric relationship with spks 
        # so that we can reference which spks were used to compute this metric
        self.states.set_data_spk_rel(apex_spks, key)
        self.states.set_data_spk_rel(base_spks, key)
        # Break down computation by each 'set of spks'
        if reduce_by is None:
            reduce_by = {"group"}
        if "group" in reduce_by:
            logger.debug("Reducing metric by group for '{}'".format(key))
            self._reduce_metric_by_group(apex_spks, key, **kwargs)
            self._reduce_metric_by_group(base_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group.".format(key))
        if "name" in reduce_by:
            logger.debug("Reducing metric by name for '{}'".format(key))
            self._reduce_metric_by_name(apex_spks, key, **kwargs)
            self._reduce_metric_by_name(base_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by names.".format(key))
        if "subset" in reduce_by:
            logger.debug("Reducing metric by subset for '{}'".format(key))
            self._reduce_metric_by_subset(apex_spks, key, **kwargs)
            self._reduce_metric_by_name(base_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(apex_spks, key, **kwargs)
            self._reduce_metric_by_group_and_name(base_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
    
    # -------------------------
    # ---- Radial distance ---- 

    def compute_spk_radial_distance(self, spk: object, 
                           dtype: np.dtype = np.float64, 
                           approach="moving_vector",
                           log_level=logging.INFO,
                           **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        key = self.STATES.RADIAL_DISTANCE
        logger.setLevel(log_level)
        logger.debug("Computing speckle RADIAL_DISTANCE for spk: '{}'".format(spk))
        logger.debug("Using approach: '{}'".format(approach))
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get nodal position for all timesteps
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        if approach == "moving_vector":
            from project_heart.utils.vector_utils import dist_from_line
            # check if speckle apex and base values were computed. If not, compute them.
            if not self.states.check_key(self.STATES.BASE_REF) or \
            not self.states.check_key(self.STATES.APEX_REF):
                raise RuntimeError("STATES.APEX_REF and STATES.BASE_REF references were not computed. "
                                   "Either add them manually or use "
                                   "'compute_base_apex_ref_over_timesteps'.")
            # get apex and base points over timesteps
            apex_ts = self.states.get(self.STATES.APEX_REF)
            base_ts = self.states.get(self.STATES.BASE_REF)
            spk_res = np.array([
                        np.mean(dist_from_line(coords, apts, bpts)) for coords, apts, bpts in zip(xyz, apex_ts, base_ts)], 
                        dtype=dtype)
        elif approach == "fixed_vector":
            from project_heart.utils.vector_utils import dist_from_line
            long_line = self.get_long_line() # based on ref normal
            p2, p3 = long_line[0], long_line[1]
            spk_res = np.array([np.mean(dist_from_line(coords, p2, p3)) for coords in xyz], dtype=dtype)
        else:
            raise ValueError("Unknown approach. Avaiable approaches are: "
                             "'moving_vector', 'fixed_vector'.Received: '{}'. "
                             "Please, check documentation for further details."
                             .format(approach))
        logger.debug("-mean_coords:'{}\n-RADIAL_DISTANCE:'{}'"
                     .format(np.mean(xyz, axis=0), spk_res))
        
        # record explainable_metrics
        exm = ExplainableMetric()
        exm.key = key
        exm.approach = approach
        exm.speckles = spk
        exm.used_reference_apex_base=True
        self.explainable_metrics[self.states.get_spk_state_key(spk, key)] = exm
        
        self.states.add_spk_data(spk, key, spk_res)  # save to states
        return self.states.get_spk_data(spk, key) # return pointer

    def compute_radial_distance(self, spks, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.RADIAL_DISTANCE
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric
        res = [self.compute_spk_radial_distance(spk, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean radius across entire LV)
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

    # -----------------------
    # ---- Radial length ---- 

    def compute_spk_radial_length(self, spk: object, 
                           dtype: np.dtype = np.float64, 
                           approach="moving_centers",
                           log_level=logging.INFO,
                           **kwargs):
        key = self.STATES.RADIAL_LENGTH
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle RADIAL_LENGTH for spk: '{}'".format(spk))
        logger.debug("Using approach: '{}'".format(approach))
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get nodal position for all timesteps
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        if approach == "moving_centers":
            # check if speckle centers were computed. If not, compute them.
            if not self.states.check_spk_key(spk, self.STATES.LA_CENTERS):
                self.compute_spk_la_centers_over_timesteps(spk, log_level=log_level, **kwargs)
            # get centers data
            centers = self.states.get_spk_data(spk, self.STATES.LA_CENTERS)
            spk_res = np.array([radius(coords, center=center) for coords, center in zip(xyz, centers)], 
                            dtype=dtype)
        elif approach == "fixed_centers":
            spk_res = np.array([radius(coords, center=spk.la_center) for coords in xyz], 
                            dtype=dtype)
        else:
            raise ValueError("Unknown approach. Avaiable approaches are: "
                             "'moving_centers', 'fixed_centers'. Received: '{}'. "
                             "Please, check documentation for further details."
                             .format(approach))
            
        logger.debug("-mean_coords:'{}\n-RADIAL_LENGTH:'{}'".
                     format(np.mean(xyz, axis=0), spk_res))
        # record explainable_metrics
        exm = ExplainableMetric()
        exm.key = key
        exm.approach = approach
        exm.speckles = spk
        exm.used_reference_apex_base=True
        self.explainable_metrics[self.states.get_spk_state_key(spk, key)] = exm
        
        self.states.add_spk_data(spk, key, spk_res)  # save to states
        return self.states.get_spk_data(spk, key) # return pointer

    def compute_radial_length(self, spks, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.RADIAL_LENGTH
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric
        res = [self.compute_spk_radial_length(spk, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean radius across entire LV)
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

    # ------------------------
    # ---- Wall thickness ---- 

    def compute_spk_thickness(self, endo_spk, epi_spk, 
                              approach="radial_distance",
                              dist_tol_between_spks_la_center=1.0,
                              assert_subsets=True,
                              log_level=logging.INFO,
                              **kwargs):
        logger.setLevel(log_level)
        logger.debug("Computing speckle thickness for spks: '{}' and '{}"
                     .format(endo_spk, epi_spk))
        
        assert self.check_spk(
            endo_spk), "endo_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            epi_spk), "epi_spk must be a valid 'Speckle' object."
        
        # safety assertions
        if dist_tol_between_spks_la_center >= 0:
            endo_center = endo_spk.la_center
            epi_center = epi_spk.la_center
            dist = np.linalg.norm(endo_center - epi_center)
            assert dist <= dist_tol_between_spks_la_center, (""
                    "Distance between endo and epi speckles is greater than allowed tolerance. "
                    "This means that speckles might not be related in order to compute thickness. "
                    "For instance, trying to use 'base' and 'apex' speckle; which will "
                    "result in wrong thickness computation. If you want to proceed, please "
                    "set 'dist_tol_between_spks_la_center' to -1.\n"
                    "Distance: {}. Allowed: {}\n"
                    "Endo reference center is located at: {}. \n"
                    "Epi reference center is located at: {}. \n"
                    "Endo spk: {}.\nEpi spk: {}"
                    "".format(dist, dist_tol_between_spks_la_center,
                              endo_center, epi_center, endo_spk, epi_spk)
                    )
        if assert_subsets:
            endo_subset = endo_spk.subset
            epi_subset = epi_spk.subset
            assert endo_subset == epi_subset, (""
                "Subsets are not equal. This means that you might be trying to use speckles "
                "at different segements to compute thickness. For instance, trying to use "
                "speckles at '0' and '90' degrees, respectively, w.r.t. X axis, which will "
                "result in wrong thickness computation. If you would like to proceed, please "
                " set 'assert_subsets' to False.\n"
                "Endo spk: {}.\nEpi spk: {}"
                "".format(endo_spk, epi_spk)
                )
        
        # check for valid approaches
        if approach == "radial_distance":
            radial_metric = self.STATES.RADIAL_DISTANCE
        elif approach == "radial_length":
            radial_metric = self.STATES.RADIAL_LENGTH
        else:
            raise ValueError("Unknown approach. Options are: "
                             "'radial_distance' or 'radial_length'. "
                             "Received '{}'"
                             "Check documentation for further details."
                             .format(approach))
            
        # check if radial metric was computed for ENDOCARDIUM
        if not self.states.check_spk_key(endo_spk, radial_metric):
            logger.debug("Metric 'RADIAL_DISTANCE' not found for spk '{}'. Will try to compute.".format(endo_spk))
            try:
                if approach == "radial_distance":
                    self.compute_radial_distance(endo_spk, **kwargs)
                else:
                    self.compute_radial_length(endo_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for endo spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIAL_DISTANCE' manually."
                    .format(endo_spk.str))
        # check if radial metric was computed for EPICARDIUM
        if not self.states.check_spk_key(epi_spk, radial_metric):
            logger.debug("Metric 'RADIAL_DISTANCE' not found for spk '{}'. Will try to compute.".format(epi_spk))
            try:
                if approach == "radial_distance":
                    self.compute_radial_distance(epi_spk, **kwargs)
                else:
                    self.compute_radial_length(epi_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for epi spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIAL_DISTANCE' manually."
                    .format(epi_spk.str))
        # get spk data
        r_endo = self.states.get_spk_data(endo_spk, radial_metric)
        r_epi = self.states.get_spk_data(epi_spk, radial_metric)
        # compute thickness
        thickness = r_epi - r_endo
        logger.debug("-r_endo:\n'{}'\n-r_epi:\n'{}\n-thickness:\n'{}'\n".
                     format(r_endo, r_epi, thickness))
        self.states.add_spk_data(endo_spk, self.STATES.WALL_THICKNESS, thickness)  # save to states
        self.states.add_spk_data(epi_spk, self.STATES.WALL_THICKNESS, thickness)  # save to states
        # return pointer
        return self.states.get_spk_data(endo_spk, self.STATES.WALL_THICKNESS)

    def compute_thickness(self, endo_spks, epi_spks, reduce_by={"name"}, **kwargs):
        # set key for this function
        key = self.STATES.WALL_THICKNESS
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        endo_spks = self._resolve_spk_args(endo_spks)
        epi_spks = self._resolve_spk_args(epi_spks)
        # compute metric
        res = [self.compute_spk_thickness(endo_s, epi_s, **kwargs) for (endo_s, epi_s) in zip(endo_spks, epi_spks)]
        # reduce metric (here we compute the mean radius across entire LV)
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

    # -----------------------------
    # ---- Longitudinal length ----  

    def compute_spk_longitudinal_length(self,
                                        spk,
                                        approach:str = "k_ids",
                                        as_global=False,
                                        line_seg_filter_kwargs=None,
                                        dtype: np.dtype = np.float64,
                                        log_level=logging.INFO,
                                        **kwargs):
        from functools import partial
        if not as_global:
            key = self.STATES.LONGITUDINAL_LENGTH
        else:
            key = self.STATES.GLOBAL_LONGITUDINAL_LENGTH
        # check for valid spk object
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle LONGITUDINAL_LENGTH  for spk: '{}'".format(spk))
        logger.debug("Using approach: '{}'".format(approach))
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # resolve computation method
        if approach == "k_ids":
            from project_heart.utils.spatial_utils import compute_length_from_predefined_cluster_list
            fun = partial(compute_length_from_predefined_cluster_list, 
                          clusters=spk.c_local_ids,
                          assume_sorted=True,
                          filter_args=line_seg_filter_kwargs,
                          join_ends=False,
                          dtype=dtype,**kwargs)
        elif approach == "kmeans":
            from project_heart.utils.spatial_utils import compute_length_by_clustering
            fun = partial(compute_length_by_clustering, **kwargs)
        elif approach == "grouping":
            from project_heart.utils.spatial_utils import compute_longitudinal_length
            fun = partial(compute_longitudinal_length, **kwargs)
        else:
            raise ValueError("Unknown method '{}'. options are: 'fast', 'clusters'.")
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        # compute response for given spk
        spk_res = np.array([fun(coords) for coords in xyz], dtype=dtype)
        # save to states
        self.states.add_spk_data(spk, key, spk_res)
        # return pointer
        return self.states.get_spk_data(spk, key)

    def compute_longitudinal_length(self, spks, reduce_by={"group"}, as_global=False, log_level=logging.INFO, **kwargs):
        # set key for this function
        logger.setLevel(log_level)
        if not as_global:
            key = self.STATES.LONGITUDINAL_LENGTH
        else:
            key = self.STATES.GLOBAL_LONGITUDINAL_LENGTH
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for each speckle -> results in the length of each spk
        res = [self.compute_spk_longitudinal_length(spk, as_global=as_global, 
                                                    log_level=log_level, **kwargs) for spk in spks]
        if not as_global:
            # apply metric reduction -> Note: this is not longitudinal length, but instead
            # the 'reduced' longitudinal length for each speckle. Default will be avg. spk length.
            self.STATES, (_, sub_key) = add_to_enum(self.STATES, key, "SPK_REDUCED")
            self._reduce_metric_and_save(res, sub_key, **kwargs) # reduction by all spks
            # now let's reduce by group. This should apply reduction across similar regions.
            res_group = self._reduce_metric_by_group_and_name(spks, key, method="sum")
            # we now have the reduce value by similar regions, we need to get a 'single' 
            # longitudinal length. We will do so by reduction across groups.
            res_group_as_arr = np.array(list(res_group.values()))
            self._reduce_metric_and_save(res_group_as_arr, key)
            # set metric relationship with spks 
            # so that we can reference which spks were used to compute this metric
            self.states.set_data_spk_rel(spks, key)
            # this function cannot use the same reduction method as other
            # as the computation must sum each length in a different manner.
            # this procedure is less expensive than re-computing using a combination
            # of _reduce_metric_by_group_and_name and _reduce_metric_by_grou,
            # for instance.
            # Break down computation by each 'set of spks'
            if "group" in reduce_by or "name" in reduce_by:
                logger.debug("res_group.keys() '{}'".format(res_group.keys()))
                # group computed values by respective group and name key
                # we will do in a single loop so we have to have both dicts here
                res_by_name = dict()
                res_by_group = dict()
                for (groupkey, namekey), res in res_group.items():
                    if not namekey in res_by_name:
                        res_by_name[namekey] = deque([])
                    if not groupkey in res_by_group:
                        res_by_group[groupkey] = deque([])
                    res_by_name[namekey].append(res)
                    res_by_group[groupkey].append(res)
                logger.debug("res_by_name.keys() '{}'".format(res_by_name.keys()))
                logger.debug("res_by_group.keys() '{}'".format(res_by_group.keys()))
                # if user requested to compute length 'namewise'
                if "name" in reduce_by:
                    logger.debug("Reducing metric by name for '{}'".format(key))
                    # compute reduced value for each 'name' in the grouped values
                    for namekey, res in res_by_name.items():
                        logger.debug("namekey '{}''".format(namekey, res))
                        # select spks relate to 'name' and add new enum to states
                        sel_spks = [spk for spk in spks if spk.name == namekey]
                        self.STATES, (_, statekey) = add_to_enum(self.STATES, key, namekey)
                        logger.debug("statekey '{}''".format(statekey))
                        # save spk-data relationship for future reference
                        self.states.set_data_spk_rel(sel_spks, statekey)
                        # reduce subgroup
                        self._reduce_metric_and_save(res, statekey, **kwargs)
                    logger.debug("Metric '{}' has reduced values by names.".format(key))
                # if user requested to compute length 'groupwise'
                if "group" in reduce_by:
                    logger.debug("Reducing metric by group and name for '{}'".format(key))
                    # compute reduced value for each 'group' in the grouped values
                    for groupkey, res in res_by_group.items():
                        logger.debug("groupkey '{}''".format(groupkey, res))
                        # select spks relate to 'group' and add new enum to states
                        sel_spks = [spk for spk in spks if spk.group == groupkey]
                        self.STATES, (_, statekey) = add_to_enum(self.STATES, key, groupkey)
                        logger.debug("statekey '{}''".format(statekey))
                        # save spk-data relationship for future reference
                        self.states.set_data_spk_rel(sel_spks, statekey)
                        # reduce subgroup
                        self._reduce_metric_and_save(res, statekey, **kwargs)
                    logger.debug("Metric '{}' has reduced values by group and name.".format(key))
            # reduce by subset
            if "subset" in reduce_by:
                logger.debug("Reducing metric by subset for '{}'".format(key))
                self._reduce_metric_by_subset(spks, key, **kwargs)
                logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        else:
            # reduce metric (here we compute the mean radius across entire LV)
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
        return self.states.get(key)  # return pointer

    # --------------------------------
    # ---- Circumferential length ----

    def compute_spk_circumferential_length(self, spk, 
                                           approach="k_ids",
                                           as_global=False,
                                           line_seg_filter_kwargs=None,
                                           dtype: np.dtype = np.float64,
                                           **kwargs):
        from functools import partial
        if not as_global:
            key = self.STATES.CIRCUMFERENTIAL_LENGTH
            join_ends=False
        else:
            key = self.STATES.GLOBAL_CIRCUMFERENTIAL_LENGTH
            join_ends=True
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # resolve computation method
        if approach == "k_ids":
            from project_heart.utils.spatial_utils import compute_length_from_predefined_cluster_list
            fun = partial(compute_length_from_predefined_cluster_list, 
                          clusters=spk.c_local_ids,
                          assume_sorted=True,
                          filter_args=line_seg_filter_kwargs,
                          join_ends=join_ends,
                          dtype=dtype,**kwargs)
        elif approach == "kmeans":
            from project_heart.utils.spatial_utils import compute_length_by_clustering
            fun = partial(compute_length_by_clustering, **kwargs)
        elif approach == "grouping":
            from project_heart.utils.spatial_utils import compute_circumferential_length
            fun = partial(compute_circumferential_length, **kwargs)
        else:
            raise ValueError("Unknown method '{}'. options are: 'fast', 'clusters'.")
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        # compute response for given spk
        spk_res = np.array([fun(coords) for coords in xyz], dtype=dtype)
        # save to states
        self.states.add_spk_data(spk, key, spk_res)
        # return pointer
        return self.states.get_spk_data(spk, key)

    def compute_circumferential_length(self, spks, reduce_by={"group"}, as_global=False, **kwargs):
        # set key for this function
        if not as_global:
            key = self.STATES.CIRCUMFERENTIAL_LENGTH
        else:
            key = self.STATES.GLOBAL_CIRCUMFERENTIAL_LENGTH
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for each speckle -> results in the length of each spk
        res = [self.compute_spk_circumferential_length(spk, as_global=as_global, **kwargs) for spk in spks]
        if not as_global:
            # apply metric reduction -> Note: this is not circ. length, but instead
            # the 'reduced' circ. length for each speckle. Default will be avg. spk length.
            self.STATES, (_, sub_key) = add_to_enum(self.STATES, key, "SPK_REDUCED")
            self._reduce_metric_and_save(res, sub_key, **kwargs) # reduction by all spks
            # now let's reduce by group. This should apply reduction across similar regions.
            res_group = self._reduce_metric_by_group_and_name(spks, key, method="sum")
            logger.debug("res_group '{}'".format(res_group))
            # we now have the reduce value by similar regions, we need to get a 'single' 
            # circ. length. We will do so by reduction across groups.
            res_group_as_arr = np.array(list(res_group.values()))
            self._reduce_metric_and_save(res_group_as_arr, key)
            # set metric relationship with spks 
            # so that we can reference which spks were used to compute this metric
            self.states.set_data_spk_rel(spks, key)
            # this function cannot use the same reduction method as other
            # as the computation must sum each length in a different manner.
            # this procedure is less expensive than re-computing using a combination
            # of _reduce_metric_by_group_and_name and _reduce_metric_by_grou,
            # for instance.
            # Break down computation by each 'set of spks'
            if "group" in reduce_by or "name" in reduce_by:
                logger.debug("res_group.keys() '{}'".format(res_group.keys()))
                # group computed values by respective group and name key
                # we will do in a single loop so we have to have both dicts here
                res_by_name = dict()
                res_by_group = dict()
                for (groupkey, namekey), res in res_group.items():
                    if not namekey in res_by_name:
                        res_by_name[namekey] = deque([])
                    if not groupkey in res_by_group:
                        res_by_group[groupkey] = deque([])
                    res_by_name[namekey].append(res)
                    res_by_group[groupkey].append(res)
                logger.debug("res_by_name.keys() '{}'".format(res_by_name.keys()))
                logger.debug("res_by_group.keys() '{}'".format(res_by_group.keys()))
                # if user requested to compute length 'namewise'
                if "name" in reduce_by:
                    logger.debug("Reducing metric by name for '{}'".format(key))
                    # compute reduced value for each 'name' in the grouped values
                    for namekey, res in res_by_name.items():
                        logger.debug("namekey '{}''".format(namekey, res))
                        # select spks relate to 'name' and add new enum to states
                        sel_spks = [spk for spk in spks if spk.name == namekey]
                        self.STATES, (_, statekey) = add_to_enum(self.STATES, key, namekey)
                        logger.debug("statekey '{}''".format(statekey))
                        # save spk-data relationship for future reference
                        self.states.set_data_spk_rel(sel_spks, statekey)
                        # reduce subgroup
                        self._reduce_metric_and_save(res, statekey, **kwargs)
                    logger.debug("Metric '{}' has reduced values by names.".format(key))
                # if user requested to compute length 'groupwise'
                if "group" in reduce_by:
                    logger.debug("Reducing metric by group and name for '{}'".format(key))
                    # compute reduced value for each 'group' in the grouped values
                    for groupkey, res in res_by_group.items():
                        logger.debug("groupkey '{}''".format(groupkey, res))
                        # select spks relate to 'group' and add new enum to states
                        sel_spks = [spk for spk in spks if spk.group == groupkey]
                        self.STATES, (_, statekey) = add_to_enum(self.STATES, key, groupkey)
                        logger.debug("statekey '{}''".format(statekey))
                        # save spk-data relationship for future reference
                        self.states.set_data_spk_rel(sel_spks, statekey)
                        # reduce subgroup
                        self._reduce_metric_and_save(res, statekey, **kwargs)
                    logger.debug("Metric '{}' has reduced values by group and name.".format(key))
            # reduce by subset
            if "subset" in reduce_by:
                logger.debug("Reducing metric by subset for '{}'".format(key))
                self._reduce_metric_by_subset(spks, key, **kwargs)
                logger.debug("Metric '{}' has reduced values by subsets.".format(key))
        else:
            # reduce metric (here we compute the mean radius across entire LV)
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
        return self.states.get(key)  # return pointer

    # -------------------
    # ----- Rotation ---- 

    def compute_spk_vectors(self, spk, dtype: np.dtype = np.float64, **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # check if speckle centers were computed. If not, compute them.
        if not self.states.check_spk_key(spk, self.STATES.LA_CENTERS):
            self.compute_spk_la_centers_over_timesteps(spk, **kwargs)
        # get centers data
        centers = self.states.get_spk_data(spk, self.STATES.LA_CENTERS)
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        vecs = np.array([xyz[i] - centers[i] for i in range(len(xyz))], dtype=dtype)
        # save to states
        self.states.add_spk_data(spk, self.STATES.SPK_VECS, vecs)  
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.SPK_VECS)

    def compute_spk_rotation(self, 
                             spk,
                             t_ed: float = 0.0,
                             dtype: np.dtype = np.float64,
                             check_orientation=False,
                             degrees=True,
                             **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        if not self.states.check_spk_key(spk, self.STATES.SPK_VECS):
            try:
                self.compute_spk_vectors(spk, dtype=dtype, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute vectors for spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'SPK_VECS' manually."
                    .format(spk.str))
        # get reference position vector
        p_ref = self.states.get_spk_data(spk, self.STATES.SPK_VECS, t=t_ed)
        # get vecs for each state
        p_vec = self.states.get_spk_data(spk, self.STATES.SPK_VECS)
        logger.debug("rot:' p_ref.shape {}'".format(p_ref.shape))
        logger.debug("rot:' p_vec.shape {}'".format(p_vec.shape))
        # compute rotation for each timestep
        rot = [angle_between(xyz_vec, p_ref, check_orientation=check_orientation) for xyz_vec in p_vec]
        rot = np.mean(rot, axis=1)
        logger.debug("rot: rot_vec.shape '{}'".format(rot.shape))
        # convert to degrees
        if degrees:
            rot = np.degrees(rot)
        # save to states
        self.states.add_spk_data(spk, self.STATES.ROTATION, rot)  
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.ROTATION)

    def compute_rotation(self, spks, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.ROTATION
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric
        res = [self.compute_spk_rotation(spk, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean radius across entire LV)
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

    # ----------------------------
    # ----- Twist and torsion ---- 

    def compute_spk_twist(self, apex_spk, base_spk, t_ed: float = 0.0, **kwargs):
        assert self.check_spk(
            apex_spk), "apex_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            base_spk), "base_spk must be a valid 'Speckle' object."

        if not self.states.check_spk_key(apex_spk, self.STATES.ROTATION):
            try:
                self.compute_spk_rotation(
                    apex_spk, t_ed=t_ed, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute rotation for spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'ROTATION' manually."
                    .format(apex_spk.str))
        if not self.states.check_spk_key(base_spk, self.STATES.ROTATION):
            try:
                self.compute_spk_rotation(
                    base_spk, t_ed=t_ed, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute rotation for spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'ROTATION' manually."
                    .format(base_spk.str))

        r_apex = self.states.get_spk_data(apex_spk, self.STATES.ROTATION)
        r_base = self.states.get_spk_data(base_spk, self.STATES.ROTATION)
        twist = r_base - r_apex
        self.states.add_spk_data(apex_spk, self.STATES.TWIST, twist)  # save to states
        self.states.add_spk_data(base_spk, self.STATES.TWIST, twist)  # save to states
        # return pointer
        return self.states.get_spk_data(apex_spk, self.STATES.TWIST)

    def compute_spk_torsion(self, apex_spk, base_spk, t_ed: float = 0.0, relative=False, **kwargs):
        assert self.check_spk(
            apex_spk), "apex_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            base_spk), "base_spk must be a valid 'Speckle' object."

        if not self.states.check_spk_key(apex_spk, self.STATES.TWIST) \
                or not self.states.check_spk_key(base_spk, self.STATES.TWIST):
            try:
                self.compute_spk_twist(
                    apex_spk, base_spk, t_ed=t_ed, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute twist for spk '{}' and '{}'."
                    "Please, either verify required data or add"
                    "state data for 'TWIST' manually."
                    .format(apex_spk.str, base_spk.str))

        twist = self.states.get_spk_data(apex_spk, self.STATES.TWIST)
        D = np.linalg.norm(apex_spk.la_center - base_spk.la_center)
        torsion = twist / D  # units = angle/mm
        if relative:  # relative will multiply by the mean radius of base and apex -> units: angle
            c = 0.5 * (r_apex + r_base)
            torsion *= c

        self.states.add_spk_data(
            apex_spk, self.STATES.TORSION, torsion)  # save to states
        self.states.add_spk_data(
            base_spk, self.STATES.TORSION, torsion)  # save to states
        return self.states.get_spk_data(base_spk, self.STATES.TORSION)

    # ===============================
    # Not a geometric, but it is related
    # ===============================
    
    # -----------------------------
    # Stress
    
    def extract_principal_stress(self, stress_key=None, dtype=np.float32):
        def to_sym_arr(data):
            return [
                        [data[0], data[3], data[5]],
                        [data[3], data[1], data[4]],
                        [data[5], data[4], data[2]]
                    ]
        
        key = self.STATES.PRINCIPAL_STRESS # how to name principal stress
        
        if stress_key is None:
            stress_key = self.STATES.STRESS
        
        stress = self.states.get(stress_key)
        from collections import deque
        principal_stress = deque()
        for step in stress:
            step_data = deque()
            for cell in step:
                data = to_sym_arr(cell)
                max_stress = np.max(np.linalg.eigvals(data))
                step_data.append(max_stress)
            principal_stress.append(step_data)
        principal_stress = np.asarray(principal_stress, dtype=dtype)
        self.states.add(self.STATES.PRINCIPAL_STRESS, principal_stress)
        
    def extract_principal_strain(self, strain_key=None, dtype=np.float32):
        def to_sym_arr(data):
            return [
                        [data[0], data[3], data[5]],
                        [data[3], data[1], data[4]],
                        [data[5], data[4], data[2]]
                    ]
        
        key = self.STATES.PRINCIPAL_STRAIN # how to name principal strain
        
        if strain_key is None:
            strain_key = self.STATES.STRAIN
        
        strain = self.states.get(strain_key)
        from collections import deque
        principal_strain = deque()
        for step in strain:
            step_data = deque()
            for cell in step:
                data = to_sym_arr(cell)
                max_strain = np.max(np.linalg.eigvals(data))
                step_data.append(max_strain)
            principal_strain.append(step_data)
        principal_strain = np.asarray(principal_strain, dtype=dtype)
        self.states.add(self.STATES.PRINCIPAL_STRAIN, principal_strain)
       
    
    def compute_spk_stress(self, spk: object,
                            dtype: np.dtype = np.float64, 
                            approach="mean",
                            cylindrical=False, 
                            effective=False,
                            principal=True,
                            use_axis=None,
                            log_level=logging.INFO,
                            **kwargs):
        if cylindrical:
            key = self.STATES.CYLINDRICAL_STRESS
            if not self.states.check_key(key):
                try:
                    if not self.states.check_key(self.STATES.STRESS):
                        raise RuntimeError("Could not compute cylindrical stress. Did you compute/recorded stress?")
                    else:
                        cy_stress = self.convert_to_cylindrical_coordinates(self.STATES.STRESS)
                        self.states.add(self.STATES.CYLINDRICAL_STRESS, cy_stress)
                except:
                    raise RuntimeError("Could not compute cylindrical stress. Try adding it manually. ")
        elif principal:
            key = self.STATES.PRINCIPAL_STRESS
            if not self.states.check_key(key):
                try:
                    if not self.states.check_key(self.STATES.STRESS):
                        raise RuntimeError("Could not compute principal stress. Did you compute/recorded stress?")
                    else:
                        self.extract_principal_stress(self.STATES.STRESS)
                except:
                    raise RuntimeError("Could not compute cylindrical stress. Try adding it manually. ")
        
        else:
            key = self.STATES.STRESS
        
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle STRESS for spk: '{}'".format(spk))
        logger.debug("Using approach: '{}'".format(approach))
        # check if stress variable exists in states
        if not self.states.check_key(key):
            raise RuntimeError("Could not find '{}' in states. Did you compute it?".format(key))
        # get stress value for each timestep
        stress = self.states.get(key, mask=spk.elem_ids)
        logger.debug("-stress.shape:'{}".format(stress.shape))
        if use_axis is not None:
            logger.debug("-use_axis:'{}".format(use_axis))
            stress = stress[:, use_axis]
            logger.debug("-stress.shape:'{}".format(stress.shape))
        
        if approach == "mean":
            spk_res = np.mean(stress, axis=1)
        elif approach == "max":
            spk_res = np.max(stress, axis=1)
        elif approach == "min":
            spk_res = np.min(stress, axis=1)
        else:
            raise ValueError("Unknown approach. Avaiable approaches are: "
                             "'mean', 'max' or 'min'. Received: '{}'. "
                             "Please, check documentation for further details."
                             .format(approach))
        if effective and use_axis is not None:
            self.STATES, (_, key) = add_to_enum(self.STATES, "effective", key)
            spk_res = self._to_von_mises(spk_res)
            
        logger.debug("-spk_res.shape:'{}".format(spk_res.shape))
        # record explainable_metrics
        # exm = ExplainableMetric()
        # exm.key = key
        # exm.approach = approach
        # exm.speckles = spk
        # exm.used_reference_apex_base=True
        # self.explainable_metrics[self.states.get_spk_state_key(spk, key)] = exm
        self.states.add_spk_data(spk, key, spk_res)  # save to states
        return self.states.get_spk_data(spk, key)    # return pointer

    def compute_stress(self, spks, 
                            cylindrical=False, 
                            effective=False,
                            principal=True,
                            reduce_by={"group"}, **kwargs):
        
        # set key for this function
        if cylindrical:
            key = self.STATES.CYLINDRICAL_STRESS
        elif principal:
            key = self.STATES.PRINCIPAL_STRESS
        else:
            key = self.STATES.STRESS
        if effective:
            _, (_, key) = add_to_enum(self.STATES, "effective", key)
            
        logger.info("Computing metric '{}' with 'effective' set to '{}'".format(key, effective))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric
        res = [self.compute_spk_stress(spk, cylindrical=cylindrical, effective=effective, principal=principal, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean radius across entire LV)
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

    # -----------------------------
    # strain
    
    def compute_spk_strain(self, spk: object,
                            dtype: np.dtype = np.float64, 
                            approach="mean",
                            cylindrical=False, 
                            effective=False,
                            principal=True,
                            use_axis=None,
                            log_level=logging.INFO,
                            **kwargs):
        
        _strain_key = None
        if self.states.check_key(self.STATES.STRAIN):
            _strain_key = self.STATES.STRAIN
        elif self.states.check_key(self.STATES.LAGRANGE_STRAIN):
            _strain_key = self.STATES.LAGRANGE_STRAIN
        if _strain_key is None:
            raise RuntimeError("Could not found default strain values. Did you specify a custom strain key at self.STATES?")
        
        if cylindrical:
            key = self.STATES.CYLINDRICAL_STRAIN
            if not self.states.check_key(key):
                try:
                    if not self.states.check_key(_strain_key):
                        raise RuntimeError("Could not compute cylindrical strain. Did you compute/recorded strain?")
                    else:
                        cy_strain = self.convert_to_cylindrical_coordinates(_strain_key)
                        self.states.add(self.STATES.CYLINDRICAL_STRAIN, cy_strain)
                except:
                    raise RuntimeError("Could not compute cylindrical strain. Try adding it manually. ")
        elif principal:
            key = self.STATES.PRINCIPAL_STRAIN
            if not self.states.check_key(key):
                try:
                    if not self.states.check_key(_strain_key):
                        raise RuntimeError("Could not compute principal stress. Did you compute/recorded stress?")
                    else:
                        self.extract_principal_strain(_strain_key)
                except:
                    raise RuntimeError("Could not compute cylindrical stress. Try adding it manually. ")
        else:
            key = _strain_key
        
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle STRAIN for spk: '{}'".format(spk))
        logger.debug("Using approach: '{}'".format(approach))
        # check if strain variable exists in states
        if not self.states.check_key(key):
            raise RuntimeError("Could not find '{}' in states. Did you compute it?".format(key))
        # get strain value for each timestep
        strain = self.states.get(key, mask=spk.elem_ids)
        logger.debug("-strain.shape:'{}".format(strain.shape))
        if use_axis is not None:
            logger.debug("-use_axis:'{}".format(use_axis))
            strain = strain[:, use_axis]
            logger.debug("-strain.shape:'{}".format(strain.shape))
        if approach == "mean":
            # get centers data
            spk_res = np.mean(strain, axis=1)
        elif approach == "max":
            spk_res = np.max(strain, axis=1)
        elif approach == "min":
            spk_res = np.min(strain, axis=1)
        else:
            raise ValueError("Unknown approach. Avaiable approaches are: "
                             "'mean', 'max' or 'min'. Received: '{}'. "
                             "Please, check documentation for further details."
                             .format(approach))
        if effective and use_axis is not None:
            self.STATES, (_, key) = add_to_enum(self.STATES, "effective", key)
            spk_res = self._to_von_mises(spk_res)
            
        logger.debug("-spk_res.shape:'{}".format(spk_res.shape))
        # record explainable_metrics
        # exm = ExplainableMetric()
        # exm.key = key
        # exm.approach = approach
        # exm.speckles = spk
        # exm.used_reference_apex_base=True
        # self.explainable_metrics[self.states.get_spk_state_key(spk, key)] = exm
        self.states.add_spk_data(spk, key, spk_res)  # save to states
        return self.states.get_spk_data(spk, key)    # return pointer

    def compute_strain(self, spks, 
                            cylindrical=False, 
                            effective=False,
                            principal=True,
                            reduce_by={"group"}, **kwargs):
        
        # set key for this function
        if cylindrical:
            key = self.STATES.CYLINDRICAL_STRAIN
        elif principal:
            key = self.STATES.PRINCIPAL_STRAIN
        else:
            key = None
            if self.states.check_key(self.STATES.STRAIN):
                key = self.STATES.STRAIN
            elif self.states.check_key(self.STATES.LAGRANGE_STRAIN):
                key = self.STATES.LAGRANGE_STRAIN
            if key is None:
                raise RuntimeError("Could not found default strain values. Did you specify a custom strain key at self.STATES?")
        
        if effective:
            _, (_, key) = add_to_enum(self.STATES, "effective", key)
            
        logger.info("Computing metric '{}' with 'effective' set to '{}'".format(key, effective))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric
        res = [self.compute_spk_strain(spk, cylindrical=cylindrical, effective=effective, principal=principal, **kwargs) for spk in spks]
        # reduce metric (here we compute the mean radius across entire LV)
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

    

    # ===============================
    # Other
    # ===============================

    def apply_noise_filter(self, timeseries, 
                           mfilter_ws=0, 
                           sfilter_ws=0, 
                           sfilter_or=0,
                           keep_first=False,
                           keep_last=False):
        # reduce noise with filters
        new_ts = np.copy(timeseries)
        if mfilter_ws > 0 and len(timeseries) > mfilter_ws:
            from scipy import signal
            new_ts = signal.medfilt(timeseries, mfilter_ws)
            if keep_first:
                new_ts[0] = timeseries[0]
            if keep_last:
                new_ts[-1] = timeseries[-1]
        if sfilter_ws > 0 and len(timeseries) > sfilter_ws:
            from scipy import signal
            new_ts = signal.savgol_filter(timeseries, sfilter_ws, sfilter_or)
            if keep_first:
                new_ts[0] = timeseries[0]
            if keep_last:
                new_ts[-1] = timeseries[-1]
        return new_ts

    # ===============================
    # Generic methods
    # ===============================
    
    def _resolve_spk_args(self, spk_args):
        from collections.abc import Iterable
        if isinstance(spk_args, dict):
            spks = self.get_speckles(**spk_args)
            if len(spks) == 0:
                raise ValueError("No spks found for given spk_args: {}".format(spk_args))
        elif isinstance(spk_args, Iterable):
            if not issubclass(spk_args.__class__, SpeckeDeque):
                spks = SpeckeDeque([spk_args])
            else:
                spks = spk_args
        elif issubclass(spk_args.__class__, Speckle):
            spks = SpeckeDeque([spk_args])
        else:
            raise TypeError("'spks' must be a Iterable of spk objects."\
                            "The respective function argument can be either a "\
                            "dictionary containing 'get_speckles' args or one "\
                            "of the following types: 'list', 'tuple', 'np.ndarray'. "
                            "Received type: {}".format(type(spk_args))
                            )
        return spks
    
    def _reduce_metric(self, res, method="mean", axis=0, sum=False, 
                        mfilter_ws=0, sfilter_ws=0, sfilter_or=0, **kwargs):
        if method == "mean":
            res = np.mean(res, axis=axis)
        elif method == "max":
            res = np.max(res, axis=axis)
        elif method == "min":
            res = np.min(res, axis=axis)
        elif method == "median":
            res = np.median(res, axis=axis)
        elif method == "sum":
            res = np.sum(res, axis=axis)
        else:
            raise ValueError("Invalid method. Options are: 'mean', 'max', 'min', 'median', 'sum'.")
        if mfilter_ws > 0 or sfilter_ws > 0:
            res = self.apply_noise_filter(res, 
                mfilter_ws=mfilter_ws, sfilter_ws=sfilter_ws, sfilter_or=sfilter_or)
        if sum:
            return np.sum(res, axis=axis)
        else:
            if len(res) == 1:
                return res.reshape(-1,)
            else:
                return res

    def _save_metric(self, res, key):
        self.states.add(key, res)  # save to states
        return self.states.get(key)  # return pointer

    def _reduce_metric_and_save(self, res, key, **kwargs):
        res = self._reduce_metric(res, **kwargs)
        return self._save_metric(res, key)

    def _reduce_metric_by_group(self, spks, data_key, **kwargs):
        all_res = dict()
        for _, values in spks.by("group").items():
            res_key = values[0].group
            self.STATES, (_, key) = add_to_enum(self.STATES, data_key, res_key)
            res = [self.states.get_spk_data(spk, data_key) for spk in values]
            self.states.set_data_spk_rel(spks, key)
            if res_key not in all_res:
                all_res[res_key] = deque([])
            all_res[res_key].append(self._reduce_metric_and_save(res, key, **kwargs))
        return all_res

    def _reduce_metric_by_name(self, spks, data_key, **kwargs):
        all_res = dict()
        for _, values in spks.by("name").items():
            res_key = values[0].name
            self.STATES, (_, key) = add_to_enum(self.STATES, data_key, res_key)
            res = [self.states.get_spk_data(spk, data_key) for spk in values]
            self.states.set_data_spk_rel(spks, key)
            if res_key not in all_res:
                all_res[res_key] = deque([])
            all_res[res_key].append(self._reduce_metric_and_save(res, key, **kwargs))
        return all_res
    
    def _reduce_metric_by_subset(self, spks, data_key, merge_subset=None, **kwargs):
        all_res = dict()
        should_merge = False
        if merge_subset is not None:
            assert isinstance(merge_subset, dict), (
                "merge_subset must be dictionary with keys as which values to merge and "
                "values as the corresponding merging value. ")
            should_merge = True
             
        for _, values in spks.by("subset").items():
            res_key = values[0].subset
            if should_merge:
                if res_key in merge_subset:
                    res_key = merge_subset[res_key]
            self.STATES, (_, key) = add_to_enum(self.STATES, data_key, res_key)
            res = [self.states.get_spk_data(spk, data_key) for spk in values]
            self.states.set_data_spk_rel(spks, key)
            if res_key not in all_res:
                all_res[res_key] = deque([])
            all_res[res_key].append(self._reduce_metric_and_save(res, key, **kwargs))
        return all_res

    def _reduce_metric_by_group_and_name(self, spks, data_key, **kwargs):
        all_res = dict()
        for _, values in spks.by("group_name").items():
            res_key = (values[0].group, values[0].name)
            self.STATES, (_, key) = add_to_enum(self.STATES, data_key, *res_key)
            res = [self.states.get_spk_data(spk, data_key) for spk in values]
            self.states.set_data_spk_rel(spks, key)
            if res_key not in all_res:
                all_res[res_key] = deque([])
            all_res[res_key].append(self._reduce_metric_and_save(res, key, **kwargs))
        return all_res

    def _compute_relative_error(self, d1, d2):
        return (d2 - d1) / (d1 + self.EPSILON) * 100.0

    def _compute_spk_relative_error(self, spk, geo_key, cm_key, t_ed=0.0, switch_es=False, **kwargs):
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

    
    def _to_von_mises(self, data):
        """Computes the 'effective stress' based on von mises stress.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        xy = data[:, 3]
        yz = data[:, 4]
        xz = data[:, 5]
        
        vm = x**2 + y**2 + z**2
        vm -= x*y + y*z + x*z
        vm += 3*(xy**2 + yz**2 + xz**2)
        return np.sqrt(vm)