
import numpy as np
from .lv_speckles import LV_Speckles
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.spatial_utils import radius
from project_heart.modules.speckles.speckle import Speckle, SpeckeDeque
from project_heart.utils.spatial_utils import compute_longitudinal_length, compute_circumferential_length, compute_length_by_clustering
from project_heart.utils.vector_utils import angle_between
from project_heart.utils.enum_utils import add_to_enum
import logging

logger = logging.getLogger('LV.BaseMetricsComputations')

from collections import deque

class LVBaseMetricsComputations(LV_Speckles):
    def __init__(self, log_level=logging.INFO, *args, **kwargs):
        super(LVBaseMetricsComputations, self).__init__(log_level=log_level, *args, **kwargs)
        self.EPSILON = 1e-10
        self.metric_geochar_map = {
            self.STATES.LONGITUDINAL_SHORTENING.value: self.STATES.LONGITUDINAL_DISTANCE.value,
            self.STATES.RADIAL_SHORTENING.value: self.STATES.RADIUS.value,
            self.STATES.WALL_THICKENING.value: self.STATES.THICKNESS.value,
            self.STATES.LONG_STRAIN.value: self.STATES.LONG_LENGTH.value,
            self.STATES.CIRC_STRAIN.value: self.STATES.CIRC_LENGTH.value,
            self.STATES.TWIST.value: self.STATES.ROTATION.value,
            self.STATES.TORSION.value: self.STATES.ROTATION.value,
            self.STATES.ROTATION.value: self.STATES.ROTATION.value # required just for spk computation
        }
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

    # ---- Volume and volumetric fraction (ejection fraction)

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

    # =============================
    # 'Simple' computations
    # =============================

    def compute_base_apex_ref_over_timesteps(self,
                                             nodeset: str = None,
                                             dtype: np.dtype = np.float64,
                                             **kwargs
                                             ) -> np.ndarray:

        # check if xyz was computed; If not, try to compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get node ids (index array) from nodeset
        nodeset = self.REGIONS.EPI if nodeset is None else nodeset
        nodeids = self.get_nodeset(nodeset)
        # get node positions from nodeset at specified state
        xyz = self.states.get(self.STATES.XYZ, mask=nodeids)
        # compute distances for each timesteps
        base = np.zeros((len(xyz), 3), dtype=dtype)
        apex = np.zeros((len(xyz), 3), dtype=dtype)
        for i, pts in enumerate(xyz):
            # because nodes can shift position, we need to re-estimate
            # base and apex positions at each timestep.
            (es_base, es_apex), _ = self.est_apex_and_base_refs(pts, **kwargs)
            base[i] = es_base
            apex[i] = es_apex

        self.states.add(self.STATES.BASE_REF, base)  # save to states
        self.states.add(self.STATES.APEX_REF, apex)  # save to states

        # return pointers
        return (self.states.get(self.STATES.BASE_REF), self.states.get(self.STATES.APEX_REF))

    # ---- Longitudinal shortening

    def compute_nodeset_longitudinal_distance(self,
                                      nodeset: str,
                                      dtype: np.dtype = np.float64,
                                      apex_base_kwargs=None,
                                      ) -> np.ndarray:            
        # check if xyz was computed; If not, try to compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        if apex_base_kwargs is None:
                apex_base_kwargs = {}
        # get node ids (index array) from nodeset
        nodeids = self.get_nodeset(nodeset)
        # get node positions from nodeset at specified state
        xyz = self.states.get(self.STATES.XYZ, mask=nodeids)
        # compute distances for each timesteps
        dists = np.zeros(len(xyz), dtype=dtype)
        for i, pts in enumerate(xyz):
            # because nodes can shift position, we need to re-estimate
            # base and apex positions at each timestep.
            (es_base, es_apex), _ = self.est_apex_and_base_refs(pts, **apex_base_kwargs)
            dists[i] = np.linalg.norm(es_base - es_apex)
        # resolve reference key
        nodeset = self.check_enum(nodeset)
        self.STATES, (_, key) = add_to_enum(self.STATES, self.STATES.LONGITUDINAL_DISTANCE, nodeset)
        logger.info("State key added:'{}'".format(key))
        self.states.add(key, dists)  # save to states
        return self.states.get(key)  # return pointer

    def compute_longitudinal_distance(self, nodesets:set=None, dtype: np.dtype = np.float64) -> np.ndarray:
        # make sure we have endo and epi surface ids
        if nodesets is None:
            nodesets = {self.REGIONS.ENDO, self.REGIONS.EPI}
        # compute long dists for each nodeset
        res = [self.compute_nodeset_longitudinal_distance(key, dtype) for key in nodesets]
        # reduce metric and return pointer
        return self._reduce_metric_and_save(res, self.STATES.LONGITUDINAL_DISTANCE)

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

    # ===============================
    # Spk computations
    # ===============================
    
    def compute_spk_centers_over_timesteps(self, spk, 
                                           apex_base_kwargs=None, 
                                           log_level=logging.INFO,
                                           **kwargs):
        # check for speckle input
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle center for spk: '{}'".format(spk))
        # check if apex and base references were computed
        if not self.states.check_key(self.STATES.BASE_REF) or \
            not self.states.check_key(self.STATES.APEX_REF):
            if apex_base_kwargs is None:
                apex_base_kwargs = {}
            self.compute_base_apex_ref_over_timesteps(**apex_base_kwargs)
        # compute spk centers over timesteps based on 'k' height
        from project_heart.utils.spatial_utils import get_p_along_line
        k = spk.k
        apex_ts = self.states.get(self.STATES.APEX_REF)
        base_ts = self.states.get(self.STATES.BASE_REF)
        spk_res = [get_p_along_line(k, [apex,base]) for apex, base in zip(apex_ts, base_ts)]
        spk_res = np.vstack(spk_res)
        logger.debug("-k: '{}'\n-apex:'{}\n-base:'{}'\n-centers:'{}'".
                     format(k, apex_ts, base_ts, spk_res))
        self.states.add_spk_data(spk, self.STATES.CENTERS, spk_res)  # save to states
        return self.states.get_spk_data(spk, self.STATES.CENTERS) # return pointer
    
    # ---------------------------
    # ---- Radial shortening ---- 

    # ---------- Geo metric

    def compute_spk_radius(self, spk: object, 
                           dtype: np.dtype = np.float64, 
                           approach="moving_centers",
                           log_level=logging.INFO,
                           **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        logger.setLevel(log_level)
        logger.debug("Computing speckle radius for spk: '{}'".format(spk))
        logger.debug("Using approach: '{}'".format(approach))
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get nodal position for all timesteps
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        if approach == "moving_centers":
            # check if speckle centers were computed. If not, compute them.
            if not self.states.check_spk_key(spk, self.STATES.CENTERS):
                self.compute_spk_centers_over_timesteps(spk, log_level=log_level, **kwargs)
            # get centers data
            centers = self.states.get_spk_data(spk, self.STATES.CENTERS)
            spk_res = np.array([radius(coords, center=center) for coords, center in zip(xyz, centers)], 
                            dtype=dtype)
        elif approach == "fixed_centers":
            spk_res = np.array([radius(coords, center=spk.center) for coords in xyz], 
                            dtype=dtype)
        elif approach == "moving_vector":
            from project_heart.utils.vector_utils import dist_from_line
            # check if speckle apex and base values were computed. If not, compute them.
            if not self.states.check_key(self.STATES.BASE_REF) or \
            not self.states.check_key(self.STATES.APEX_REF):
                self.compute_base_apex_ref_over_timesteps(spk, log_level=log_level, **kwargs)
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
            raise ValueError("Unknown method. Avaiable methods are: "
                             "'moving_centers', 'fixed_centers', 'moving_vector', 'fixed_vector'."
                             "Please, check documentation for further details.")
            
        logger.debug("-mean_coords:'{}\n-radius:'{}'".
                     format(np.mean(xyz, axis=0), spk_res))
        self.states.add_spk_data(spk, self.STATES.RADIUS, spk_res)  # save to states
        return self.states.get_spk_data(spk, self.STATES.RADIUS) # return pointer

    def compute_radius(self, spks, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.RADIUS
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric
        res = [self.compute_spk_radius(spk, **kwargs) for spk in spks]
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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))

    # ---------- Clinical metric

    def compute_radial_shortening(self, spks, t_ed=0.0, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.RADIAL_SHORTENING
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for all spks
        res = [self._compute_spk_relative_error(spk, 
                    self.STATES.RADIUS, key,
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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    # ---------------------------
    # ---- Wall thickneing ---- 

    # ---------- Geo metric

    def compute_spk_thickness(self, endo_spk, epi_spk, 
                              log_level=logging.INFO,
                              **kwargs):
        assert self.check_spk(
            endo_spk), "endo_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            epi_spk), "epi_spk must be a valid 'Speckle' object."

        # check if radius were computed for ENDOCARDIUM
        if not self.states.check_spk_key(endo_spk, self.STATES.RADIUS):
            try:
                self.compute_radius(endo_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for endo spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIUS' manually."
                    .format(endo_spk.str))
        # check if radius were computed for EPICARDIUM
        if not self.states.check_spk_key(epi_spk, self.STATES.RADIUS):
            try:
                self.compute_radius(epi_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for epi spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIUS' manually."
                    .format(epi_spk.str))
        logger.setLevel(log_level)
        logger.debug("Computing speckle thickness for spks: '{}' and '{}"
                     .format(endo_spk, epi_spk))
        r_endo = self.states.get_spk_data(endo_spk, self.STATES.RADIUS)
        r_epi = self.states.get_spk_data(epi_spk, self.STATES.RADIUS)
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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(endo_spks, key, **kwargs)
            self._reduce_metric_by_group_and_name(epi_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))

    # ---------- Clinical metric

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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(endo_spks, key, **kwargs)
            self._reduce_metric_by_group_and_name(epi_spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    # ---------------------------
    # ---- Longitudinal Strain ---- 

    # ---------- Geo metric 

    def compute_spk_longitudinal_length(self,
                                        spk,
                                        method = "fast",
                                        mfilter_ws=0,
                                        sfilter_ws=0,
                                        sfilter_or=0,
                                        dtype: np.dtype = np.float64,
                                        **kwargs):
        from functools import partial
        # check for valid spk object
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # resolve computation method
        if method == "fast":
            fun = partial(compute_longitudinal_length, **kwargs)
        elif method == "clusters":
            fun = partial(compute_length_by_clustering, **kwargs)
        else:
            raise ValueError("Unknown method '{}'. options are: 'fast', 'clusters'.")
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        # compute response for given spk
        spk_res = np.array([fun(coords) for coords in xyz], dtype=dtype)
        # apply filter (if requested)
        spk_res = self.apply_noise_filter(spk_res, 
                        mfilter_ws=mfilter_ws, 
                        sfilter_ws=sfilter_ws,
                        sfilter_or=sfilter_or)
        # save to states
        self.states.add_spk_data(spk, self.STATES.LONG_LENGTH, spk_res)
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.LONG_LENGTH)

    def compute_longitudinal_length(self, spks, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.LONG_LENGTH
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for each speckle -> results in the length of each spk
        res = [self.compute_spk_longitudinal_length(spk, **kwargs) for spk in spks]
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
        return self.states.get(key)  # return pointer

    # ---------- Clinical metric

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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    # ---------------------------
    # ---- Circumferential Strain 

    # ---------- Geo metric

    def compute_spk_circumferential_length(self, spk, method="fast",
                                           mfilter_ws=0,
                                           sfilter_ws=0,
                                           sfilter_or=0,
                                           dtype: np.dtype = np.float64,
                                           **kwargs):
        from functools import partial
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # resolve computation method
        if method == "fast":
            fun = partial(compute_circumferential_length, **kwargs)
        elif method == "clusters":
            fun = partial(compute_length_by_clustering, **kwargs)
        else:
            raise ValueError("Unknown method '{}'. options are: 'fast', 'clusters'.")
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        # compute response for given spk
        spk_res = np.array([fun(coords) for coords in xyz], dtype=dtype)
        # apply filter (if requested)
        spk_res = self.apply_noise_filter(spk_res, 
                        mfilter_ws=mfilter_ws, 
                        sfilter_ws=sfilter_ws,
                        sfilter_or=sfilter_or)
        # save to states
        self.states.add_spk_data(spk, self.STATES.CIRC_LENGTH, spk_res)
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.CIRC_LENGTH)

    def compute_circumferential_length(self, spks, reduce_by={"group"}, **kwargs):
        # set key for this function
        key = self.STATES.CIRC_LENGTH
        logger.info("Computing metric '{}'".format(key))
        # resolve spks (make sure you have a SpeckeDeque)
        spks = self._resolve_spk_args(spks)
        # compute metric for each speckle -> results in the length of each spk
        res = [self.compute_spk_circumferential_length(spk, **kwargs) for spk in spks]
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
        return self.states.get(key)  # return pointer

    # ---------- Clinical metric
    
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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))
        return self.states.get(key, t=t_ed)

    # ---------------------------
    # ----- Rotation ---- 

    def compute_spk_vectors(self, spk, dtype: np.dtype = np.float64, **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # check if speckle centers were computed. If not, compute them.
        if not self.states.check_spk_key(spk, self.STATES.CENTERS):
            self.compute_spk_centers_over_timesteps(spk, **kwargs)
        # get centers data
        centers = self.states.get_spk_data(spk, self.STATES.CENTERS)
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        vecs = np.array(xyz - centers, dtype=dtype)
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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(spks, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))

    # ---------------------------
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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(apex_spk, key, **kwargs)
            self._reduce_metric_by_group_and_name(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))

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
        D = np.linalg.norm(apex_spk.center - base_spk.center)
        torsion = twist / D  # units = angle/mm
        if relative:  # relative will multiply by the mean radius of base and apex -> units: angle
            c = 0.5 * (r_apex + r_base)
            torsion *= c

        self.states.add_spk_data(
            apex_spk, self.STATES.TORSION, torsion)  # save to states
        self.states.add_spk_data(
            base_spk, self.STATES.TORSION, torsion)  # save to states
        return self.states.get_spk_data(base_spk, self.STATES.TORSION)

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
        if "group_name" in reduce_by:
            logger.debug("Reducing metric by group and name for '{}'".format(key))
            self._reduce_metric_by_group_and_name(apex_spk, key, **kwargs)
            self._reduce_metric_by_group_and_name(base_spk, key, **kwargs)
            logger.debug("Metric '{}' has reduced values by group and name.".format(key))

    # ===============================
    # Other
    # ===============================

    def apply_noise_filter(self, timeseries, mfilter_ws=0, sfilter_ws=0, sfilter_or=0):
        # reduce noise with filters
        if mfilter_ws > 0 and len(timeseries) > mfilter_ws:
            from scipy import signal
            timeseries = signal.medfilt(timeseries, mfilter_ws)
        if sfilter_ws > 0 and len(timeseries) > sfilter_ws:
            from scipy import signal
            timeseries = signal.savgol_filter(timeseries, sfilter_ws, sfilter_or)
        return timeseries

    # ===============================
    # Generic spk compilation
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
            return res.reshape(-1,)

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