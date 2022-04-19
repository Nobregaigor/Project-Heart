
import numpy as np
from .lv_speckles import LV_Speckles
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.spatial_utils import radius
from project_heart.modules.speckles.speckle import Speckle
from project_heart.utils.spatial_utils import compute_longitudinal_length, compute_circumferential_length
from project_heart.utils.vector_utils import angle_between


class LVBaseMetricsComputations(LV_Speckles):
    def __init__(self, *args, **kwargs):
        super(LVBaseMetricsComputations, self).__init__(*args, **kwargs)
        self.EPSILON = 1e-10
        self.metric_geochar_map = {
            self.STATES.LONGITUDINAL_SHORTENING.value: self.STATES.LONGITUDINAL_DISTANCES.value,
            self.STATES.RADIAL_SHORTENING.value: self.STATES.RADIUS.value,
            self.STATES.WALL_THICKENING.value: self.STATES.THICKNESS.value,
            self.STATES.LONG_STRAIN.value: self.STATES.LONG_LENGTH.value,
            self.STATES.CIRC_STRAIN.value: self.STATES.CIRC_LENGTH.value,
            self.STATES.TWIST.value: self.STATES.ROTATION.value,
            self.STATES.TORSION.value: self.STATES.ROTATION.value,
            self.STATES.ROTATION.value: self.STATES.ROTATION.value # required just for spk computation
        }

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
        base = np.zeros(len(xyz), dtype=dtype)
        apex = np.zeros(len(xyz), dtype=dtype)
        for i, pts in enumerate(xyz):
            # because nodes can shift position, we need to re-estimate
            # base and apex positions at each timestep.
            (es_base, es_apex), _ = self.est_apex_and_base_refs(pts)
            base[i] = es_base
            apex[i] = es_apex

        self.states.add(self.STATES.BASE_REF, base)  # save to states
        self.states.add(self.STATES.APEX_REF, apex)  # save to states

        # return pointers
        return (self.states.get(self.STATES.BASE_REF), self.states.get(self.STATES.APEX_REF))

    # ---- Longitudinal shortening

    def compute_longitudinal_distance(self,
                                      nodeset: str = None,
                                      dtype: np.dtype = np.float64,
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
        dists = np.zeros(len(xyz), dtype=dtype)
        for i, pts in enumerate(xyz):
            # because nodes can shift position, we need to re-estimate
            # base and apex positions at each timestep.
            (es_base, es_apex), _ = self.est_apex_and_base_refs(pts)
            dists[i] = np.linalg.norm(es_base - es_apex)
        self.states.add(self.STATES.LONG_DISTS, dists)  # save to states
        return self.states.get(self.STATES.LONG_DISTS)  # return pointer

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

    # ---- Radial shortening

    def compute_spk_radius_for_each_timestep(self, spk: object, dtype: np.dtype = np.float64):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get speckle center and node ids from speckle
        spk_center = spk.center
        nodeids = spk.ids
        # get nodal position for all timesteps
        xyz = self.states.get(self.STATES.XYZ, mask=nodeids)
        rvec = np.zeros(len(xyz), dtype=dtype)
        for i, pts in enumerate(xyz):
            rvec[i] = radius(pts)
        self.states.add_spk_data(
            spk, self.STATES.RADIUS, rvec)  # save to states
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.RADIUS)

    def compute_spk_radial_shortening(self, spk: object, t_ed=0.0, **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        if not self.states.check_spk_key(spk, self.STATES.RADIUS):
            try:
                self.compute_spk_radius_for_each_timestep(spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIUS' manually."
                    .format(spk.str))

        d2 = self.states.get_spk_data(spk, self.STATES.RADIUS)
        d1 = self.states.get_spk_data(spk, self.STATES.RADIUS, t=t_ed)
        # compute % shortening
        rs = (d1 - d2) / (d1 + self.EPSILON) * 100.0
        self.states.add_spk_data(spk, self.STATES.RS, rs)  # save to states
        return self.states.get_spk_data(spk, self.STATES.RS)  # return pointer

    # ---- Wall thickneing

    def compute_spk_thickness(self, endo_spk, epi_spk, **kwargs):
        assert self.check_spk(
            endo_spk), "endo_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            epi_spk), "epi_spk must be a valid 'Speckle' object."

        # check if radius were computed for ENDOCARDIUM
        if not self.states.check_spk_key(endo_spk, self.STATES.RADIUS):
            try:
                self.compute_spk_radius_for_each_timestep(endo_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for endo spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIUS' manually."
                    .format(endo_spk.str))
        # check if radius were computed for EPICARDIUM
        if not self.states.check_spk_key(epi_spk, self.STATES.RADIUS):
            try:
                self.compute_spk_radius_for_each_timestep(epi_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute radius data for epi spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'RADIUS' manually."
                    .format(epi_spk.str))

        r_endo = self.states.get_spk_data(endo_spk, self.STATES.RADIUS)
        r_epi = self.states.get_spk_data(epi_spk, self.STATES.RADIUS)
        thickness = r_epi - r_endo
        self.states.add_spk_data(
            endo_spk, self.STATES.THICKNESS, thickness)  # save to states
        self.states.add_spk_data(
            epi_spk, self.STATES.THICKNESS, thickness)  # save to states
        # return pointer
        return self.states.get_spk_data(endo_spk, self.STATES.THICKNESS)

    def compute_spk_thickening(self, endo_spk, epi_spk, t_ed: float = 0.0, **kwargs):

        assert self.check_spk(
            endo_spk), "endo_spk must be a valid 'Speckle' object."
        assert self.check_spk(
            epi_spk), "epi_spk must be a valid 'Speckle' object."

        # check if thickness was computed for given spk
        if not self.states.check_spk_key(endo_spk, self.STATES.THICKNESS):
            try:
                self.compute_spk_thickness(endo_spk, epi_spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute thickness data for spks '{}' and '{}."
                    "Please, either verify required data or add"
                    "state data for 'THICKNESS' manually."
                    .format(endo_spk.str, epi_spk.str))

        d1 = self.states.get_spk_data(endo_spk, self.STATES.THICKNESS)
        d2 = self.states.get_spk_data(endo_spk, self.STATES.THICKNESS, t=t_ed)
        # compute % thickening
        th = (d1 - d2) / (d1 + self.EPSILON) * 100.0
        self.states.add_spk_data(
            endo_spk, self.STATES.WT, th)  # save to states
        self.states.add_spk_data(
            epi_spk, self.STATES.WT, th)  # save to states
        # return pointer
        return self.states.get_spk_data(endo_spk, self.STATES.WT)

    # ---- Longitudinal Strain

    def compute_spk_longitudinal_length(self,
                                        spk,
                                        mfilter_ws=3,
                                        sfilter_ws=9,
                                        sfilter_or=3,
                                        dtype: np.dtype = np.float64,
                                        **kwargs):

        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        # compute longitudinal length for each timestep
        ll = np.zeros(len(xyz), dtype=dtype)
        for i, pts in enumerate(xyz):
            ll[i] = compute_longitudinal_length(pts, **kwargs)
        # reduce noise with filters
        if mfilter_ws > 0 and len(ll) > mfilter_ws:
            from scipy import signal
            ll = signal.medfilt(ll, mfilter_ws)
        if sfilter_ws > 0 and len(ll) > sfilter_ws:
            from scipy import signal
            ll = signal.savgol_filter(ll, sfilter_ws, sfilter_or)
        # save to states
        self.states.add_spk_data(
            spk, self.STATES.LONG_LENGTH, ll)
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.LONG_LENGTH)

    def compute_spk_longitudinal_strain(self, spk, t_ed: float = 0.0, **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        if not self.states.check_spk_key(spk, self.STATES.LONG_LENGTH):
            try:
                self.compute_spk_longitudinal_length(spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute longitudinal length data for spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'LONG_LENGTH' manually."
                    .format(spk.str))
        d2 = self.states.get_spk_data(spk, self.STATES.LONG_LENGTH)
        d1 = self.states.get_spk_data(spk, self.STATES.LONG_LENGTH, t=t_ed)
        # compute % shortening
        sl = (d1 - d2) / (d1 + self.EPSILON) * 100.0
        self.states.add_spk_data(
            spk, self.STATES.LONG_STRAIN, sl)  # save to states
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.LONG_STRAIN)

    # ---- Circumferential Strain

    def compute_spk_circumferential_length(self,
                                           spk,
                                           dtype: np.dtype = np.float64,
                                           mfilter_ws=3,
                                           sfilter_ws=9,
                                           sfilter_or=3,
                                           **kwargs):

        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=spk.ids)
        ll = np.zeros(len(xyz), dtype=dtype)
        # compute circumferential length for each timestep
        for i, pts in enumerate(xyz):
            ll[i] = compute_circumferential_length(pts, **kwargs)
        # reduce noise with filters
        if mfilter_ws > 0 and len(ll) > mfilter_ws:
            from scipy import signal
            ll = signal.medfilt(ll, mfilter_ws)
        if sfilter_ws > 0 and len(ll) > sfilter_ws:
            from scipy import signal
            ll = signal.savgol_filter(ll, sfilter_ws, sfilter_or)
        # save to states
        self.states.add_spk_data(
            spk, self.STATES.CIRC_LENGTH, ll)
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.CIRC_LENGTH)

    def compute_spk_circumferential_strain(self, spk, t_ed: float = 0.0, **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        if not self.states.check_spk_key(spk, self.STATES.CIRC_LENGTH):
            try:
                self.compute_spk_circumferential_length(spk, **kwargs)
            except:
                raise RuntimeError(
                    "Unable to compute circumferential length data for spk '{}'."
                    "Please, either verify required data or add"
                    "state data for 'CIRC_LENGTH' manually."
                    .format(spk.str))
        d2 = self.states.get_spk_data(spk, self.STATES.CIRC_LENGTH)
        d1 = self.states.get_spk_data(spk, self.STATES.CIRC_LENGTH, t=t_ed)
        # compute % shortening
        sl = (d1 - d2) / (d1 + self.EPSILON) * 100.0
        self.states.add_spk_data(
            spk, self.STATES.CIRC_STRAIN, sl)  # save to states
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.CIRC_STRAIN)

    # ----- Rotation

    def compute_spk_vectors(self, spk, dtype: np.dtype = np.float64, **kwargs):
        assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
        # check if xyz was computed, otherwise try to automatically compute it.
        if not self.states.check_key(self.STATES.XYZ):
            self.compute_xyz_from_displacement()
        # get speckle center and node ids from speckle
        spk_center = spk.center
        nodeids = spk.ids
        # get nodal position for all timesteps for given spk
        xyz = self.states.get(self.STATES.XYZ, mask=nodeids)
        vecs = xyz - spk_center
        self.states.add_spk_data(
            spk, self.STATES.SPK_VECS, vecs)  # save to states
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

        d2 = self.states.get_spk_data(spk, self.STATES.SPK_VECS)
        d1 = self.states.get_spk_data(spk, self.STATES.SPK_VECS, t=t_ed)
        # compute rotation length for each timestep
        rot = np.zeros(len(d2), dtype=dtype)
        for i, (xyz1, xyz2) in enumerate(zip(d1, d2)):
            rot[i] = np.mean(angle_between(
                xyz2, xyz1, check_orientation=check_orientation))
        if degrees:
            rot = np.degrees(rot)
        self.states.add_spk_data(
            spk, self.STATES.ROTATION, rot)  # save to states
        # return pointer
        return self.states.get_spk_data(spk, self.STATES.ROTATION)

    # ----- Twist and torsion

    def compute_spk_twist(self,
                          apex_spk,
                          base_spk,
                          t_ed: float = 0.0,
                          **kwargs):
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
        self.states.add_spk_data(
            apex_spk, self.STATES.TWIST, twist)  # save to states
        self.states.add_spk_data(
            base_spk, self.STATES.TWIST, twist)  # save to states
        # return pointer
        return self.states.get_spk_data(apex_spk, self.STATES.TWIST)

    def compute_spk_torsion(self,
                            apex_spk,
                            base_spk,
                            t_ed: float = 0.0,
                            relative=False,
                            **kwargs):
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

    # ===============================
    # Generic spk compilation
    # ===============================
    
    def _reduce_metric(self, res, method="mean", axis=0, **kwargs):
        if method == "mean":
            return np.mean(res, axis=axis)
        elif method == "max":
            return np.max(res, axis=axis)
        elif method == "min":
            return np.min(res, axis=axis)
        elif method == "median":
            return np.median(res, axis=axis)
        else:
            raise ValueError("Invalid method. Options are: 'mean', 'max', 'min', 'median'.")
    
    def _save_metric(self, res, key):
        self.states.add(key, res)  # save to states
        return self.states.get(key)  # return pointer
    
    def _resolve_spk_args(self, spk_args):
        if isinstance(spk_args, dict):
            spks = self.get_speckles(**spk_args)
            if len(spks) == 0:
                raise ValueError("No spks found for given spk_args: {}".format(spk_args))
        elif isinstance(spk_args, (list, tuple, np.ndarray)):
            spks = spk_args
        else:
            raise TypeError("'spks' must be a list (interable) of spk objects."\
                            "The respective function argument can be either a "\
                            "dictionary containing 'get_speckles' args or one "\
                            "of the following types: 'list', 'tuple', 'np.ndarray'.")
        return spks
    
    def _reduce_geochar(self, spk_args, key, **kwargs):
        from collections import deque
        spks = self._resolve_spk_args(spk_args)
        key = self.check_enum(key)

        if key in self.metric_geochar_map:
            geo_key = self.metric_geochar_map[key]
            res = deque()
            for spk in spks:
                res.append(self.states.get_spk_data(spk, geo_key))
            # save for reduced for each group
            res = self._reduce_metric(res, **kwargs)
            return self._save_metric(res, geo_key)

    def _compute_metric_from_spks(self, 
                                  spk_args, 
                                  fun, 
                                  key,
                                  reduce:str="mean",
                                  geochar=True,
                                  dtype:np.dtype=np.float64,
                                  **kwargs):
        spks = self._resolve_spk_args(spk_args)
        res = np.zeros((len(spks), self.states.n()), dtype=dtype)
        for i, spk in enumerate(spks):
            try:
                res[i] = fun(spk, dtype=dtype, **kwargs)
            except:
                assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
                raise RuntimeError("Unable to compute metric '{}' for spk '{}'"
                                   .format(fun, spk.str))
        res = self._reduce_metric(res, method=reduce, axis=0)
        self._save_metric(res, key)
        if geochar:
            self._reduce_geochar(spk_args, key, method=reduce, axis=0)
        return self.states.get(key)  # return pointer
    
    def _compute_metric_from_coupled_spks(self, 
                                          spk_args_1, 
                                          spk_args_2, 
                                          fun, 
                                          key,
                                          reduce:str="mean",
                                          geochar=True,
                                          dtype:np.dtype=np.float64,
                                          **kwargs):
        spks_1 = self._resolve_spk_args(spk_args_1)
        spks_2 = self._resolve_spk_args(spk_args_2)  
              
        if len(spks_1) != len(spks_2):
            raise ValueError("Number of speckles must match for coupled computation: "
                             "{}, {}".format(len(spks_1), len(spks_2)))

        res = np.zeros((len(spks_1), self.states.n()), dtype=dtype)
        for i, (spk1, spk2) in enumerate(zip(spks_1, spks_2)):
            try:
                res[i] = fun(spk1, spk2, dtype=dtype, **kwargs)
            except:
                assert self.check_spk(spk), "Spk must be a valid 'Speckle' object."
                raise RuntimeError("Unable to compute metric '{}' for spk '{}'"
                                   .format(fun, spk.str))
        res = self._reduce_metric(res, method=reduce, axis=0)
        self._save_metric(res, key)
        if geochar:
            spks_1.extend(spks_2)
            self._reduce_geochar(spks_1, key, method=reduce, axis=0)
        return self.states.get(key)  # return pointer
    
    def _reduce_metric_from_spks(self, spk_args, key, by, geochar=True, **kwargs):
        from collections import deque
        
        spks = self._resolve_spk_args(spk_args)
        key = self.check_enum(key)
        by = self.check_enum(by)
        grouped = {}
        if by == self.SPK_SETS.GROUP.value:
            # Group spks
            for spk in spks:
                if spk.group not in grouped:
                    grouped[spk.group] = deque([spk])
                else:
                    grouped[spk.group].append(spk)
        elif by == self.SPK_SETS.NAME.value:
            # Group spks
            for spk in spks:
                if spk.name not in grouped:
                    grouped[spk.name] = deque([spk])
                else:
                    grouped[spk.name].append(spk)
        elif by == self.SPK_SETS.GROUP_NAME.value:
            # Group spks
            for spk in spks:
                if (spk.group, spk.name) not in grouped:
                    grouped[(spk.group, spk.name)] = deque([spk])
                else:
                    grouped[(spk.group, spk.name)].append(spk)
        else:
            raise ValueError("Reduce by options are: name, group, group_name.")
        
        # Retrieve data
        for group_key, group_spks in grouped.items():
            res = deque()
            for spk in grouped[group_key]:
                res.append(self.states.get_spk_data(spk, key))
            # save for reduced for each group
            res = self._reduce_metric(res, **kwargs)
            self.states.add_spk_data(spk, key, res, by)

            if geochar and key in self.metric_geochar_map:
                geo_key = self.metric_geochar_map[key]
                res = deque()
                for spk in grouped[group_key]:
                    res.append(self.states.get_spk_data(spk, geo_key))
                # save for reduced for each group
                res = self._reduce_metric(res, **kwargs)
                self.states.add_spk_data(spk, geo_key, res, by)
    
    def _apply_generic_spk_metric_schematics(self, 
        spks_1, 
        key,
        fun,
        t_ed=0.0,
        spks_2=None,
        reduce_additional_info=True,
        reduce="mean",
        geochar=True,
        recompute=False,
        **kwargs
        ):
        
        if spks_2 is None:
            # check if metric 'key' was computed
            if not self.states.check_key(key) or recompute:
                if t_ed is not None:
                    self._compute_metric_from_spks(spks_1,fun,key,t_ed=t_ed, geochar=geochar, **kwargs)
                else:
                    self._compute_metric_from_spks(spks_1,fun,key, geochar=geochar, **kwargs)
                # compute additional info 
                if reduce_additional_info:
                    # reduce metric
                    self._reduce_metric_from_spks(spks_1, key, self.SPK_SETS.GROUP, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_1, key, self.SPK_SETS.NAME, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_1, key, self.SPK_SETS.GROUP_NAME, reduce=reduce, geochar=geochar)
                
                # save spks used to compute this metric
                self.states.set_data_spk_rel(spks_1, key)
                if geochar:
                    key = self.check_enum(key)
                    self.states.set_data_spk_rel(spks_1, self.metric_geochar_map[key])
        else:
            # check if metric 'key' was computed
            if not self.states.check_key(key) or recompute:
                if t_ed is not None:
                    self._compute_metric_from_coupled_spks(spks_1,spks_2,fun,key,t_ed=t_ed,geochar=geochar,**kwargs)
                else:
                    self._compute_metric_from_coupled_spks(spks_1,spks_2,fun,key,geochar=geochar,**kwargs)
                # compute additional info 
                if reduce_additional_info:
                    # reduce metric
                    self._reduce_metric_from_spks(spks_1, key, self.SPK_SETS.GROUP, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_1, key, self.SPK_SETS.NAME, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_1, key, self.SPK_SETS.GROUP_NAME, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_2, key, self.SPK_SETS.GROUP, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_2, key, self.SPK_SETS.NAME, reduce=reduce, geochar=geochar)
                    self._reduce_metric_from_spks(spks_2, key, self.SPK_SETS.GROUP_NAME, reduce=reduce, geochar=geochar)
                
                # save spks used to compute this metric
                spks_1 = self._resolve_spk_args(spks_1)
                spks_2 = self._resolve_spk_args(spks_2)
                spks_1.extend(spks_2)
                self.states.set_data_spk_rel(spks_1, key)
                if geochar:
                    key = self.check_enum(key)
                    self.states.set_data_spk_rel(spks_1, self.metric_geochar_map[key])
 
