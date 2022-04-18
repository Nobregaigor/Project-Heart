
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

    # ---- Longitudinal shortening
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
            endo_spk, self.STATES.THICKENING, th)  # save to states
        self.states.add_spk_data(
            epi_spk, self.STATES.THICKENING, th)  # save to states
        # return pointer
        return self.states.get_spk_data(endo_spk, self.STATES.THICKENING)

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

    def compute_spk_rotation(self, spk,
                             t_ed: float = 0.0,
                             dtype: np.dtype = np.float64,
                             check_orientation=False,
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
