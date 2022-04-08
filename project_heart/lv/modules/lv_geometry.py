from project_heart.modules.geometry import Geometry
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import *
from project_heart.utils.cloud_ops import *
from collections import deque

from project_heart.enums import *
from sklearn.cluster import KMeans

from functools import reduce

from pathlib import Path
import os


class LV_Geometry(Geometry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rot_chain = deque()
        self._normal = None

        self.aortic_info = {}
        self.mitral_info = {}

        self._aligment_data = {}

        # self._centroid = self.est_centroid()

        # ------ default values
        self._default_fiber_markers = markers = {
            "epi": LV_SURFS.EPI.value,
            "lv": LV_SURFS.ENDO.value,
            "base": LV_SURFS.BASE_BORDER.value
        }

        # ------ Flags
        self._surfaces_identified_with_class_method = False

    def est_centroid(self) -> np.ndarray:
        """Estimates the centroid of the geometry based on surface mesh.

        Returns:
            np.ndarray: [x,y,z] coordinates of center
        """
        lvsurf = self.get_surface_mesh()
        # center = np.mean(lvsurf.points, axis=0)
        return centroid(lvsurf.points)  # center

    @staticmethod
    def est_apex_ref(points, ql=0.03, **kwargs):
        zvalues = points[:, 2]
        thresh = np.quantile(zvalues, ql)
        apex_region_idxs = np.where(zvalues <= thresh)[0]
        apex_region_pts = points[apex_region_idxs]
        return np.mean(apex_region_pts, 0), apex_region_idxs

    @staticmethod
    def est_base_ref(points, qh=0.90, **kwargs):
        zvalues = points[:, 2]
        thresh = np.quantile(zvalues, qh)
        base_region_idxs = np.where(zvalues >= thresh)[0]
        base_region_pts = points[base_region_idxs]
        base_ref = np.mean(base_region_pts, 0)
        return base_ref, base_region_idxs

    @staticmethod
    def est_apex_and_base_refs(points, **kwargs):
        apex_ref, apex_region_idxs = LV_Geometry.est_apex_ref(points, **kwargs)
        base_ref, base_region_idxs = LV_Geometry.est_base_ref(points, **kwargs)
        info = {
            "apex_ref": apex_ref,
            "base_ref": base_ref,
            "apex_region": apex_region_idxs,
            "base_region": base_region_idxs}
        return np.vstack((base_ref, apex_ref)), info

    def est_pts_aligment_with_lv_normal(self, points, n=5, rot_chain=[], **kwargs):
        pts = np.copy(points)
        for _ in range(n):
            long_line, _ = LV_Geometry.est_apex_and_base_refs(pts, **kwargs)
            lv_normal = unit_vector(long_line[0] - long_line[1])
            curr_rot = get_rotation(lv_normal, self._Z)
            pts = curr_rot.apply(pts)
            rot_chain.append(curr_rot)
        long_line, info = LV_Geometry.est_apex_and_base_refs(pts, **kwargs)
        lv_normal = unit_vector(long_line[0] - long_line[1])
        info["rot_pts"] = pts
        info["normal"] = lv_normal
        info["long_line"] = long_line
        info["rot_chain"] = rot_chain
        self._aligment_data = info
        return info

    def identify_base_and_apex_surfaces(self, ab_n=10, ab_ql=0.03, ab_qh=0.75, **kwargs):
        """
        """
        # extract surface mesh (use extract)
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points

        # split the surface into two main clusters
        # this will result in top and bottom halves
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pts)
        kcenters = kmeans.cluster_centers_
        kcenters = kcenters[np.argsort(kcenters[:, -1])]

        rot_chain = deque()
        # set vectors for rotation
        edge_long_vec = kcenters[1] - kcenters[0]
        # get rotation matrix
        rot = get_rotation(edge_long_vec, self._Z)
        rot_chain.append(rot)
        initial_rot = rot.apply(pts)
        aligment_data = self.est_pts_aligment_with_lv_normal(initial_rot,
                                                             rot_chain=rot_chain,
                                                             n=ab_n, ql=ab_ql, qh=ab_qh)

        # set surface region ids (for surface mesh)
        surf_regions = np.zeros(len(pts))
        surf_regions[aligment_data["apex_region"]] = LV_SURFS.APEX_REGION
        surf_regions[aligment_data["base_region"]] = LV_SURFS.BASE_REGION
        self._surface_mesh.point_data[LV_MESH_DATA.APEX_BASE_REGIONS.value] = surf_regions.astype(
            np.int64)

        # set global region ids (for entire mesh)
        surf_to_global = lvsurf.point_data["vtkOriginalPointIds"]
        global_regions = np.zeros(self.mesh.n_points)
        global_regions[surf_to_global[aligment_data["apex_region"]]
                       ] = LV_SURFS.APEX_REGION
        global_regions[surf_to_global[aligment_data["base_region"]]
                       ] = LV_SURFS.BASE_REGION
        self.mesh.point_data[LV_MESH_DATA.APEX_BASE_REGIONS.value] = global_regions.astype(
            np.int64)

        # save virtual nodes
        apex_pt = np.mean(pts[aligment_data["apex_region"]], axis=0)
        base_pt = np.mean(pts[aligment_data["base_region"]], axis=0)

        self.add_virtual_node(LV_VIRTUAL_NODES.APEX, apex_pt, True)
        self.add_virtual_node(LV_VIRTUAL_NODES.BASE, base_pt, True)
        self.set_normal(unit_vector(base_pt - apex_pt))

        return surf_regions.astype(np.int64), global_regions.astype(np.int64)

    def identify_epi_endo_surfaces(self, threshold: float = 90.0, ref_point: np.ndarray = None) -> tuple:
        """
            Estimates Epicardium and Endocardium surfaces based on the angle between \
                vectors from the reference point to each node and their respective\
                surface normals. If ref_point is not provided, it will use the center \
                of the geometry.
        Args:
            threshold (float, optional): Angles less than threshold will be considered \
                part of 'Endocardium' while others will be considered part of 'Epicardium'. \
                Defaults to 90.0.
            ref_point (np.ndarray or None, optional): Reference point; must be a (1x3) \
                np.ndarray specifying [x,y,z] coordinates. If set to None, the function \
                will use the estimated center of the geometry.

        Returns:
            tuple: Arrays containing values for each node as 'Endo', 'Epi' or Neither \ 
                    at surface mesh and global mesh.
        """
        # extract surface mesh (use extract)
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points
        # get surface Normals and get respective points-data
        lvsurf.compute_normals(inplace=True)
        surf_normals = lvsurf.get_array("Normals", "points")
        # if ref_point was not specified, est geometry center
        if ref_point is None:
            ref_point = self.est_centroid()
        else:
            if not isinstance(ref_point, np.ndarray):
                ref_point = np.array(ref_point)
        # get vector from pts at surface to center
        pts_to_ref = ref_point - pts
        # compute angle difference between surface normals and pts to ref
        angles = angle_between(pts_to_ref, surf_normals,
                               check_orientation=False)  # returns [0 to pi]
        # set initial endo-epi surface guess
        endo_epi_guess = np.zeros(len(pts))
        initial_thresh = np.radians(threshold)
        endo_ids = np.where(angles < initial_thresh)[0]
        epi_ids = np.where(angles >= initial_thresh)[0]
        endo_epi_guess[endo_ids] = LV_SURFS.ENDO
        endo_epi_guess[epi_ids] = LV_SURFS.EPI

        # -- save data at surface and global mesh

        # save data at mesh surface
        self._surface_mesh.point_data[LV_MESH_DATA.EPI_ENDO_GUESS.value] = endo_epi_guess.astype(
            np.int64)
        # convert local surf ids to global ids
        epi_ids_mesh = self.map_surf_ids_to_global_ids(epi_ids, dtype=np.int64)
        endo_ids_mesh = self.map_surf_ids_to_global_ids(
            endo_ids, dtype=np.int64)
        # set epi/endo ids at mesh (global ids)
        mesh_endo_epi_guess = np.zeros(self.mesh.n_points)
        mesh_endo_epi_guess[epi_ids_mesh] = LV_SURFS.EPI
        mesh_endo_epi_guess[endo_ids_mesh] = LV_SURFS.ENDO
        # save data at global mesh
        self.mesh.point_data[LV_MESH_DATA.EPI_ENDO_GUESS.value] = mesh_endo_epi_guess.astype(
            np.int64)

        return endo_epi_guess.astype(np.int64), mesh_endo_epi_guess.astype(np.int64)

    # ----------------------------------------------------------------
    # Aortic and Mitral identification/set functions

    def identify_mitral_and_aortic_surfaces(self,
                                            a1=0.4,
                                            a2=0.5,
                                            a3=0.4,
                                            a4=80,
                                            a5=120,

                                            m1=0.15,
                                            m2=0.05,
                                            m3=0.0666,
                                            m4=0.333
                                            ):

        # -------------------------------
        # get surface mesh
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points
        surf_normals = lvsurf.get_array("Normals", "points")

        # -------------------------------
        # compute gradients
        lvsurf = lvsurf.compute_derivative(LV_MESH_DATA.EPI_ENDO_GUESS.value)
        # select gradients of interest (threshold based on magnitude)
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        # select points of interest (where grads_mag is positive)
        ioi = np.where(grads_mag > 0)[0]  # indexes of interest
        poi = pts[ioi]                    # pts of interest
        # compute centroids at mitral and aortic valves
        kmeans = KMeans(n_clusters=2, random_state=0).fit(poi)
        klabels = kmeans.labels_
        kcenters = kmeans.cluster_centers_
        # determine labels based on centroid closest to center
        kdist = np.linalg.norm(self.est_centroid() - kcenters, axis=1)
        label = np.zeros(len(klabels))

        if kdist[0] > kdist[1]:
            label[klabels == 1] = LV_SURFS.MITRAL
            label[klabels == 0] = LV_SURFS.AORTIC
            wc = [m4, 1.0-m4]
        else:
            label[klabels == 0] = LV_SURFS.MITRAL
            label[klabels == 1] = LV_SURFS.AORTIC
            wc = [1.0-m4, m4]
        # define clusters
        clustered = np.zeros(len(pts))
        clustered[ioi] = label

        # -------------------------------
        # Estimate aortic region
        # select aortic points
        atr_mask = np.where(clustered == LV_SURFS.AORTIC)[0]
        atr_pts = pts[atr_mask]
        # compute centers and radius
        c_atr = centroid(atr_pts)  # np.mean(atr_pts, axis=0)
        r_atr = np.mean(np.linalg.norm(atr_pts - c_atr, axis=1))
        # compute distance from pts to aortic centers
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        # filter by radius
        atr = np.where(d_atr <= r_atr * (a1+1.0))[0]

        # -------------------------------
        # Estimate mitral region
        # select mitral points
        mtr_mask = np.where(clustered == LV_SURFS.MITRAL)[0]
        mtr_pts = pts[mtr_mask]
        # compute center -> roughly 2/3 of aortic kcenter and mitral kcenter
        c_mtr = np.average(kcenters, axis=0, weights=wc)
        # adjust center -> due to concentration of nodes at 'right' side
        #                  (from mitral to aortic), center was always
        #                  skewed to the right. We kind of adjust by
        #                  adding moving m3*r_mtr dist to the left.
        r_mtr = np.mean(np.linalg.norm(mtr_pts - c_mtr, axis=1))
        c_mtr += r_mtr*m3 * \
            np.cross(self.get_normal(), unit_vector(c_atr-c_mtr))
        # recompute radius
        r_mtr = np.mean(np.linalg.norm(mtr_pts - c_mtr, axis=1))
        # compute distance from pts to mitral centers
        d_mtr = np.linalg.norm(pts - c_mtr, axis=1)
        # filter by radius
        mtr = np.where(d_mtr <= r_mtr * (m1+1.0))[0]

        # -------------------------------
        # define mitral border
        # filter by radius
        mtr_border = np.where(d_mtr <= r_mtr * (m2+1.0))[0]
        mtr_border_pts = pts[mtr_border]
        c_mtr_border = np.copy(c_mtr)
        r_mtr_border = np.mean(np.linalg.norm(
            mtr_border_pts - c_mtr_border, axis=1))

        # -------------------------------
        # compute intersection between mitral and aortic values
        its = np.intersect1d(atr, mtr)  # intersection

        # -------------------------------
        # define endo and epi mitral
        mtr_vecs_1 = centroid(pts) - pts[mtr]
        mtr_angles = angle_between(
            surf_normals[mtr], mtr_vecs_1, check_orientation=False)
        # select endo and epi mitral ids based on angle thresholds
        endo_mitral = np.setdiff1d(
            mtr[np.where((mtr_angles < np.radians(60)))[0]], its)
        epi_mitral = np.setdiff1d(
            mtr[np.where((mtr_angles > np.radians(60)))[0]], its)

        # -------------------------------
        # Refine aortic region
        # select aortic pts, including those of intersection
        atr_its = np.union1d(atr, its)
        atr_pts = pts[atr_its]
        c_atr = np.mean(atr_pts, axis=0)  # centroid(atr_pts)
        # compute distance from pts to aortic and mitral centers
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        # filter by radius
        atr = np.where(d_atr <= r_atr * (1.0+a2))[0]
        its = np.intersect1d(atr, mtr)  # intersection

        # -------------------------------
        # define endo_aortic and epi_aortic
        # compute angles between pts at aortic and its center
        atr_vecs_1 = c_atr - pts[atr]
        atr_angles = angle_between(
            surf_normals[atr], atr_vecs_1, check_orientation=False)
        # select endo and epi aortic ids based on angle thresholds
        endo_aortic = atr[np.where((atr_angles < np.radians(a4)))[0]]
        epi_aortic = atr[np.where((atr_angles > np.radians(a5)))[0]]

        # -------------------------------
        # define ids at aortic border
        # set endo and epi aortic ids as 'mask' values at lv surface mesh
        #   -> This step is performed so that we can use 'compute_derivative'
        endo_aortic_mask = np.zeros(len(pts))
        endo_aortic_mask[endo_aortic] = 1.0
        lvsurf.point_data["TMP_ENDO_AORTIC_MASK"] = endo_aortic_mask
        epi_aortic_mask = np.zeros(len(pts))
        epi_aortic_mask[epi_aortic] = 1.0
        lvsurf.point_data["TMP_EPI_AORTIC_MASK"] = epi_aortic_mask
        # select ids at the border of endo aortic mask using gradient method
        lvsurf = lvsurf.compute_derivative("TMP_ENDO_AORTIC_MASK")
        lvsurf = lvsurf.compute_derivative("gradient")
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        ioi_atr_endo = np.where(grads_mag > 0)[0]
        # select ids at the border of epi aortic mask using gradient method
        lvsurf = lvsurf.compute_derivative("TMP_EPI_AORTIC_MASK")
        lvsurf = lvsurf.compute_derivative("gradient")
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        ioi_atr_epi = np.where(grads_mag > 0)[0]
        # select ids at aortic border by intersecting ids fround at endo and
        # epi aortic masks. Note: these masks are too far apart, no border
        # will be detected
        atr_border = np.intersect1d(ioi_atr_endo, ioi_atr_epi)

        # -------------------------------
        # refine aortic border ids
        # set current aortic border ids as 'mask' values at lv surface mesh
        border_aortic_mask = np.zeros(len(pts))
        border_aortic_mask[atr_border] = 1.0
        lvsurf.point_data["TMP_BORDER_AORTIC_MASK"] = border_aortic_mask
        # expand ids at aortic border with gradient method
        lvsurf = lvsurf.compute_derivative("TMP_BORDER_AORTIC_MASK")
        lvsurf = lvsurf.compute_derivative("gradient")
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        atr_border = np.where(grads_mag > 0)[0]

        # -------------------------------
        # refine aortic at endocardio
        # select current pts at aortic border anc compute its center
        atr_border_pts = pts[atr_border]
        c_atr_border = centroid(atr_border_pts)
        r_atr_border = np.mean(np.linalg.norm(
            atr_border_pts - c_atr_border, axis=1))
        # c_atr_border = np.mean(atr_border_pts, axis=0)
        # select current pts at aortic border
        endo_aortic_pts = pts[endo_aortic]
        # compute distances between center of aortic border and endo aortic pts
        d_atr = np.linalg.norm(endo_aortic_pts - c_atr_border, axis=1)
        # filter by radius
        endo_aortic = endo_aortic[np.where(d_atr <= r_atr * (a3+1.0))[0]]

        # -------------------------------
        # update atr ids
        atr = reduce(np.union1d, (endo_aortic, epi_aortic, atr_border))

        # -------------------------------
        # adjust endo-epi am intersection
        its_vecs_1 = c_atr - pts[its]
        its_angles = angle_between(
            surf_normals[its], its_vecs_1, check_orientation=False)
        # select endo and epi aortic ids based on angle thresholds
        endo_am_its = its[np.where((its_angles <= np.radians(60)))[0]]
        epi_am_its = np.setdiff1d(
            its[np.where((its_angles > np.radians(60)))[0]], endo_am_its)

        its_vecs_2 = centroid(pts) - pts[epi_am_its]
        its_angles_2 = angle_between(
            surf_normals[epi_am_its], its_vecs_2, check_orientation=False)
        endo_am_its = np.union1d(
            endo_am_its, epi_am_its[np.where((its_angles_2 < np.radians(80)))[0]])
        epi_am_its = np.setdiff1d(epi_am_its, endo_am_its)

        # -------------------------------
        # set mask by layering values
        clustered = np.zeros(len(pts))
        clustered[epi_aortic] = LV_SURFS.EPI_AORTIC
        clustered[endo_aortic] = LV_SURFS.ENDO_AORTIC
        clustered[atr_border] = LV_SURFS.BORDER_AORTIC
        clustered[epi_mitral] = LV_SURFS.EPI_MITRAL
        clustered[endo_mitral] = LV_SURFS.ENDO_MITRAL
        clustered[mtr_border] = LV_SURFS.BORDER_MITRAL
        clustered[epi_am_its] = LV_SURFS.EPI_AM_INTERCECTION
        clustered[endo_am_its] = LV_SURFS.ENDO_AM_INTERCECTION

        surf_to_global = np.array(
            lvsurf.point_data["vtkOriginalPointIds"], dtype=np.int64)

        # -------------------------------
        # transform ids from local surf values to global mesh ids
        mesh_clustered = np.zeros(self.mesh.n_points)
        mesh_clustered[surf_to_global] = clustered

        # -------------------------------
        # set atr and mitral as a mesh data
        am_highlighted = np.zeros(len(pts))
        am_highlighted[atr] = LV_SURFS.AORTIC
        am_highlighted[mtr] = LV_SURFS.MITRAL
        am_highlighted[its] = LV_SURFS.AM_INTERCECTION
        mesh_am_highlighted = np.zeros(self.mesh.n_points)
        mesh_am_highlighted[surf_to_global] = am_highlighted

        # -------------------------------
        # set endo-epi over entire AM area
        atr_border_vecs_1 = c_atr_border - pts[atr_border]
        atr_border_angles = angle_between(
            surf_normals[atr_border], atr_border_vecs_1, check_orientation=False)
        # select endo and epi mitral ids based on angle thresholds
        endo_atr_border = atr_border[np.where(
            (atr_border_angles < np.radians(70)))[0]]
        epi_atr_border = atr_border[np.where(
            (atr_border_angles > np.radians(70)))[0]]
        endo_am_region = reduce(
            np.union1d, (endo_aortic, endo_mitral, endo_am_its, endo_atr_border))
        epi_am_region = reduce(
            np.union1d, (epi_aortic, epi_mitral, epi_am_its, epi_atr_border))

        am_endo_epi_regions = np.zeros(len(pts))
        am_endo_epi_regions[endo_am_region] = LV_SURFS.ENDO_AM_REGION
        am_endo_epi_regions[epi_am_region] = LV_SURFS.EPI_AM_REGION
        mesh_am_endo_epi_regions = np.zeros(self.mesh.n_points)
        mesh_am_endo_epi_regions[surf_to_global] = am_endo_epi_regions

        # -------------------------------
        # save mesh data
        self._surface_mesh.point_data[LV_MESH_DATA.AM_DETAILED.value] = clustered.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.AM_DETAILED.value] = mesh_clustered.astype(
            np.int64)

        self._surface_mesh.point_data[LV_MESH_DATA.AM_SURFS.value] = am_highlighted.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.AM_SURFS.value] = mesh_am_highlighted.astype(
            np.int64)

        self._surface_mesh.point_data[LV_MESH_DATA.AM_EPI_ENDO.value] = am_endo_epi_regions.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.AM_EPI_ENDO.value] = mesh_am_endo_epi_regions.astype(
            np.int64)

        # -------------------------------
        # save virtual reference nodes
        self.add_virtual_node(LV_VIRTUAL_NODES.MITRAL, c_mtr, True)
        self.add_virtual_node(LV_VIRTUAL_NODES.AORTIC, c_atr, True)
        self.add_virtual_node(
            LV_VIRTUAL_NODES.AORTIC_BORDER, c_atr_border, True)

        # -------------------------------
        # save aortic and mitral info
        self.aortic_info = {
            LV_AM_INFO.RADIUS.value: r_atr,
            LV_AM_INFO.CENTER.value: c_atr,
            LV_AM_INFO.SURF_IDS.value: atr,  # ids at surface
            LV_AM_INFO.MESH_IDS.value: surf_to_global[atr],  # ids at mesh

            LV_AM_INFO.BORDER_RADIUS.value: r_atr_border,
            LV_AM_INFO.BORDER_CENTER.value: c_atr_border,
            LV_AM_INFO.BORDER_SURF_IDS.value: atr_border,
            LV_AM_INFO.BORDER_MESH_IDS.value: surf_to_global[atr_border]
        }
        self.mitral_info = {
            LV_AM_INFO.RADIUS.value: np.mean(np.linalg.norm(pts[mtr] - c_mtr, axis=1)),
            LV_AM_INFO.CENTER.value: np.array(c_mtr, dtype=np.float64),
            LV_AM_INFO.SURF_IDS.value: mtr,
            LV_AM_INFO.MESH_IDS.value: surf_to_global[mtr],

            LV_AM_INFO.BORDER_RADIUS.value: r_mtr_border,
            LV_AM_INFO.BORDER_CENTER.value: c_mtr_border,
            LV_AM_INFO.BORDER_SURF_IDS.value: mtr_border,  # ids at surface
            # ids at mesh
            LV_AM_INFO.BORDER_MESH_IDS.value: surf_to_global[mtr_border]
        }

        return clustered.astype(np.int64), mesh_clustered.astype(np.int64)

    def set_aortic_info(self,
                        aortic_id=LV_SURFS.AORTIC,
                        border_aortic_id=LV_SURFS.BORDER_AORTIC,
                        add_virtual_nodes=True,
                        ) -> None:
        """Sets aortic information for other computations, such as boundary conditions.\
           Creates 'aortic_info' dictionary with radius, center, and mesh ids.\
           If border information is provided, it adds information for the border as well.
           If 'add_virtual_nodes' is set to true, it will create virtual node information\
               based on respective centers.

        Args:
            aortic_id (int, str or Enum, optional): identification of aortic nodeset. \
                Defaults to LV_SURFS.AORTIC.
            border_aortic_id (int, str or Enum, optional): identification of border aortic nodeset. \
                Defaults to LV_SURFS.BORDER_AORTIC.
            add_virtual_nodes (bool, optional): Whether to create virtual node information.
        """
        if aortic_id is not None:
            try:
                ioi = self.get_nodeset(aortic_id)
            except KeyError:
                raise KeyError("aortic_id not found within nodesets. Did you create it? \
                    You can use the function 'identify_surfaces' to automatically create it.")
            pts = self.nodes(ioi)
            center = centroid(pts)
            r = radius(pts, center)
            self.aortic_info = {
                LV_AM_INFO.RADIUS.value: r,
                LV_AM_INFO.CENTER.value: center,
                LV_AM_INFO.SURF_IDS.value: None,  # ids at surface
                LV_AM_INFO.MESH_IDS.value: ioi,  # ids at mesh
            }
            if add_virtual_nodes:
                self.add_virtual_node(LV_VIRTUAL_NODES.AORTIC, center, True)

        if border_aortic_id is not None:
            try:
                ioi_b = self.get_nodeset(border_aortic_id)
            except KeyError:
                raise KeyError("border_aortic_id not found within nodesets. Did you create it? \
                    You can use the function 'identify_surfaces' to automatically create it.")
            pts_b = self.nodes(ioi_b)
            center_b = centroid(pts_b)
            r_b = radius(pts, center_b)
            self.aortic_info.update({
                LV_AM_INFO.BORDER_RADIUS.value: r_b,
                LV_AM_INFO.BORDER_CENTER.value: center_b,
                LV_AM_INFO.BORDER_SURF_IDS.value: None,
                LV_AM_INFO.BORDER_MESH_IDS.value: ioi_b
            })
            if add_virtual_nodes:
                self.add_virtual_node(
                    LV_VIRTUAL_NODES.AORTIC_BORDER, center_b, True)

    def set_mitral_info(self,
                        mitral_id=LV_SURFS.MITRAL,
                        border_mitral_id=LV_SURFS.BORDER_MITRAL,
                        add_virtual_nodes=True,
                        ) -> None:
        """Sets mitral information for other computations, such as boundary conditions.\
           Creates 'mitral_info' dictionary with radius, center, and mesh ids.\
           if border information is provided, it adds information for the border as well.\
            If 'add_virtual_nodes' is set to true, it will create virtual node information\
            based on respective centers.

        Args:
            mitral_id (int, str or Enum, optional): identification of mitral nodeset. \
                Defaults to LV_SURFS.MITRAL.
            border_mitral_id (int, str or Enum, optional): identification of border mitral nodeset. \
                Defaults to LV_SURFS.BORDER_MITRAL.
            add_virtual_nodes (bool, optional): Whether to create virtual node information.
        """
        if mitral_id is not None:
            try:
                ioi = self.get_nodeset(mitral_id)
            except KeyError:
                raise KeyError("mitral_id not found within nodesets. Did you create it? \
                    You can use the function 'identify_surfaces' to automatically create it.")
            pts = self.nodes(ioi)
            center = centroid(pts)
            r = radius(pts, center)
            self.mitral_info = {
                LV_AM_INFO.RADIUS.value: r,
                LV_AM_INFO.CENTER.value: center,
                LV_AM_INFO.SURF_IDS.value: None,  # ids at surface
                LV_AM_INFO.MESH_IDS.value: ioi,  # ids at mesh
            }
            if add_virtual_nodes:
                self.add_virtual_node(LV_VIRTUAL_NODES.MITRAL, center, True)

        if border_mitral_id is not None:
            try:
                ioi_b = self.get_nodeset(border_mitral_id)
            except KeyError:
                raise KeyError("border_mitral_id not found within nodesets. Did you create it? \
                    You can use the function 'identify_surfaces' to automatically create it.")
            pts_b = self.nodes(ioi_b)
            center_b = centroid(pts_b)
            r_b = radius(pts, center_b)
            self.mitral_info.update({
                LV_AM_INFO.BORDER_RADIUS.value: r_b,
                LV_AM_INFO.BORDER_CENTER.value: center_b,
                LV_AM_INFO.BORDER_SURF_IDS.value: None,
                LV_AM_INFO.BORDER_MESH_IDS.value: ioi_b
            })
            if add_virtual_nodes:
                self.add_virtual_node(
                    LV_VIRTUAL_NODES.MITRAL_BORDER, center, True)

    def identify_surfaces(self,
                          endo_epi_args={},
                          apex_base_args={},
                          aortic_mitral_args={},
                          create_nodesets=True,

                          ):

        endo_epi, mesh_endo_epi = self.identify_epi_endo_surfaces(
            **endo_epi_args)
        apex_base, mesh_apex_base = self.identify_base_and_apex_surfaces(
            **apex_base_args)
        aortic_mitral, mesh_aortic_mitral = self.identify_mitral_and_aortic_surfaces(
            **aortic_mitral_args)

        # ------------------------------
        # update endo-epi based on detailed info from aortic_mitral surfs
        # now that we have all surface IDs, we can adjust endo_epi based on detailed
        # information from aortic and mitral data.
        surf_to_global = self._surface_mesh.point_data["vtkOriginalPointIds"]

        # set copy of values (do not modify them directly)
        updated_endo_epi = np.copy(endo_epi)
        mesh_updated_endo_epi = np.copy(mesh_endo_epi)
        # get data
        am_epi_endo = self.get_surface_mesh(
        ).point_data[LV_MESH_DATA.AM_EPI_ENDO.value]
        mesh_am_epi_endo = self.mesh.point_data[LV_MESH_DATA.AM_EPI_ENDO.value]
        # overwrite values at specified locations
        updated_endo_epi[np.where(am_epi_endo == LV_SURFS.ENDO_AM_REGION.value)[
            0]] = LV_SURFS.ENDO
        updated_endo_epi[np.where(am_epi_endo == LV_SURFS.EPI_AM_REGION.value)[
            0]] = LV_SURFS.EPI
        # mesh_am_epi_endo[surf_to_global] = updated_endo_epi
        mesh_updated_endo_epi[np.where(
            mesh_am_epi_endo == LV_SURFS.ENDO_AM_REGION.value)[0]] = LV_SURFS.ENDO
        mesh_updated_endo_epi[np.where(mesh_am_epi_endo == LV_SURFS.EPI_AM_REGION.value)[
            0]] = LV_SURFS.EPI
        # save new data
        self._surface_mesh[LV_MESH_DATA.EPI_ENDO.value] = updated_endo_epi
        self.mesh[LV_MESH_DATA.EPI_ENDO.value] = mesh_updated_endo_epi

        # ------------------------------
        # update APEX_BASE_REGIONS based on endo-epi
        updated_apex_base = np.copy(apex_base)
        mesh_updated_apex_base = np.copy(mesh_apex_base)
        # overwrite values at specified locations
        ioi = np.where((updated_apex_base == LV_SURFS.BASE_REGION.value) & (
            updated_endo_epi == LV_SURFS.ENDO))[0]
        updated_apex_base[ioi] = LV_SURFS.ENDO_BASE_REGION
        ioi = np.where((updated_apex_base == LV_SURFS.BASE_REGION.value) & (
            updated_endo_epi == LV_SURFS.EPI))[0]
        updated_apex_base[ioi] = LV_SURFS.EPI_BASE_REGION
        ioi = np.where((updated_apex_base == LV_SURFS.APEX_REGION.value) & (
            updated_endo_epi == LV_SURFS.ENDO))[0]
        updated_apex_base[ioi] = LV_SURFS.ENDO_APEX_REGION
        ioi = np.where((updated_apex_base == LV_SURFS.APEX_REGION.value) & (
            updated_endo_epi == LV_SURFS.EPI))[0]
        updated_apex_base[ioi] = LV_SURFS.EPI_APEX_REGION
        # overwrite values at specified locations at mesh
        mesh_updated_apex_base[surf_to_global] = updated_apex_base

        # save new data
        self._surface_mesh[LV_MESH_DATA.AB_ENDO_EPI.value] = updated_apex_base.astype(
            np.int64)
        self.mesh[LV_MESH_DATA.AB_ENDO_EPI.value] = mesh_updated_apex_base.astype(
            np.int64)

        # ------------------------------
        # To 'merge' result, we will overlay each info layer on top of each other
        # endo_epi will serve as backgroun  (will be lowest layer)
        # aortic_mitral is the last merge (will be top-most layer and overwrite apex_base)

        layers = np.copy(updated_endo_epi)
        mesh_layers = np.copy(mesh_updated_endo_epi)
        # match indexes of interest at surface level
        ioi = np.where(aortic_mitral != LV_SURFS.OTHER.value)[0]
        layers[ioi] = aortic_mitral[ioi]
        # match indexes of interest at mesh (global) level
        mesh_layers[surf_to_global] = layers
        # ioi = np.where(mesh_aortic_mitral!=LV_SURFS.OTHER.value)[0]
        # mesh_layers[ioi] = mesh_aortic_mitral[ioi]
        # save results at surface and mesh levels
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS_DETAILED.value] = layers.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.SURFS_DETAILED.value] = mesh_layers.astype(
            np.int64)

        # ------------------------------
        # Lets create another merge -> EPI_ENDO + AM_SURFS
        # endo_epi will serve as backgroun  (will be lowest layer)
        # aortic_mitral is the last merge (will be top-most layer and overwrite apex_base)
        layers = np.copy(updated_endo_epi)
        mesh_layers = np.copy(mesh_updated_endo_epi)
        # match indexes of interest at surface level
        am_values = self._surface_mesh.get_array(
            LV_MESH_DATA.AM_SURFS.value, "points")
        ioi = np.where(am_values != LV_SURFS.OTHER.value)[0]
        layers[ioi] = am_values[ioi]
        # match indexes of interest at mesh (global) level
        mesh_layers[surf_to_global] = layers
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS.value] = layers.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.SURFS.value] = mesh_layers.astype(
            np.int64)

        # create nodesets
        if create_nodesets:
            self.create_nodesets_from_surfaces(
                mesh_data=LV_MESH_DATA.APEX_BASE_REGIONS.value, overwrite=False)
            self.create_nodesets_from_surfaces(
                mesh_data=LV_MESH_DATA.EPI_ENDO.value, overwrite=False)
            self.create_nodesets_from_surfaces(
                mesh_data=LV_MESH_DATA.SURFS.value, overwrite=False)
            self.create_nodesets_from_surfaces(
                mesh_data=LV_MESH_DATA.AM_SURFS.value, overwrite=False)
            self.create_nodesets_from_surfaces(
                mesh_data=LV_MESH_DATA.SURFS_DETAILED.value, overwrite=False)

        # set flag to indicate surfaces were identified from this method:
        self._surfaces_identified_with_class_method = True

    def create_nodesets_from_surfaces(self,
                                      mesh_data=LV_MESH_DATA.SURFS.value,
                                      skip={},
                                      overwrite=False
                                      ):
        mesh_data = self.check_enum(mesh_data)

        ids = self.mesh.point_data[mesh_data]
        for surf_enum in LV_SURFS:
            if surf_enum.value in self._nodesets and overwrite == False:
                continue
            if surf_enum.name != "OTHER" and surf_enum.name not in skip:
                found_ids = np.copy(np.where(ids == surf_enum.value)[0])
                if len(found_ids) > 0:
                    self.add_nodeset(surf_enum, found_ids, overwrite)

    def peek_unique_values_in_surface(self, surface_name: str, enum_like: Enum = None) -> list:
        """Returns a list of unique values in the given surface. If Enum is specified, \
            list will contain strings representing each surface region.

        Args:
            surface_name (str): Mesh data name containing surface region data.
            enum_like (Enum, optional): Enum representation of values in the surface. Defaults to None.

        Returns:
            list: Unique values in the given surface.
        """

        surface_name = self.check_enum(surface_name)

        surfmap = self.mesh.point_data[surface_name]
        unique_vals = np.unique(surfmap)
        if enum_like is not None:
            return [enum_like(item).name for item in unique_vals]
        else:
            return unique_vals

    # =============================================================================
    # Check methods

    def check_surf_initialization(self):
        return self._surfaces_identified_with_class_method

    # =============================================================================
    # Fiber methods

    def identify_fibers_region_ids_ldrb_1(self) -> tuple:
        """Identifies mesh surface regions for fiber computation based on nodesets. \
            LDRB only accpets three regions: 'endo', 'epi' and 'border(s)', therefore,\
            we must combine aortic and mitral regions of interest into a single region.\
            This method we combine: BORDER_MITRAL and BORDER_AORTIC

        Raises:
            RuntimeError: Length of one of required nodesets is 0.

        Returns:
            tuple: (surface regions ids, mesh regions ids)
        """
        # get nodesets
        endo = self.get_nodeset(LV_SURFS.ENDO)
        epi = self.get_nodeset(LV_SURFS.EPI)
        mitral = self.get_nodeset(LV_SURFS.BORDER_MITRAL)
        border_aortic = self.get_nodeset(LV_SURFS.BORDER_AORTIC)
        # check possible errors
        if len(endo) == 0:
            raise RuntimeError(
                "No ids found for 'endo'. Please, ensure that 'endo' nodeset is not empty.")
        if len(epi) == 0:
            raise RuntimeError(
                "No ids found for 'epi'. Please, ensure that 'epi' nodeset is not empty.")
        if len(mitral) == 0:
            raise RuntimeError(
                "No ids found for 'mitral'. Please, ensure that 'EPI_MITRAL' and 'BORDER_MITRAL' nodeset are not empty.")
        if len(border_aortic) == 0:
            raise RuntimeError(
                "No ids found for 'border_aortic'. Please, ensure that 'border_aortic' nodeset is not empty.")
        # combine ids into a single surfac emap
        ldrb_mesh_region_ids = np.zeros(self.mesh.n_points)
        ldrb_mesh_region_ids[endo] = LV_SURFS.ENDO
        ldrb_mesh_region_ids[epi] = LV_SURFS.EPI
        ldrb_mesh_region_ids[mitral] = LV_SURFS.BASE_BORDER
        ldrb_mesh_region_ids[border_aortic] = LV_SURFS.BASE_BORDER
        # get ids for surface
        lvsurf_map_id = self.get_surface_id_map_from_mesh()
        lvsurf = self.get_surface_mesh()
        # save data at surface and mesh level
        lvsurf[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids[lvsurf_map_id]
        self.mesh[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids

        return ldrb_mesh_region_ids, ldrb_mesh_region_ids

    def identify_fibers_region_ids_ldrb_2(self) -> tuple:
        """Identifies mesh surface regions for fiber computation based on nodesets. \
            LDRB only accpets three regions: 'endo', 'epi' and 'border(s)', therefore,\
            we must combine aortic and mitral regions of interest into a single region.\
            This method we combine: EPI_MITRAL, BORDER_MITRAL and BORDER_AORTIC

        Raises:
            RuntimeError: Length of one of required nodesets is 0.

        Returns:
            tuple: (surface regions ids, mesh regions ids)
        """
        # get nodesets
        endo = self.get_nodeset(LV_SURFS.ENDO)
        epi = self.get_nodeset(LV_SURFS.EPI)
        mitral = np.union1d(self.get_nodeset(
            LV_SURFS.EPI_MITRAL), self.get_nodeset(LV_SURFS.BORDER_MITRAL))
        border_aortic = self.get_nodeset(LV_SURFS.BORDER_AORTIC)
        # check possible errors
        if len(endo) == 0:
            raise RuntimeError(
                "No ids found for 'endo'. Please, ensure that 'endo' nodeset is not empty.")
        if len(epi) == 0:
            raise RuntimeError(
                "No ids found for 'epi'. Please, ensure that 'epi' nodeset is not empty.")
        if len(mitral) == 0:
            raise RuntimeError(
                "No ids found for 'mitral'. Please, ensure that 'EPI_MITRAL' and 'BORDER_MITRAL' nodeset are not empty.")
        if len(border_aortic) == 0:
            raise RuntimeError(
                "No ids found for 'border_aortic'. Please, ensure that 'border_aortic' nodeset is not empty.")
        # combine ids into a single surfac emap
        ldrb_mesh_region_ids = np.zeros(self.mesh.n_points)
        ldrb_mesh_region_ids[endo] = LV_SURFS.ENDO
        ldrb_mesh_region_ids[epi] = LV_SURFS.EPI
        ldrb_mesh_region_ids[mitral] = LV_SURFS.BASE_BORDER
        ldrb_mesh_region_ids[border_aortic] = LV_SURFS.BASE_BORDER
        # get ids for surface
        lvsurf_map_id = self.get_surface_id_map_from_mesh()
        lvsurf = self.get_surface_mesh()
        # save data at surface and mesh level
        lvsurf[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids[lvsurf_map_id]
        self.mesh[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids

        return ldrb_mesh_region_ids, ldrb_mesh_region_ids

    def identify_fibers_region_ids_ldrb_3(self) -> tuple:
        """Identifies mesh surface regions for fiber computation based on nodesets. \
            LDRB only accpets three regions: 'endo', 'epi' and 'border(s)', therefore,\
            we must combine aortic and mitral regions of interest into a single region.\
            This method we combine: ENDO_MITRAL, EPI_MITRAL, BORDER_MITRAL and BORDER_AORTIC

        Raises:
            RuntimeError: Length of one of required nodesets is 0.

        Returns:
            tuple: (surface regions ids, mesh regions ids)
        """
        # get nodesets
        endo = self.get_nodeset(LV_SURFS.ENDO)
        epi = self.get_nodeset(LV_SURFS.EPI)
        mitral = np.union1d(
            self.get_nodeset(LV_SURFS.ENDO_MITRAL),
            self.get_nodeset(LV_SURFS.EPI_MITRAL),
            self.get_nodeset(LV_SURFS.BORDER_MITRAL)
        )
        border_aortic = self.get_nodeset(LV_SURFS.BORDER_AORTIC)
        # check possible errors
        if len(endo) == 0:
            raise RuntimeError(
                "No ids found for 'endo'. Please, ensure that 'endo' nodeset is not empty.")
        if len(epi) == 0:
            raise RuntimeError(
                "No ids found for 'epi'. Please, ensure that 'epi' nodeset is not empty.")
        if len(mitral) == 0:
            raise RuntimeError(
                "No ids found for 'mitral'. Please, ensure that 'EPI_MITRAL' and 'BORDER_MITRAL' nodeset are not empty.")
        if len(border_aortic) == 0:
            raise RuntimeError(
                "No ids found for 'border_aortic'. Please, ensure that 'border_aortic' nodeset is not empty.")
        # combine ids into a single surfac emap
        ldrb_mesh_region_ids = np.zeros(self.mesh.n_points)
        ldrb_mesh_region_ids[endo] = LV_SURFS.ENDO
        ldrb_mesh_region_ids[epi] = LV_SURFS.EPI
        ldrb_mesh_region_ids[mitral] = LV_SURFS.BASE_BORDER
        ldrb_mesh_region_ids[border_aortic] = LV_SURFS.BASE_BORDER
        # get ids for surface
        lvsurf_map_id = self.get_surface_id_map_from_mesh()
        lvsurf = self.get_surface_mesh()
        # save data at surface and mesh level
        lvsurf[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids[lvsurf_map_id]
        self.mesh[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids

        return ldrb_mesh_region_ids, ldrb_mesh_region_ids

    def identify_fibers_region_ids_ldrb(self, mode: str) -> tuple:
        """
            Identifies mesh surface regions for fiber computation based on nodesets. \
            LDRB only accpets three regions: 'endo', 'epi' and 'border(s)', therefore,\
            we must combine aortic and mitral regions of interest into a single region.\

            This method wraps all implemented modes to identify surfaces for LDRB.\
            The main difference between each method is the union1d of 'BASE_BORDER':

            LDRB_1: BORDER_MITRAL and BORDER_AORTIC

            LDRB_2: EPI_MITRAL, BORDER_MITRAL and BORDER_AORTIC

            LDRB_3: ENDO_MITRAL, EPI_MITRAL, BORDER_MITRAL and BORDER_AORTIC

        Args:
            mode (str or Enum): Mode of fiber regions to identify. Check LV Enum for details.

        Raises:
            NotImplementedError: If mode is not supported or not implemented, it will raise Error.

        Returns:
            tuple: _description_
        """
        mode = self.check_enum(mode)
        if mode == LV_FIBER_MODES.LDRB_1.value:
            return self.identify_fibers_region_ids_ldrb_1()
        elif mode == LV_FIBER_MODES.LDRB_2.value:
            return self.identify_fibers_region_ids_ldrb_2()
        elif mode == LV_FIBER_MODES.LDRB_3.value:
            return self.identify_fibers_region_ids_ldrb_3()
        else:
            raise NotImplementedError(
                "Mode '%s' not implemented or not supported." % mode)

    def add_fibers(self, name: str, fdata: np.ndarray):
        name = self.check_enum(name)
        if len(fdata) == self.mesh.n_points:
            self.mesh.point_data[name] = np.copy(fdata)
        elif len(fdata) == self.mesh.n_cells:
            self.mesh.cell_data[name] = np.copy(fdata)
        else:
            raise ValueError("Length of fiber data must be equal to the number of points\
                or cells in the mesh. Fibers are not saved surface mesh by default, if\
                you are trying to add fibers only at surface level, please do it manually.")

    def compute_fibers(self,
                       surfRegionsIds: str,
                       fiber_space="P_1",
                       alpha_endo_lv=60,  # Fiber angle on the endocardium
                       alpha_epi_lv=-60,  # Fiber angle on the epicardium
                       beta_endo_lv=0,  # Sheet angle on the endocardium
                       beta_epi_lv=0,  # Sheet angle on the epicardium
                       markers={},
                       ldrb_kwargs={},

                       save_xdmfs=False,
                       xdmfs_dir=None,
                       xdmfs_basename=None,

                       del_generated_files=True,
                       ):
        """DOC PENDING.

        This is a wrap method for the LDRB library: https://github.com/finsberg/ldrb.\
        Credits should be to the owners of the original LDRB library.

        Args:
            surfRegionsIds (str): _description_
            fiber_space (str, optional): _description_. Defaults to "P_1".
            alpha_endo_lv (int, optional): _description_. Defaults to 60.
            ldrb_kwargs (dict, optional): _description_. Defaults to {}.
            save_xdmfs (bool, optional): _description_. Defaults to False.
            xdmfs_dir (_type_, optional): _description_. Defaults to None.
            xdmfs_basename (_type_, optional): _description_. Defaults to None.
            del_generated_files (bool, optional): _description_. Defaults to True.

        Raises:
            ImportError: _description_
            ImportError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ImportError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        # ------------------
        # check for error conditions before inital fiber computation steps
        # try to import ldrb library
        try:
            import ldrb
        except ImportError:
            raise ImportError("ldrb library is required for fiber computation.\
                Please, see https://github.com/finsberg/ldrb for details.")

        # try to import meshion for mesh file manipulation (gmsh)
        try:
            import meshio
        except ImportError:
            raise ImportError("Meshio library is required for mesh file manipulation\
                during fiber computation.")
        # check if given surfRegionsIds is Enum
        surfRegionsIds = self.check_enum(surfRegionsIds)
        # check for mesh types (required pure tetrahedral mesh)
        assert(self.check_tet4_mesh()
               ), "Mesh must be composed of pure tetrahedrons."
        assert(self.check_tri3_surfmesh()
               ), "Surface mesh must be composed of pure triangles."
        # check for surface region ids (values for each node in surface describing region id)
        (in_mesh, in_surf_mesh) = self.check_mesh_data(surfRegionsIds)
        if not in_surf_mesh:
            found_ids = False
            try:
                self.identify_fibers_region_ids_ldrb(surfRegionsIds)
                found_ids = True
            except NotImplementedError:
                raise ValueError("Surface regions ids '{}' not found in mesh data. \
                    Tried to compute using 'identify_fibers_region_ids_ldrb', but method \
                    was not implemented. Please, check surfRegionsIds mesh data key.")
            if not found_ids:
                if self.check_surf_initialization():
                    raise ValueError("Surface regions ids '{}' is not initialized within internal algorithm. \
                        Did you add to surface mesh data?".format(surfRegionsIds))
                else:
                    raise ValueError("Surface regions ids '{}' not found in mesh data. \
                        Did you identify surfaces? See 'LV.identify_surfaces'.")
        # check markers
        if len(markers) == 0:  # empty
            markers = self._default_fiber_markers
        if (len(markers) != 3) or ("epi" not in markers) or ("lv" not in markers) or ("base" not in markers):
            raise ValueError("Markers must represent dictionary values for ids at surface. \
                It must contain 3 key/value pairs for 'epi', 'lv', and 'base'.\
                Please, see https://github.com/finsberg/ldrb for details.")
        # check xdmfs_dir if xdmfs was requested
        if save_xdmfs:
            try:
                import dolfin
            except ImportError:
                raise ImportError(
                    "Fenics dolfin library is required to save xdmfs.")
            # check for suitable directory
            if xdmfs_dir is not None:
                if not os.path.isdir(str(xdmfs_dir)):
                    os.makedirs(xdmfs_dir)
            else:
                if self._ref_dir is None:
                    raise ValueError("save_xdmfs was requested but could not find suitable directory\
                        to save files. Did you not initialized from a file? Please use xdmfs_dir.")
                else:
                    xdmfs_dir = self.xdmfs_dir
            # check for xdmfs basename
            if xdmfs_basename is not None:
                if not isinstance(xdmfs_basename, str):
                    raise ValueError("xdmfs_basename must be a string.")
            else:
                if self._ref_file is None:
                    raise ValueError("save_xdmfs was requested but could not find suitable basename\
                        to save files. Did you not initialized from a file? Please use xdmfs_basename.")
                else:
                    xdmfs_basename = os.path.basename(
                        self._ref_file).split('.')[0]

        # ------------------
        # prep for ldrb library
        # transform point region ids into cell ids at surface level
        cellregionIdsSurf = self.transform_point_data_to_cell_data(
            surfRegionsIds, surface=True)
        # combine volumetric mesh with surface mesh
        mesh = self.merge_mesh_and_surface_mesh()
        # adjust regions to include both surface and volume (with zeros)
        cellregionIds = np.hstack(
            (cellregionIdsSurf, np.zeros(mesh.n_cells - len(cellregionIdsSurf))))
        # add gmsh data
        mesh.clear_data()  # for some reason, no other info is accepted when loading in ldrb
        # adds "gmsh:physical" and "gmsh:geometrical"
        mesh = self.prep_for_gmsh(cellregionIds, mesh=mesh)
        # create temporary directory for saving current files
        import tempfile
        # with tempfile.TemporaryDirectory() as tmpdirname:
        # tmpdir = Path(tmpdirname)
        # save using meshio (I did not test other gmsh formats and binary files.)
        gmshfilepath = "gmshfile.msh"
        pv.save_meshio(gmshfilepath, mesh, file_format="gmsh22", binary=False)
        # create fenics mesh and face function
        mesh, ffun, _ = ldrb.gmsh2dolfin(gmshfilepath, unlink=False)
        # compute fibers
        fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
            mesh=mesh,
            fiber_space=fiber_space,
            ffun=ffun,
            markers=markers,
            alpha_endo_lv=alpha_endo_lv,  # Fiber angle on the endocardium
            alpha_epi_lv=alpha_epi_lv,  # Fiber angle on the epicardium
            beta_endo_lv=beta_endo_lv,  # Sheet angle on the endocardium
            beta_epi_lv=beta_epi_lv,  # Sheet angle on the epicardium
            **ldrb_kwargs
        )
        # remove created files
        if del_generated_files:
            for ldrb_support_file in [
                gmshfilepath,
                "mesh_gmshfile.h5",
                "mesh_gmshfile.xdmf",
                "triangle_mesh_gmshfile.h5",
                    "triangle_mesh_gmshfile.xdmf"]:
                if os.path.exists(ldrb_support_file):
                    os.remove(ldrb_support_file)
        # save each fiber component, if requested
        if save_xdmfs:
            xdmfs_basepath = xdmfs_dir/xdmfs_basename
            with dolfin.XDMFFile(mesh.mpi_comm(), xdmfs_basepath/"_fiber.xdmf") as xdmf:
                xdmf.write(fiber)
            with dolfin.XDMFFile(mesh.mpi_comm(), xdmfs_basepath/"_sheet.xdmf") as xdmf:
                xdmf.write(sheet)
            with dolfin.XDMFFile(mesh.mpi_comm(), xdmfs_basepath/"_sheet_normal.xdmf") as xdmf:
                xdmf.write(sheet_normal)
        # re-arrange fibers data to fit our mesh structure
        fiber_pts_vec = fiber.compute_vertex_values().reshape((3, -1)).T
        sheet_pts_vec = sheet.compute_vertex_values().reshape((3, -1)).T
        sheet_normal_pts_vec = sheet_normal.compute_vertex_values().reshape((3, -1)).T
        # when converting our mesh to fenics format using dolfin_ldrb, we lose our structure
        # and nodes/cells are completely re-arranged. To work around this, we need to map
        # new indexes (point locations) to old indexes.
        map_from_mesh_to_pts = relate_closest(
            self.mesh.points, mesh.coordinates())[0][:, 1]
        fiber_pts_vec = fiber_pts_vec.take(map_from_mesh_to_pts, axis=0)
        sheet_pts_vec = sheet_pts_vec.take(map_from_mesh_to_pts, axis=0)
        sheet_normal_pts_vec = sheet_normal_pts_vec.take(
            map_from_mesh_to_pts, axis=0)
        # Add data to mesh
        self.add_fibers(LV_FIBERS.F0, fiber_pts_vec)
        self.add_fibers(LV_FIBERS.S0, sheet_pts_vec)
        self.add_fibers(LV_FIBERS.N0, sheet_normal_pts_vec)
        # Convert nodal data to cell data
        self.transform_point_data_to_cell_data(LV_FIBERS.F0, "mean", axis=0)
        self.transform_point_data_to_cell_data(LV_FIBERS.S0, "mean", axis=0)
        self.transform_point_data_to_cell_data(LV_FIBERS.N0, "mean", axis=0)
        # Compute angles between normal and fiber vectors
        fiber_angles = np.degrees(self.compute_angles_wrt_normal(
            fiber_pts_vec, False, False) - np.pi*0.5)
        sheet_angles = np.degrees(self.compute_angles_wrt_normal(
            sheet_pts_vec, False, False) - np.pi*0.5)
        sheet_normal_angles = np.degrees(self.compute_angles_wrt_normal(
            sheet_normal_pts_vec, False, False) - np.pi*0.5)
        # add angles
        self.mesh.point_data[LV_FIBERS.F0_ANGLES.value] = fiber_angles
        self.mesh.point_data[LV_FIBERS.S0_ANGLES.value] = sheet_angles
        self.mesh.point_data[LV_FIBERS.N0_ANGLES.value] = sheet_normal_angles
        # Convert nodal data to cell data
        self.transform_point_data_to_cell_data(
            LV_FIBERS.F0_ANGLES, "mean", axis=0)
        self.transform_point_data_to_cell_data(
            LV_FIBERS.S0_ANGLES, "mean", axis=0)
        self.transform_point_data_to_cell_data(
            LV_FIBERS.N0_ANGLES, "mean", axis=0)

    # =============================================================================
    # Boundary conditions

    def create_spring_rim_bc(self,
                             bc_name: str,
                             surface: str,
                             dist_from_c: float = 10.0,
                             height: float = 2,
                             r_alpha: float = 0.8
                             ) -> dict:
        """Adds information for boundary condition at defined surface. \
            The current setup creates a 'rim' geometry at a distance from \
            the surface's center and relates the 'rim' nodes with the surface \
            nodes based on closest distances. This setup is used to create \
            springs during FEA simulations as BCs.

        Args:
            dist_from_c (float, optional): Perpendicular distance determining \
                the offset from surface's center. Defaults to 10.0.
            height (float, optional): Height of the rim. Defaults to 2.
            r_alpha (float, optional): Percentage of found surface's radius. \
                Will adjust the radius of the rim based on surface_r*r_alpha.\
                Defaults to 0.8.

        Returns:
            dict: Rim data. Keys are defined based on LV_RIM Enum.
        """
        # set surface
        if isinstance(surface, Enum):
            surf_name = surface.name
            surface = surface.value
        else:
            surf_name = surface

        #  -- possible mitral BCs --
        if surface == LV_SURFS.MITRAL.value:
            if len(self.mitral_info) == 0:
                raise RuntimeError(
                    "Mitral info not found. Did you identify LV surfaces?")
            c = self.mitral_info[LV_AM_INFO.CENTER.value]
            r = self.mitral_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.BORDER_MITRAL.value:
            if len(self.mitral_info) == 0:
                raise RuntimeError(
                    "Mitral info not found. Did you identify LV surfaces?")
            c = self.mitral_info[LV_AM_INFO.BORDER_CENTER.value]
            r = self.mitral_info[LV_AM_INFO.BORDER_RADIUS.value]
        # -- possible aortic BCs --
        elif surface == LV_SURFS.AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError(
                    "Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.CENTER.value]
            r = self.aortic_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.ENDO_AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError(
                    "Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.CENTER.value]
            r = self.aortic_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.EPI_AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError(
                    "Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.CENTER.value]
            r = self.aortic_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.BORDER_AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError(
                    "Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.BORDER_CENTER.value]
            r = self.aortic_info[LV_AM_INFO.BORDER_RADIUS.value]
        else:
            raise ValueError("Surface '{}' not valid or not yet implemented \
                for this boundary condition.".format(surf_name))

        # select pts at surface
        ioi = self.get_nodeset(surface)
        pts = self.points(mask=ioi)
        # get lv normal
        # lvnormal = self.get_normal()
        # fit a plat on pts and get plane normal (will be the rim's normal)
        n, _ = fit_plane(pts)
        n = -n if n[2] < 0 else n
        # get a second reference vector
        x = np.cross(n, self._Z)
        # get center of the rim and adjust its position based on user-defined distance
        c = c + n*dist_from_c
        # set radius of the rim as a percentage 'alpha' of the rim
        r = r * r_alpha
        # create rim nodes and set relations with surface nodes
        rim, rim_center, rim_el = create_rim_circunference(c, r, height, -n, x)
        nodes_rim_relations, nodes_rim_dists = relate_closest(pts, rim)
        nodes_rim_relations[:, 0] = ioi

        rim_data = {
            LV_RIM.NODES.value: rim,
            LV_RIM.CENTER.value: rim_center,
            LV_RIM.ELEMENTS.value: rim_el,
            LV_RIM.RELATIONS.value: nodes_rim_relations,
            LV_RIM.DISTS.value: nodes_rim_dists,
            LV_RIM.REF_NODESET.value: surface
        }

        # save discrete set
        self.add_discrete_set(bc_name, nodes_rim_relations)
        self.add_bc(bc_name, LV_BCS.RIM_SPRINGS.value,
                    rim_data)  # save bc data

        return rim_data

    def get_rim_springs_for_plot(
            self,
            rim_data: dict,
            n_skip: int = 1):

        rim_ref_nods = rim_data[LV_RIM.REF_NODESET.value]
        rim_pts = rim_data[LV_RIM.NODES.value]
        relations = rim_data[LV_RIM.RELATIONS.value]

        # geo_pts = self.points(mask=self.get_nodeset(rim_ref_nods))
        pts_a = self.points(mask=relations[:, 0][::n_skip])
        pts_b = rim_pts[relations[:, 1]][::n_skip]

        lines = None
        for a, b in zip(pts_a, pts_b):
            if lines is None:
                lines = lines_from_points(np.array([a, b]))
            else:
                lines = lines.merge(lines_from_points(np.array([a, b])))
        return lines
