from project_heart.modules.container import BaseContainerHandler
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import *
from project_heart.utils.cloud_ops import *
from .lv_base import LV_Base

from collections import deque

from project_heart.enums import *
from sklearn.cluster import KMeans

from functools import reduce

from pathlib import Path
import os


class LV_RegionIdentifier(LV_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def identify_base_and_apex_regions(self, ab_n=10, ab_ql=0.03, ab_qh=0.75, **kwargs):
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
        self.surface_mesh.point_data[LV_MESH_DATA.APEX_BASE_REGIONS.value] = surf_regions.astype(
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

    def identify_epi_endo_regions(self, threshold: float = 90.0, ref_point: np.ndarray = None) -> tuple:
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
        self.surface_mesh.point_data[LV_MESH_DATA.EPI_ENDO_GUESS.value] = endo_epi_guess.astype(
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

    def identify_mitral_and_aortic_regions(self,
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
        self.surface_mesh.point_data[LV_MESH_DATA.AM_DETAILED.value] = clustered.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.AM_DETAILED.value] = mesh_clustered.astype(
            np.int64)

        self.surface_mesh.point_data[LV_MESH_DATA.AM_SURFS.value] = am_highlighted.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.AM_SURFS.value] = mesh_am_highlighted.astype(
            np.int64)

        self.surface_mesh.point_data[LV_MESH_DATA.AM_EPI_ENDO.value] = am_endo_epi_regions.astype(
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

    def identify_regions(self,
                         endo_epi_args={},
                         apex_base_args={},
                         aortic_mitral_args={},
                         create_nodesets=True,

                         ):

        endo_epi, mesh_endo_epi = self.identify_epi_endo_regions(
            **endo_epi_args)
        apex_base, mesh_apex_base = self.identify_base_and_apex_regions(
            **apex_base_args)
        aortic_mitral, mesh_aortic_mitral = self.identify_mitral_and_aortic_regions(
            **aortic_mitral_args)

        # ------------------------------
        # update endo-epi based on detailed info from aortic_mitral surfs
        # now that we have all surface IDs, we can adjust endo_epi based on detailed
        # information from aortic and mitral data.
        surf_to_global = self.surface_mesh.point_data["vtkOriginalPointIds"]

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
        self.surface_mesh[LV_MESH_DATA.EPI_ENDO.value] = updated_endo_epi
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
        self.surface_mesh[LV_MESH_DATA.AB_ENDO_EPI.value] = updated_apex_base.astype(
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
        self.surface_mesh.point_data[LV_MESH_DATA.SURFS_DETAILED.value] = layers.astype(
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
        am_values = self.surface_mesh.get_array(
            LV_MESH_DATA.AM_SURFS.value, "points")
        ioi = np.where(am_values != LV_SURFS.OTHER.value)[0]
        layers[ioi] = am_values[ioi]
        # match indexes of interest at mesh (global) level
        mesh_layers[surf_to_global] = layers
        self.surface_mesh.point_data[LV_MESH_DATA.SURFS.value] = layers.astype(
            np.int64)
        self.mesh.point_data[LV_MESH_DATA.SURFS.value] = mesh_layers.astype(
            np.int64)

        # create nodesets
        if create_nodesets:
            self.create_nodesets_from_regions(
                mesh_data=LV_MESH_DATA.APEX_BASE_REGIONS.value, overwrite=False)
            self.create_nodesets_from_regions(
                mesh_data=LV_MESH_DATA.EPI_ENDO.value, overwrite=False)
            self.create_nodesets_from_regions(
                mesh_data=LV_MESH_DATA.SURFS.value, overwrite=False)
            self.create_nodesets_from_regions(
                mesh_data=LV_MESH_DATA.AM_SURFS.value, overwrite=False)
            self.create_nodesets_from_regions(
                mesh_data=LV_MESH_DATA.SURFS_DETAILED.value, overwrite=False)

        # set flag to indicate surfaces were identified from this method:
        self._surfaces_identified_with_class_method = True

    def peek_unique_values_in_region(self, surface_name: str, enum_like: Enum = None) -> list:
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
