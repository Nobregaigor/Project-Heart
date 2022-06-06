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

import logging
logger = logging.getLogger('LV_RegionIdentifier')

class LV_RegionIdentifier(LV_Base):
    def __init__(self, geo_type=None, *args, **kwargs):
        super(LV_RegionIdentifier, self).__init__(*args, **kwargs)
        self.geo_type = geo_type
                
        self.apex_and_base_from_nodeset = None
        self._apex_from_nodeset = None
        self._base_from_nodeset = None

    # =========================================================================
    # POINT DATA REGION IDENTIFICATION

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # REGION METHODS

    # ----------------------------------------------------------------
    # Common to all types of geometries

    def identify_est_base_and_apex_regions(self, n=5, ql=None, qh=None, 
            log_level=logging.INFO, **kwargs) -> tuple:
        """Identifies the estimated basal and apical regions. 
        The algorithm starts by computing 2 kmeans cluster with the entire 
        geometry and uses both centers as an estimation for LV orientation 
        (longitudinal axis). It then starts to iteractively and virtually 
        rotate geometry until longitudinal axis is closely aligned with 
        Z axis. During iteractive process, apex and base are computed 
        using ql and qh (percentage low and percentage high based on  
        lowest and highest node along Z axis).  
  
        Note: this is an estimated value. It will not compute the exact
        base and apex regions; see region identifier function for proper 
        geometry. This methods is a starting point for other methods and 
        assumes that no information about LV mesh is avaiable. 

        Saved as 'LV_MESH_DATA.APEX_BASE_EST' at surface and mesh containers.

        Args:
            n (int, optional): Number of iterations to perform during 
                iteractive process. Defaults to 5.
            ql (_type_, optional): Percentage low from lowest point along Z axis.  
                Defaults to None. 
            qh (_type_, optional): Percentage high from highest point along Z axis. 
                Defaults to None.
            log_level (_type_, optional): Logger logging level. Defaults to logging.INFO.

        Returns:
            tuple: (surface region, mesh region)
        """
        log = logger.getChild("identify_base_and_apex_regions")
        log.setLevel(log_level)
        log.debug("Starting identification of base and apex regions.")

        # extract surface mesh (use extract)
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points

        # split the surface into two main clusters
        # this will result in top and bottom halves
        log.debug("Perfoming kmeans to find LV halves")
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pts)
        kcenters = kmeans.cluster_centers_
        kcenters = kcenters[np.argsort(kcenters[:, -1])]
        log.debug("kcenters: {}".format(kcenters))

        # estimate initial rotation based on kcenters
        # this will provide a general estimation of LV orientation and 
        # initiate the approximation of apex/base based on normal orientation
        # along z-axis.
        rot_chain = deque()
        # set vectors for rotation
        edge_long_vec = kcenters[1] - kcenters[0]
        # get rotation matrix
        rot = get_rotation(edge_long_vec, self._Z)
        rot_chain.append(rot)
        initial_rot = rot.apply(pts)
        # estimate apex and base based on iteractive aligment
        log.debug("Estimating apex and base iteractively.")
        aligment_data = self.est_apex_and_base_refs_iteratively(pts, 
                            n=n, ql=ql, qh=qh, rot_chain=rot_chain)
        apex_est = aligment_data["apex"]
        base_est = aligment_data["base"]
        log.debug("apex: {}".format(apex_est))
        log.debug("base: {}".format(base_est))

        # retrieve estimation info
        est_apex_region = aligment_data["apex_region"]
        est_base_region = aligment_data["base_region"]
        log.debug("len(est_apex_region): {}".format(len(est_apex_region)))
        log.debug("len(est_base_region): {}".format(len(est_base_region)))
        surf_regions = np.zeros(len(pts))
        surf_regions[est_apex_region] = self.REGIONS.APEX_EST.value
        surf_regions[est_base_region] = self.REGIONS.BASE_EST.value
        self.set_region_from_surface_ids(LV_MESH_DATA.APEX_BASE_EST, surf_regions)

        # add apex and base virtual nodes
        self.add_virtual_node(LV_VIRTUAL_NODES.APEX, apex_est, True)
        self.add_virtual_node(LV_VIRTUAL_NODES.BASE, base_est, True)
        # for consistency, we will call functions to compute 
        # longitudinal line and normal.
        self.compute_long_line(apex=apex_est, base=base_est)
        self.compute_normal(apex=apex_est, base=base_est)

        return (self.get(self.CONTAINERS.SURF_POINT_DATA, LV_MESH_DATA.APEX_BASE_EST),
                self.get(self.CONTAINERS.MESH_POINT_DATA, LV_MESH_DATA.APEX_BASE_EST))

    def identify_epi_endo_regions(self, threshold: float = 90.0, ref_point: np.ndarray = None,
        log_level=logging.INFO,
        **kwargs) -> tuple:
        """Estimates Epicardium and Endocardium surfaces based on the angle between 
            vectors from the reference point to each node and their respective
            surface normals. If ref_point is not provided, it will use the center 
            of the geometry.
            
        Args:
            threshold (float, optional): Angles less than threshold will be considered 
                part of 'Endocardium' while others will be considered part of 'Epicardium'. 
                Defaults to 90.0.
            ref_point (np.ndarray or None, optional): Reference point; must be a (1x3) 
                np.ndarray specifying [x,y,z] coordinates. If set to None, the function 
                will use the estimated center of the geometry.

        Returns:
            tuple: Arrays containing values for each node as 'Endo', 'Epi' or Neither  
                    at surface mesh and global mesh.
        """
        log = logger.getChild("identify_epi_endo_regions")
        log.setLevel(log_level)
        log.debug("Starting identification of endo and epi regions for ideal geometry.")

        # extract surface mesh (use extract)
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points
        # get surface Normals and get respective points-data
        lvsurf.compute_normals(inplace=True)
        surf_normals = lvsurf.get_array("Normals", "points")
        # if ref_point was not specified, est geometry center
        if ref_point is None:
            ref_point = centroid(pts)
        else:
            if not isinstance(ref_point, np.ndarray):
                ref_point = np.array(ref_point)
        log.debug("Using reference point: {}".format(ref_point))
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

        n_endo = len(endo_ids)
        n_epi = len(epi_ids)
        n_pts = len(pts)
        log.debug("Number of endo ids found: {} ({:.2f}%)".format(n_endo, (n_endo/n_pts)*100 ))
        log.debug("Number of epi ids found: {} ({:.2f}%)".format(n_epi, (n_epi/n_pts)*100))

        self.set_region_from_surface_ids(LV_MESH_DATA.EPI_ENDO_EST, endo_epi_guess)
        self.set_region_from_surface_ids(LV_MESH_DATA.EPI_ENDO, endo_epi_guess)

        return (self.get(self.CONTAINERS.SURF_POINT_DATA, LV_MESH_DATA.EPI_ENDO),
                self.get(self.CONTAINERS.MESH_POINT_DATA, LV_MESH_DATA.EPI_ENDO))

    # ----------------------------------------------------------------
    # Geometries of 'type ideal' -> Ellipsoid

    def identify_base_region_ideal(self,
                                   fe=45,
                                   db=0.1,
                                   ba=90,
                                   axis=None,
                                   log_level=logging.INFO,
                                   ):

        if not self.check_geo_type_ideal():
            raise RuntimeError(
                "LV Geometry type must be set as 'ideal' to run this function.")
        try:
            endo_epi = np.copy(
                self.get(self.CONTAINERS.SURF_POINT_DATA, LV_MESH_DATA.EPI_ENDO))
        except KeyError:
            raise RuntimeError(
                "Endo-epi regions were not identified. Either set it manually or use 'identify_epi_endo_regions'. ")
        log = logger.getChild("identify_base_region_ideal")
        log.setLevel(log_level)
        log.debug("Starting identification of base region for ideal geometry.")

        # get surface mesh
        lvsurf = self.get_surface_mesh()
        # Get edges
        edges = self.mesh.extract_feature_edges(fe)
        edge_pts = edges.points
        est_base = centroid(edge_pts)
        est_radius = radius(edge_pts)
        log.debug("Number of edge points found: {}".format(len(edge_pts)))
        log.debug("est_base: {}".format(est_base))
        log.debug("est_radius: {}".format(est_radius))
        # select pts close to est_base based on dist threshold on axis
        log.debug("Selecting points close to est_base based on 'd'.")
        pts = lvsurf.points
        # get axis orientation
        lvnormal = self.get_normal()
        if axis == None:
            axis = np.where(lvnormal == np.max(lvnormal))[0]
        log.debug("axis orientation: {}".format(axis))
        d_base = np.abs(est_base[axis] - pts[:, axis])
        ioi = np.where(d_base <= db)[0]
        log.debug("number of indexed found at 'db={}' from 'est_base={}'"
                  "along 'axis={}'".format(db, est_base, axis))
        # filter selected pts based on surface angle
        log.debug("Filtering selection based on surface normals.")
        lvsurf.compute_normals(inplace=True)
        surf_normals = lvsurf.get_array("Normals", "points")
        # select reference vector based on orientation axis
        if axis == 0:
            ref_vec = self._X
        elif axis == 1:
            ref_vec = self._Y
        elif axis == 2:
            ref_vec = self._Z
        vec_arr = np.repeat(np.expand_dims(ref_vec, 1), len(ioi), axis=1).T
        base_angles = angle_between(
            surf_normals[ioi], vec_arr, check_orientation=False)
        # filter by angle w.r.t. orientation axis
        ioi = ioi[np.where(base_angles <= np.radians(ba))[0]]
        log.debug("Number of ioi found: {}".format(len(ioi)))
        # filter by endo (don't overlap endo values)
        log.debug("Filtering selection based on surface endo region (no overlap).")
        ioi = ioi[np.where(endo_epi[ioi] != LV_SURFS.ENDO)]
        log.debug("Number of ioi found: {}".format(len(ioi)))
        # identify final surfaces
        endo_epi_base = np.copy(endo_epi)
        endo_epi_base[ioi] = LV_SURFS.BASE
        self.set_region_from_surface_ids(LV_MESH_DATA.SURFS, endo_epi_base)
        log.debug("Adding edge information to detailed surface")
        from project_heart.utils.cloud_ops import relate_closest
        rel_map, d = relate_closest(pts, edge_pts)
        edge_ioi = rel_map[:, 0][np.where(d==0)[0]]
        log.debug("Number of edge ids found: {} (expected: {})".format(len(edge_ioi), len(edge_pts)))
        rel_map, d = relate_closest(edge_pts, pts[endo_epi == LV_SURFS.ENDO])
        endo_edge = edge_ioi[rel_map[:, 0][np.where(d<=np.mean(d))[0]]]
        epi_edge = edge_ioi[rel_map[:, 0][np.where(d>np.mean(d))[0]]]
        log.debug("Number of edge ids found at ENDO: {}".format(len(endo_edge)))
        log.debug("Number of edge ids found at EPI: {}".format(len(epi_edge)))
        endo_epi_base_detailed = np.copy(endo_epi_base)
        endo_epi_base_detailed[endo_edge] = self.REGIONS.BASE_BORDER_ENDO
        endo_epi_base_detailed[epi_edge] = self.REGIONS.BASE_BORDER_EPI
        self.set_region_from_surface_ids(LV_MESH_DATA.SURFS_DETAILED, endo_epi_base_detailed)
        
        return (self.get(self.CONTAINERS.SURF_POINT_DATA, LV_MESH_DATA.SURFS),
                self.get(self.CONTAINERS.MESH_POINT_DATA, LV_MESH_DATA.SURFS))

    # ----------------------------------------------------------------
    # Geometries of 'type A' -> No mitral and aortic valve.

    def identify_base_region_typeA(self,
                                      fe=15,
                                      br=1.25,
                                      ba=100,
                                      log_level=logging.INFO):
        if not self.check_geo_type_typeA():
            raise RuntimeError(
                "LV Geometry type must be set as 'typeA' or 'nonIdeal' to run this function.")
        try:
            endo_epi = np.copy(
                self.get(GEO_DATA.SURF_POINT_DATA, LV_MESH_DATA.EPI_ENDO))
        except KeyError:
            raise RuntimeError(
                "Endo-epi regions were not identified. Either set it manually or use 'identify_epi_endo_regions'. ")
        
        log = logger.getChild("identify_base_region_typeA")
        log.setLevel(log_level)
        log.debug("Starting identification of base region for 'type A' geometry.")

        # get surface mesh
        lvsurf = self.get_surface_mesh()
        # Get edges
        edges = self.mesh.extract_feature_edges(fe)
        edges = edges.extract_largest()
        edges = edges.extract_largest()
        edge_pts = edges.points
        est_base = centroid(edge_pts)
        est_radius = radius(edge_pts)
        log.debug("Number of edge points found: {}".format(len(edge_pts)))
        log.debug("est_base: {}".format(est_base))
        log.debug("est_radius: {}".format(est_radius))

        # select pts close to est_base based on % of est_radius
        log.debug("Selecting nodes based on distance from est_base.")
        pts = lvsurf.points
        d_base = np.linalg.norm(pts - est_base, axis=1)
        ioi = np.where(d_base <= est_radius*br)[0]
        log.debug("number of indexed found at 'br={}' from 'est_radius*br={}': {}"
                  .format(br, est_radius*br, len(ioi)))
        # re-estimate base centroid and radius
        poi = pts[ioi]

        # filter selected pts based on surface angle
        log.debug("Filtering selection based on surface normals.")
        lvsurf.compute_normals(inplace=True)
        surf_normals = lvsurf.get_array("Normals", "points")
        base_vecs = est_base - poi
        base_angles = angle_between(
            surf_normals[ioi], base_vecs, check_orientation=False)
        ioi = ioi[np.where(base_angles <= np.radians(ba))[0]]
        log.debug("Number of ioi found: {}".format(len(ioi)))

        # filter by endo
        log.debug("Filtering selection based on surface endo region (no overlap).")
        ioi = ioi[np.where(endo_epi[ioi] != LV_SURFS.ENDO)]
        log.debug("Number of ioi found: {}".format(len(ioi)))

        endo_epi_base = np.copy(endo_epi)
        endo_epi_base[ioi] = LV_SURFS.BASE
        self.set_region_from_surface_ids(LV_MESH_DATA.SURFS, endo_epi_base)

    # ----------------------------------------------------------------
    # Geometries of 'type B' -> With both mitral and aortic valve.

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

        if not self.check_geo_type_typeB():
            raise RuntimeError(
                "LV Geometry type must be set as 'typeB' to run this function.")

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

        # surf_to_global = np.array(
        #     lvsurf.point_data["vtkOriginalPointIds"], dtype=np.int64)
        surf_to_global = self.get_surface_id_map_from_mesh()
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

    def _identify_typeB_regions(self,
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
        
        # surf_to_global = self.surface_mesh.point_data["vtkOriginalPointIds"]
        surf_to_global = self.get_surface_id_map_from_mesh()

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

        # set flag to indicate surfaces were identified from this method:
        self._surfaces_identified_with_class_method = True
    
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # NODESET-DEPENDENT METHODS

    def identify_endo_epi_base_borders_from_nodesets(self, thresh_base=2, thresh_endo=2, thresh_epi=2, log_level=logging.INFO):
        log = logger.getChild("identify_endo_epi_base_borders_from_nodesets")
        log.setLevel(log_level)
        endo_base = self.compute_boundary_between_nodesets(
                nodesets=[self.REGIONS.BASE, self.REGIONS.ENDO],
                thresh_vals=[thresh_base, thresh_endo],
                log_level=log_level
            )
        log.info("len(endo_base): {}".format(len(endo_base)))
        epi_base = self.compute_boundary_between_nodesets(
                nodesets=[self.REGIONS.BASE, self.REGIONS.EPI],
                thresh_vals=[thresh_base, thresh_epi],
                log_level=log_level
            )
        log.info("len(epi_base): {}".format(len(epi_base)))
        surf_data = self.get(self.CONTAINERS.MESH_POINT_DATA, LV_MESH_DATA.SURFS)
        surf_data[endo_base] = self.REGIONS.BASE_BORDER_ENDO
        surf_data[epi_base] = self.REGIONS.BASE_BORDER_EPI
        self.set_region_from_mesh_ids(LV_MESH_DATA.SURFS_DETAILED, surf_data)
    
    def identify_apex_base_from_nodesets(self, apex_nodeset=None, base_nodeset=None):
        if apex_nodeset is None:
            apex_nodeset = self.REGIONS.APEX
        if base_nodeset is None:
            base_nodeset = self.REGIONS.BASE

        apex_ids = self.get_nodeset(apex_nodeset)
        base_ids = self.get_nodeset(base_nodeset)
        region = np.zeros(self.mesh.n_points)
        region[base_ids] = self.REGIONS.BASE
        region[apex_ids] = self.REGIONS.APEX
        self.set_region_from_mesh_ids(LV_MESH_DATA.APEX_BASE, region)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # COMPILED METHODS

    def identify_regions_ideal(self, 
        base_vn_nodeset=None, # default = "base_endo"
        apex_vn_nodeset=None, # default = "endo"
        d_apex=2.5,
        apex_base_args=None, 
        endo_epi_args=None, 
        base_args=None,
        log_level=logging.INFO, 
        **kwargs):

        log = logger.getChild("identify_regions_ideal")
        log.setLevel(log_level)
        log.info("Identifying regions from ideal geometry.")
        if not self.check_geo_type_ideal():
            raise RuntimeError(
                "LV Geometry type must be set as 'ideal' to run this function.")

        # set default values
        if apex_base_args is None:
            apex_base_args = dict()
        if endo_epi_args is None:
            endo_epi_args = dict()
        if base_args is None:
            base_args = dict()
        
        # begin identification
        self.identify_est_base_and_apex_regions(log_level=log_level, **apex_base_args)
        self.identify_epi_endo_regions(log_level=log_level, **endo_epi_args)
        self.identify_base_region_ideal(log_level=log_level, **base_args)
        # create nodesets
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.APEX_BASE_EST, overwrite=False)
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.EPI_ENDO, overwrite=False)
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.SURFS, overwrite=True)
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.SURFS_DETAILED, overwrite=False)

        # compute base from endo_border (or user-provided) nodeset
        if base_vn_nodeset is None:
            base_vn_nodeset = self.REGIONS.BASE_BORDER_ENDO
        base_ref = self.compute_base_from_nodeset(base_vn_nodeset)
        log.debug("base_ref: {}".format(base_ref))

        # compute apex from "endo" (or user-provided) nodeset
        if apex_vn_nodeset is None:
            apex_vn_nodeset = self.REGIONS.ENDO
        _, apex_region_ids = self.compute_apex_from_base_vn(
                                        d=d_apex, 
                                        nodeset=apex_vn_nodeset, 
                                        log_level=log_level)
        self.add_nodeset(self.REGIONS.APEX, apex_region_ids, overwrite=True)

        # compute 'final' apex and base regions (mainly for debuging)
        self.identify_apex_base_from_nodesets()
        
        # set infos
        self.set_base_info(LV_SURFS.BASE)

        # compute longitudinal axis and normals
        self.compute_long_line()
        self.compute_normal()

    def identify_regions_typeA(self, 
        border_thresh_base=2, 
        border_thresh_endo=2, 
        border_thresh_epi=2,
        base_vn_nodeset=None, # default = "base_endo"
        apex_vn_nodeset=None, # default = "endo"
        d_apex=5,
        apex_base_args=None, 
        endo_epi_args=None, 
        base_args=None,
        log_level=logging.INFO, 
        **kwargs):

        log = logger.getChild("identify_regions_typeA")
        log.setLevel(log_level)
        log.info("Identifying regions from 'type A' geometry.")
        if not self.check_geo_type_typeA():
            raise RuntimeError(
                "LV Geometry type must be set as 'type A' to run this function.")

        # set default values
        if apex_base_args is None:
            apex_base_args = dict(n=1, ql=0.1, qh=0.7)
        if endo_epi_args is None:
            endo_epi_args = dict(threshold=85)
        if base_args is None:
            base_args = dict()
        
        # begin identification
        self.identify_est_base_and_apex_regions(log_level=log_level, **apex_base_args)
        self.identify_epi_endo_regions(log_level=log_level, **endo_epi_args)
        self.identify_base_region_typeA(log_level=log_level, **base_args)
        # create nodesets
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.APEX_BASE_EST, overwrite=False)
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.EPI_ENDO, overwrite=False)
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.SURFS, overwrite=True)
        
        # Compute endo-epi base borders (adds base_endo and base_epi)
        self.identify_endo_epi_base_borders_from_nodesets(thresh_base=border_thresh_base, 
                                                          thresh_endo=border_thresh_endo, 
                                                          thresh_epi=border_thresh_epi,
                                                          log_level=log_level)
        self.create_nodesets_from_regions(
            mesh_data=LV_MESH_DATA.SURFS_DETAILED, overwrite=False)

        # compute base from endo_border (or user-provided) nodeset
        if base_vn_nodeset is None:
            base_vn_nodeset = self.REGIONS.BASE_BORDER_ENDO
        base_ref = self.compute_base_from_nodeset(base_vn_nodeset)
        log.debug("base_ref: {}".format(base_ref))

        # compute apex from "endo" (or user-provided) nodeset
        if apex_vn_nodeset is None:
            apex_vn_nodeset = self.REGIONS.ENDO
        _, apex_region_ids = self.compute_apex_from_base_vn(
                                        d=d_apex, 
                                        nodeset=apex_vn_nodeset, 
                                        log_level=log_level)
        self.add_nodeset(self.REGIONS.APEX, apex_region_ids, overwrite=True)

        # compute 'final' apex and base regions (mainly for debuging)
        self.identify_apex_base_from_nodesets()
        
        # set infos
        self.set_base_info(LV_SURFS.BASE)

        # compute longitudinal axis and normals
        self.compute_long_line()
        self.compute_normal()

    def identify_regions_typeB(self):
        raise NotImplementedError("Working on it.")

    def identify_regions(self,
                         geo_type=None,
                        #  apex_base_args=None,
                        #  endo_epi_args=None,
                        #  base_args=None,
                        #  aortic_mitral_args=None,
                        #  create_nodesets=True,
                        #  set_infos=True,
                        #  set_normal=False,  
                        #  recompute_apex_base=True,
                         **kwargs
                         ):

        if geo_type is None:
            geo_type = self.geo_type
        else:
            geo_type = self.check_enum(geo_type)
            self.geo_type = geo_type

        if self.check_no_geo_type():
            raise ValueError(
                "Must specify a geo_type either at object initialization, or with 'geo_type' argument.")

        if geo_type == LV_GEO_TYPES.IDEAL:
            self.identify_regions_ideal(**kwargs)
        elif geo_type == LV_GEO_TYPES.TYPE_A:
            self.identify_regions_typeA(**kwargs)
        elif geo_type == LV_GEO_TYPES.TYPE_B:
            raise NotImplementedError("Working on fix.")
            # self._identify_typeB_regions(
            #     apex_base_args=apex_base_args,
            #     endo_epi_args=endo_epi_args,
            #     aortic_mitral_args=aortic_mitral_args)
            # # create nodesets
            # # if create_nodesets:
            # self.create_nodesets_from_regions(
            #     mesh_data=LV_MESH_DATA.APEX_BASE_REGIONS.value, overwrite=False)
            # self.create_nodesets_from_regions(
            #     mesh_data=LV_MESH_DATA.EPI_ENDO.value, overwrite=False)
            # self.create_nodesets_from_regions(
            #     mesh_data=LV_MESH_DATA.SURFS.value, overwrite=False)
            # self.create_nodesets_from_regions(
            #     mesh_data=LV_MESH_DATA.AM_SURFS.value, overwrite=False)
            # self.create_nodesets_from_regions(
            #     mesh_data=LV_MESH_DATA.SURFS_DETAILED.value, overwrite=True)
            
            # # create 'base' nodeset from AM data
            # from functools import reduce
            # base = reduce(np.union1d, 
            #                     (
            #                         self.get_nodeset(self.REGIONS.AORTIC),
            #                         self.get_nodeset(self.REGIONS.MITRAL),
            #                         self.get_nodeset(self.REGIONS.AM_INTERCECTION)
            #                     )
            #                 )
            # self.add_nodeset(self.REGIONS.BASE, base, True)
            # # if set_infos: -> these are already created from '_identify_typeB_regions'
            # #     self.set_aortic_info(LV_SURFS.AORTIC)
            # #     self.set_mitral_info(LV_SURFS.MITRAL)
        else:
            raise ValueError(
                "Invalid geo type: %s. Check LV_GEO_TYPES enums for valid types." % geo_type)

        # # ----
        # # apply functions for all geo types
        # # creatre nodeset for epi and endo without base ids
        # self.set_epi_endo_exclude_base_nodeset()
        # self.set_endo_plus_base_nodeset()

        # self.compute_base_intersections()
        # # recompute apex and base virtual nodes based on nodesets created from 
        # # regions. Default is BASE_REGION and APEX_REGION regions. 
        # if recompute_apex_base is not None:
        #     if isinstance(recompute_apex_base, dict):
        #     #     self.set_apex_and_base_from_nodeset(**recompute_apex_base)
        #         self.compute_apex_and_base_ref_from_nodesets(**recompute_apex_base)
        #     elif isinstance(recompute_apex_base, bool) and recompute_apex_base == True:
        #         self.compute_apex_and_base_ref_from_nodesets(
        #             apex_nodeset="APEX_REGION")
        
    # =========================================================================
    # FACET DATA REGION TRANSFORMATION

    # ----------------------------------------------------------------
    # Region to facet data

    def transform_region_to_facet_data(self, region, method="max",
                                       epi_endo_correction=True,
                                       **kwargs):

        # if self.check_geo_type_typeB() and not self.check_tri3_surfmesh():
        #     print("WARNING: This method is not properly working with this configuration. Please, consider using a mesh with triangular surface.")

        if not epi_endo_correction or method != "max":
            facet_data = self.transform_surface_point_data_to_facet_data(
                region, method, **kwargs)
        else:
            # to account to shar edges (mostly for ideal cases), we need to prevent
            # that base region expands beyond its limits. To fo so, we can simply
            # adjust numerical values at endo and epi regions in such a way that
            # if we take the maximum value, they will be selected correctly.
            # Therefore, we must raise their respective values.
            # Set temporary endo and epi values
            tmp_epi = LV_SURFS.EPI.value * 100
            tmp_endo = LV_SURFS.ENDO.value * 100
            tmp = np.copy(self.get(GEO_DATA.SURF_POINT_DATA, region))
            # modify tmp content
            tmp[np.where(tmp == LV_SURFS.EPI.value)[0]] = tmp_epi
            tmp[np.where(tmp == LV_SURFS.ENDO.value)[0]] = tmp_endo
            # apply transformation on temporary data
            self.set_surface_point_data("TMP-REGION", tmp)
            facet_data = self.transform_surface_point_data_to_facet_data(
                "TMP-REGION", method, **kwargs)
            # return original values for endo and epi
            facet_data[np.where(facet_data == tmp_epi)[0]] = LV_SURFS.EPI.value
            facet_data[np.where(facet_data == tmp_endo)[
                0]] = LV_SURFS.ENDO.value
            # set new data
            self.set_facet_data(region, facet_data)
            # remove temporary data
            self.surface_mesh.point_data.pop("TMP-REGION")
            self.surface_mesh.cell_data.pop("TMP-REGION")

        return facet_data

    # =========================================================================
    # Others

    def set_epi_endo_exclude_base_nodeset(self):
        epi = self.get_nodeset(self.REGIONS.EPI)
        endo = self.get_nodeset(self.REGIONS.ENDO)
        base = self.get_nodeset(self.REGIONS.BASE)
        epi_no_base = np.setdiff1d(epi, base)
        endo_no_base = np.setdiff1d(endo, base)
        self.add_nodeset(self.REGIONS.EPI_EXCLUDE_BASE, epi_no_base, True)
        self.add_nodeset(self.REGIONS.ENDO_EXCLUDE_BASE, endo_no_base, True)
    
    def set_endo_plus_base_nodeset(self):
        endo_base = reduce(np.union1d, 
                            (
                                self.get_nodeset(self.REGIONS.ENDO),
                                self.get_nodeset(self.REGIONS.BASE),
                            )
                        )
        self.add_nodeset(self.REGIONS.ENDO_BASE, endo_base, True)
    
    def set_region_from_surface_ids(self, key: str, surf_mesh_ids: list) -> np.ndarray:
        """Sets mesh and surface mesh region from a list of ids based on mesh. \
            Each index corresponds to node id, and each value should represent\
            region id.

        Args:
            key: (str or enum): key identifier for region data
            surf_mesh_ids (list): List of region ids for each point in mesh.

        Raises:
            ValueError: If length of mesh ids is not equal to number of points in mesh.

        Returns:
            Pointer to added surf_mesh_ids
        """

        if len(surf_mesh_ids) != self.get_surface_mesh().n_points:
            raise ValueError(
                "surf_mesh_ids length must correspond to number of points in surface mesh. "
                "Each id should be an integer determining its region."
                "Expected: {}, received: {}"
                .format(self.get_surface_mesh().n_points, len(surf_mesh_ids)))
        key = self.check_enum(key)

        # add data to mesh
        self.set_surface_point_data(key, surf_mesh_ids)
        # add data to surface mesh
        # set global region ids (for entire mesh)
        id_map = self.get_surface_id_map_from_mesh()
        mesh_ids = np.zeros(self.mesh.n_points)
        mesh_ids[id_map] = surf_mesh_ids
        self.set_mesh_point_data(key, mesh_ids)

        return self.get(self.CONTAINERS.MESH_POINT_DATA, key)
    
    def set_region_from_mesh_ids(self, key: str, mesh_ids: list) -> np.ndarray:
        """Sets mesh and surface mesh region from a list of ids based on surface mesh. \
            Each index corresponds to node id, and each value should represent\
            region id.

        Args:
            key: (str or enum): key identifier for region data
            mesh_ids (list): List of region ids for each point in surface mesh.

        Raises:
            ValueError: If length of mesh ids is not equal to number of points in surface mesh.

        Returns:
            Pointer to added mesh_ids
        """

        if len(mesh_ids) != self.mesh.n_points:
            raise ValueError(
                "mesh_ids length must correspond to number of points in surface mesh. "
                "Each id should be an integer determining its region."
                "Expected: {}, received: {}"
                .format(self.mesh.n_points, len(mesh_ids)))

        key = self.check_enum(key)
        # add data to surface mesh
        self.set_mesh_point_data(key, mesh_ids)
        # add data to surface mesh
        id_map = self.get_surface_id_map_from_mesh()
        self.set_surface_point_data(key, mesh_ids[id_map])

        return self.get(self.CONTAINERS.MESH_POINT_DATA, key)

    def set_region_from_nodesets(self, region_key: str, nodeset_keys: list):
        region = np.zeros(self.mesh.n_points, dtype=np.int64)

        for i, key in enumerate(nodeset_keys):
            nodeids = self.get_nodeset(key)
            key = self.check_enum(key)
            try:
                use_key = LV_SURFS[key].name
                use_val = LV_SURFS[key].value
            except KeyError:
                use_key = key
                use_val = i
            region[nodeids] = use_val

        return self.set_region_from_mesh_ids(region_key, region)



        
    

    def set_geo_type(self, geo_type):
        self.geo_type = geo_type
    
    # ----------------------------------------------------------------
    # Check methods

    def check_no_geo_type(self):
        return self.geo_type is None

    def check_geo_type_ideal(self):
        return self.geo_type == LV_GEO_TYPES.IDEAL

    def check_geo_type_typeA(self):
        return self.geo_type == LV_GEO_TYPES.TYPE_A

    def check_geo_type_typeB(self):
        return self.geo_type == LV_GEO_TYPES.TYPE_B

    # ----------------------------------------------------------------
    # Others
    
    # def set_apex_from_nodeset(self, nodeset=None, **kwargs) -> np.ndarray:
    #     if nodeset is None:
    #         nodeset = self.REGIONS.ENDO_BASE
    #     pts=self.points(mask=self.get_nodeset(nodeset))
    #     (_, es_apex) = self.est_apex_and_base_refs_iteratively(pts, **kwargs)["long_line"]
    #     self.add_virtual_node(LV_VIRTUAL_NODES.APEX, es_apex, replace=True)
    #     self.compute_normal() # update normal
    #     self.compute_long_line() # update long line
    #     self._apex_from_nodeset = nodeset
    #     return self.get_virtual_node(LV_VIRTUAL_NODES.APEX)
    
    # def set_base_from_nodeset(self, nodeset=None, **kwargs) -> np.ndarray:
    #     if nodeset is None:
    #         nodeset = self.REGIONS.ENDO_BASE
    #     pts=self.points(mask=self.get_nodeset(nodeset))
    #     (es_base, _) = self.est_apex_and_base_refs_iteratively(pts, **kwargs)["long_line"]
    #     self.add_virtual_node(LV_VIRTUAL_NODES.BASE, es_base, replace=True)
    #     self.compute_normal() # update normal
    #     self.compute_long_line() # update long line
    #     self._base_from_nodeset = nodeset
    #     return self.get_virtual_node(LV_VIRTUAL_NODES.BASE)
    
    # def set_apex_and_base_from_nodeset(self, nodeset=None, **kwargs) -> np.ndarray:
    #     if nodeset is None:
    #         nodeset = self.REGIONS.ENDO_BASE
    #     pts=self.points(mask=self.get_nodeset(nodeset))
    #     (es_base, es_apex) = self.est_apex_and_base_refs_iteratively(pts, **kwargs)["long_line"]
    #     self.add_virtual_node(LV_VIRTUAL_NODES.BASE, es_base, replace=True)
    #     self.add_virtual_node(LV_VIRTUAL_NODES.APEX, es_apex, replace=True)
    #     self.compute_normal() # update normal
    #     self.compute_long_line() # update long line
    #     self.apex_and_base_from_nodeset = nodeset
    #     return (self.get_virtual_node(LV_VIRTUAL_NODES.APEX), 
    #             self.get_virtual_node(LV_VIRTUAL_NODES.BASE))
        
    # overwrite class compute normal to include identify_base_and_apex_regions
    def compute_normal(self, apex=None, base=None, **kwargs):
        if apex is not None and base is not None:
            self.set_normal(unit_vector(base - apex))
        else:
            try:
                apex = self.get_virtual_node(LV_VIRTUAL_NODES.APEX)
                base = self.get_virtual_node(LV_VIRTUAL_NODES.BASE)
                self.set_normal(unit_vector(base - apex))
            except:
                try:
                    apex = centroid(self.points(
                        mask=self.get_nodeset(LV_SURFS.APEX_REGION)))
                    base = centroid(self.points(
                        mask=self.get_nodeset(LV_SURFS.BASE_REGION)))
                    self.add_virtual_node(LV_VIRTUAL_NODES.APEX, apex)
                    self.add_virtual_node(LV_VIRTUAL_NODES.BASE, base)
                    self.set_normal(unit_vector(base - apex))
                except:
                    try:
                        self.identify_base_and_apex_regions(**kwargs)
                    except:
                        raise RuntimeError(
                            """Unable to compute normal. Prooced with another method\
                            See 'identify_base_and_apex_surfaces' and 'set_normal'\
                            for details.
                            """)

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
