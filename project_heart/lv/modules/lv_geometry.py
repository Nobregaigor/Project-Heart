from project_heart.modules.geometry import Geometry
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from collections import deque

from project_heart.enums import LV_SURFS, LV_MESH_DATA
from sklearn.cluster import KMeans


class LV_Geometry(Geometry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rot_chain = deque()
        self._normal = None

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
        extra = {"apex_region": apex_region_idxs,
                 "base_region": base_region_idxs}
        return np.vstack((base_ref, apex_ref)), extra

    def est_pts_aligment_with_lv_normal(self, points, n=5, rot_chain=[], **kwargs):
        pts = np.copy(points)
        for _ in range(n):
            long_line, _ = LV_Geometry.est_apex_and_base_refs(pts, **kwargs)
            lv_normal = unit_vector(long_line[0] - long_line[1])
            curr_rot = get_rotation(lv_normal, self._Z)
            pts = curr_rot.apply(pts)
            rot_chain.append(curr_rot)
        long_line, data = LV_Geometry.est_apex_and_base_refs(pts, **kwargs)
        lv_normal = unit_vector(long_line[0] - long_line[1])
        data["rot_pts"] = pts
        data["normal"] = lv_normal
        data["long_line"] = long_line
        data["rot_chain"] = rot_chain
        return data

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
        self._surface_mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value] = surf_regions

        # set global region ids (for entire mesh)
        surf_to_global = lvsurf.point_data["vtkOriginalPointIds"]
        global_regions = np.zeros(self.mesh.n_points)
        global_regions[surf_to_global[aligment_data["apex_region"]]
                        ] = LV_SURFS.APEX_REGION
        global_regions[surf_to_global[aligment_data["base_region"]]
                        ] = LV_SURFS.BASE_REGION
        self.mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value] = global_regions
        return (surf_regions, global_regions)

    def identify_surfaces(self,
                          alpha_atr=0.5,  # coeff for radial distance computation
                          alpha_mtr=0.5,
                          beta_atr=0.9,  # coeff for second radial distance computation
                          beta_mtr=0.9,
                          gamma_atr=89,
                          gamma_mtr=89,
                          **kwargs,
                          ):

        assert(alpha_atr > 0.0), "alpha_atr must be positive and greater than 0.0"
        assert(alpha_mtr > 0.0), "alpha_mtr must be positive and greater than 0.0"
        assert(beta_atr > 0.0 and beta_atr <
               1.0), "alpha_atr must be a float greater than 0.0 and less than 1.0"
        assert(beta_mtr > 0.0 and beta_mtr <
               1.0), "beta_mtr must be a float greater than 0.0 and less than 1.0"

        # check if base and apex regions were already identified
        # this is a requirement as we will combine these regions later
        if not LV_MESH_DATA.APEX_BASE_REGION.value in self.mesh.point_data:
            (ab_surf_regions, ab_mesh_regions) = self.identify_base_and_apex_regions(**kwargs)
            ab_surf_regions = ab_surf_regions
            ab_mesh_regions = ab_mesh_regions
        else:
            ab_surf_regions = self._surface_mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value]
            ab_mesh_regions = self.mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value]
        # ensure that we are copying elements and not using original ones
        ab_surf_regions = np.copy(ab_surf_regions)
        ab_mesh_regions = np.copy(ab_mesh_regions)
        ab_surf_apex_ids = np.where(ab_surf_regions == LV_SURFS.APEX_REGION)[0]
        ab_surf_base_ids = np.where(ab_surf_regions == LV_SURFS.BASE_REGION)[0]
        ab_mesh_apex_ids = np.where(ab_mesh_regions == LV_SURFS.APEX_REGION)[0]
        ab_mesh_base_ids = np.where(ab_mesh_regions == LV_SURFS.BASE_REGION)[0]
        

        # extract surface mesh (use extract)
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points

        # ................
        # 1 - Start with an initial guess of endo-epi

        # get surface Normals and get respective points-data
        lvsurf.compute_normals(inplace=True)
        surf_normals = lvsurf.get_array("Normals", "points")
        # est geometry center
        center = np.mean(pts, axis=0)
        # get vector from pts at surface to center
        pts_to_center = center - pts
        # compute angle difference between surface normals and pts to center
        angles = angle_between(pts_to_center, surf_normals,
                                check_orientation=False)  # returns [0 to pi]
        # set initial endo-epi surface guess
        endo_epi_guess = np.zeros(len(pts))
        initial_thresh = np.radians(90)
        endo_ids = np.where(angles < initial_thresh)[0]
        epi_ids = np.where(angles >= initial_thresh)[0]
        endo_epi_guess[endo_ids] = LV_SURFS.ENDO
        endo_epi_guess[epi_ids] = LV_SURFS.EPI
        lvsurf.point_data["endo_epi_guess"] = endo_epi_guess
        lvsurf.set_active_scalars("endo_epi_guess")

        # ................
        # 2 - Find initial guess of Mitral and Aortic clusters

        # compute gradients
        lvsurf = lvsurf.compute_derivative("endo_epi_guess")
        lvsurf = lvsurf.compute_derivative("gradient")
        lvsurf = lvsurf.compute_derivative("gradient")
        # select gradients of interest (threshold based on magnitude)
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        goi = np.copy(grads_mag)
        goi[grads_mag > 0] = 1
        goi[grads_mag <= 0] = 0
        # select points of interest (where goi is positive)
        ioi = np.where(goi > 0)[0]  # indexes of interest
        poi = pts[ioi]             # pts of interest
        # compute centroids at mitral and aortic valves
        kmeans = KMeans(n_clusters=2, random_state=0).fit(poi)
        klabels = kmeans.labels_
        kcenters = kmeans.cluster_centers_
        # determine labels based on centroid closest to center
        kdist = np.linalg.norm(center - kcenters, axis=1)
        label = np.zeros(len(klabels))
        if kdist[0] < kdist[1]:
            label[klabels == 0] = LV_SURFS.MITRAL
            label[klabels == 1] = LV_SURFS.AORTIC
        else:
            label[klabels == 1] = LV_SURFS.MITRAL
            label[klabels == 0] = LV_SURFS.AORTIC
        # define clusters
        clustered = np.zeros(len(pts))
        clustered[ioi] = label
        # set initial guess
        initial_guess = np.copy(endo_epi_guess)
        initial_guess[clustered == LV_SURFS.AORTIC] = LV_SURFS.AORTIC
        initial_guess[clustered == LV_SURFS.MITRAL] = LV_SURFS.MITRAL

        self._surface_mesh.point_data["INITIAL_SURFACE_GUESS"] = initial_guess

        # ................
        # 3 - Refine guess

        # 3.1
        # Select points close to Aortic and Mitral based on the
        # distance from their respective centers and surface pts

        # adjust alpha
        alpha_adj_atr = alpha_atr+1.0
        alpha_adj_mtr = alpha_mtr+1.0

        # -- first pass --
        # select aortic and mitral pts
        atr_mask = np.where(initial_guess == LV_SURFS.AORTIC)[0]
        mtr_mask = np.where(initial_guess == LV_SURFS.MITRAL)[0]
        atr_pts = pts[atr_mask]
        mtr_pts = pts[mtr_mask]
        # compute centers and radius
        c_atr = np.mean(atr_pts, axis=0)
        r_atr = np.mean(np.linalg.norm(atr_pts - c_atr, axis=1))
        c_mtr = np.mean(mtr_pts, axis=0)
        r_mtr = np.mean(np.linalg.norm(mtr_pts - c_mtr, axis=1))
        # compute distance from pts to aortic and mitral centers
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        d_mtr = np.linalg.norm(pts - c_mtr, axis=1)
        # filter by radius
        atr = np.where(d_atr <= r_atr * alpha_adj_atr)[0]
        mtr = np.where(d_mtr <= r_mtr * alpha_adj_mtr)[0]
        # compute intersection between mitral and aortic values
        its = np.intersect1d(atr, mtr)  # intersection

        # -- second pass --
        # Adjust mask: select new aortic and mitral pts
        new_atr_mask = np.union1d(atr_mask, its)
        new_mtr_mask = np.union1d(mtr_mask, its)
        atr_pts = pts[new_atr_mask]
        mtr_pts = pts[new_mtr_mask]
        # re-compute centers and radius
        c_atr = np.mean(atr_pts, axis=0)
        r_atr = np.mean(np.linalg.norm(atr_pts - c_atr, axis=1))
        c_mtr = np.mean(mtr_pts, axis=0)
        r_mtr = np.mean(np.linalg.norm(mtr_pts - c_mtr, axis=1))
        # re-compute distance from pts to aortic and mitral centers
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        d_mtr = np.linalg.norm(pts - c_mtr, axis=1)
        # filter by radius; note that beta will reduce alpha
        atr = np.where(d_atr <= r_atr * (alpha_adj_atr*beta_atr))[0]
        mtr = np.where(d_mtr <= r_mtr * (alpha_adj_mtr*beta_mtr))[0]
        # compute intersection between mitral and aortic values
        its = np.intersect1d(atr, mtr)  # intersection

        # 3.2
        # Select points close to Aortic and Mitral based on the
        # angle in between surface normals and respective centers

        # Adjust mask: select new aortic and mitral pts
        new_atr_mask = atr
        new_mtr_mask = mtr
        atr_pts = pts[new_atr_mask]
        mtr_pts = pts[new_mtr_mask]
        atr_vecs = c_atr - atr_pts
        mtr_vecs = c_mtr - mtr_pts
        # Compute angles
        atr_angles = angle_between(
            surf_normals[new_atr_mask], atr_vecs, check_orientation=False)
        mtr_angles = angle_between(
            surf_normals[new_mtr_mask], mtr_vecs, check_orientation=False)
        # filter by angle
        atr = new_atr_mask[np.where(atr_angles <= np.radians(gamma_atr))[0]]
        mtr = new_mtr_mask[np.where(mtr_angles <= np.radians(gamma_mtr))[0]]

        est_surfaces = np.copy(initial_guess)
        # est_surfaces[ep1] = 6
        # print("epi_ids", epi_ids)
        est_surfaces[epi_ids] = LV_SURFS.EPI
        est_surfaces[endo_ids] = LV_SURFS.ENDO
        est_surfaces[atr] = LV_SURFS.AORTIC
        est_surfaces[mtr] = LV_SURFS.MITRAL
        est_surfaces[its] = LV_SURFS.AM_INTERCECTION
        
        
        # adjust atrial region
        d_atr = np.linalg.norm(pts - c_atr, axis=1) 
        near_atr = np.where(d_atr <= r_atr * alpha_adj_atr)[0]
        # print("near_atr", near_atr)
        atr_pts = pts[near_atr]
        atr_vecs = c_atr - atr_pts
        atr_angles = np.zeros(len(atr_vecs))
        for i, (pt_normal, pt_vec) in enumerate(zip(surf_normals[near_atr], atr_vecs)):
            atr_angles[i] = angle_between(pt_vec, pt_normal, check_orientation=False)
        curr_vals = est_surfaces[near_atr]
        # print("curr_vals", curr_vals)
        corr_epi = near_atr[np.where(
            (atr_angles > np.radians(gamma_atr)) & ((curr_vals==LV_SURFS.AORTIC) | (curr_vals == LV_SURFS.ENDO)))[0]]
        # print("corr_epi", corr_epi)
        
        # epi_ids = np.union1d(epi_ids, corr_epi)
        # atr = np.setdiff1d(epi_ids, corr_epi)
        est_surfaces[corr_epi] = LV_SURFS.EPI
        
        # ................
        # 4 - Save data
        # save data on surface mesh
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS_EXPT_AB.value] = est_surfaces
        # set
        surf_to_global = lvsurf.point_data["vtkOriginalPointIds"]
        mesh_est_surfs = np.zeros(self.mesh.n_points)
        mesh_est_surfs[surf_to_global[epi_ids]] = LV_SURFS.EPI
        mesh_est_surfs[surf_to_global[endo_ids]] = LV_SURFS.ENDO
        mesh_est_surfs[surf_to_global[atr]] = LV_SURFS.AORTIC
        mesh_est_surfs[surf_to_global[mtr]] = LV_SURFS.MITRAL
        mesh_est_surfs[surf_to_global[its]] = LV_SURFS.AM_INTERCECTION
        mesh_est_surfs[surf_to_global[corr_epi]] = LV_SURFS.EPI
        
        
        self.mesh.point_data[LV_MESH_DATA.SURFS_EXPT_AB.value] = mesh_est_surfs

        # 4.1 - Merge with base and apex regions
        new_ab_surf_regions = np.zeros(len(pts))
        new_ab_surf_regions[epi_ids] = LV_SURFS.EPI
        new_ab_surf_regions[endo_ids] = LV_SURFS.ENDO
        new_ab_surf_regions[atr] = LV_SURFS.AORTIC
        new_ab_surf_regions[mtr] = LV_SURFS.MITRAL
        new_ab_surf_regions[its] = LV_SURFS.AM_INTERCECTION
        new_ab_surf_regions[corr_epi] = LV_SURFS.EPI
        
        val_ids = np.where((ab_surf_regions != 0) & (new_ab_surf_regions!=LV_SURFS.AORTIC) & (new_ab_surf_regions!=LV_SURFS.MITRAL) & (new_ab_surf_regions!=LV_SURFS.AM_INTERCECTION))[0]
        new_ab_surf_regions[val_ids] = ab_surf_regions[val_ids]       
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS.value] = new_ab_surf_regions

        new_ab_mesh_regions = np.zeros(self.mesh.n_points)
        new_ab_mesh_regions[surf_to_global[epi_ids]] = LV_SURFS.EPI
        new_ab_mesh_regions[surf_to_global[endo_ids]] = LV_SURFS.ENDO
        new_ab_mesh_regions[surf_to_global[atr]] = LV_SURFS.AORTIC
        new_ab_mesh_regions[surf_to_global[mtr]] = LV_SURFS.MITRAL
        new_ab_mesh_regions[surf_to_global[its]] = LV_SURFS.AM_INTERCECTION
        new_ab_mesh_regions[surf_to_global[corr_epi]] = LV_SURFS.EPI
        
        val_ids = np.where((ab_mesh_regions != 0) & (new_ab_mesh_regions!=LV_SURFS.AORTIC) & (new_ab_mesh_regions!=LV_SURFS.MITRAL) & (new_ab_mesh_regions!=LV_SURFS.AM_INTERCECTION))[0]
        new_ab_mesh_regions[val_ids] = ab_mesh_regions[val_ids] 
        self.mesh.point_data[LV_MESH_DATA.SURFS.value] = ab_mesh_regions
        
        # save nodesets info to geometry obj -> only save refereces from large surface
        # self._nodesets[LV_SURFS.EPI] = surf_to_global[epi_ids]
        self._nodesets[LV_SURFS.ENDO.name] = np.array(surf_to_global[endo_ids], dtype=np.int64)
        self._nodesets[LV_SURFS.AORTIC.name] = np.array(surf_to_global[atr], dtype=np.int64)
        self._nodesets[LV_SURFS.MITRAL.name] = np.array(surf_to_global[mtr], dtype=np.int64)
        self._nodesets[LV_SURFS.AM_INTERCECTION.name] = np.array(surf_to_global[its], dtype=np.int64)
        self._nodesets[LV_SURFS.EPI.name] = np.array(np.union1d(surf_to_global[epi_ids], surf_to_global[corr_epi]), dtype=np.int64) # adjust for corrected epi ids
        
        
        # !!! hardcoded -> surface of intereste will be endocardio
        self._surfaces_oi[LV_SURFS.ENDO.name] = self.cells(mask=surf_to_global[endo_ids], as_json_ready=True)
        
        
        # save virtual nodes (that not in mesh but are used in other computations)
        self._virtual_nodes[LV_SURFS.AORTIC.name] = np.array(c_atr, dtype=np.float64) # represent AORTIC central node (x,y,z)
        self._virtual_nodes[LV_SURFS.MITRAL.name] = np.array(c_mtr, dtype=np.float64)
                

        return (ab_surf_regions, ab_mesh_regions)
