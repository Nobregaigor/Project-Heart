from project_heart.modules.geometry import Geometry
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import *
from project_heart.utils.cloud_ops import *
from collections import deque

from project_heart.enums import *
from sklearn.cluster import KMeans

from functools import reduce

class LV_Geometry(Geometry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rot_chain = deque()
        self._normal = None

        self.aortic_info = {}
        self.mitral_info = {}

        self._aligment_data = {}
        
        # self._centroid = self.est_centroid()
        
    
    def est_centroid(self) -> np.ndarray:
        """Estimates the centroid of the geometry based on surface mesh.

        Returns:
            np.ndarray: [x,y,z] coordinates of center
        """
        lvsurf = self.get_surface_mesh()
        center = np.mean(lvsurf.points, axis=0)
        return center
    
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
        self._surface_mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value] = surf_regions

        # set global region ids (for entire mesh)
        surf_to_global = lvsurf.point_data["vtkOriginalPointIds"]
        global_regions = np.zeros(self.mesh.n_points)
        global_regions[surf_to_global[aligment_data["apex_region"]]
                       ] = LV_SURFS.APEX_REGION
        global_regions[surf_to_global[aligment_data["base_region"]]
                       ] = LV_SURFS.BASE_REGION
        self.mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value] = global_regions

        # save virtual nodes
        apex_pt = np.mean(pts[aligment_data["apex_region"]], axis=0)
        base_pt = np.mean(pts[aligment_data["base_region"]], axis=0)
        
        self.add_virtual_node(LV_VIRTUAL_NODES.APEX, apex_pt, True)
        self.add_virtual_node(LV_VIRTUAL_NODES.BASE, base_pt, True)
        self.set_normal(unit_vector(base_pt - apex_pt))

        return surf_regions, global_regions

    def identify_epi_endo_surfaces(self, threshold: float=90.0, ref_point: np.ndarray= None) -> tuple:
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
        self._surface_mesh.point_data[LV_MESH_DATA.EPI_ENDO_GUESS.value] = endo_epi_guess
        # convert local surf ids to global ids
        epi_ids_mesh = self.map_surf_ids_to_global_ids(epi_ids, dtype=np.int64)
        endo_ids_mesh = self.map_surf_ids_to_global_ids(endo_ids, dtype=np.int64)
        # set epi/endo ids at mesh (global ids)
        mesh_endo_epi_guess = np.zeros(self.mesh.n_points)
        mesh_endo_epi_guess[epi_ids_mesh] = LV_SURFS.EPI
        mesh_endo_epi_guess[endo_ids_mesh] = LV_SURFS.ENDO
        # save data at global mesh
        self.mesh.point_data[LV_MESH_DATA.EPI_ENDO_GUESS.value] = mesh_endo_epi_guess
       
        return endo_epi_guess, mesh_endo_epi_guess      

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
            wc = [m4,1.0-m4]
        else:
            label[klabels == 0] = LV_SURFS.MITRAL
            label[klabels == 1] = LV_SURFS.AORTIC
            wc = [1.0-m4,m4]
        # define clusters
        clustered = np.zeros(len(pts))
        clustered[ioi] = label
        
        # -------------------------------
        # Estimate aortic region
        # select aortic points
        atr_mask = np.where(clustered == LV_SURFS.AORTIC)[0]
        atr_pts = pts[atr_mask]
        # compute centers and radius
        c_atr = centroid(atr_pts) #np.mean(atr_pts, axis=0)
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
        c_mtr += r_mtr*m3 * np.cross(self.get_normal(), unit_vector(c_atr-c_mtr))
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
        r_mtr_border = np.mean(np.linalg.norm(mtr_border_pts - c_mtr_border, axis=1))
        
        # -------------------------------
        # compute intersection between mitral and aortic values
        its = np.intersect1d(atr, mtr)  # intersection
        
        # -------------------------------
        # Refine aortic region
        # select aortic pts, including those of intersection
        atr_its = np.union1d(atr, its)       
        atr_pts = pts[atr_its]
        c_atr = np.mean(atr_pts, axis=0) #centroid(atr_pts)         
        # compute distance from pts to aortic and mitral centers
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        # filter by radius
        atr = np.where(d_atr <= r_atr * (1.0+a2))[0]
        its = np.intersect1d(atr, mtr)  # intersection
        
        # -------------------------------
        # define endo_aortic and epi_aortic
        # compute angles between pts at aortic and its center
        atr_vecs_1 = c_atr - pts[atr]
        atr_angles = angle_between(surf_normals[atr], atr_vecs_1, check_orientation=False)
        # select endo and epi aortic ids based on angle thresholds
        endo_aortic = atr[np.where((atr_angles < np.radians(a4)))[0]]
        epi_aortic = atr[np.where((atr_angles > np.radians(a5)))[0]]
        
        # -------------------------------
        # define ids at aortic border
        # set endo and epi aortic ids as 'mask' values at lv surface mesh
        #   -> This step is performed so that we can use 'compute_derivative'
        endo_aortic_mask = np.zeros(len(pts))
        endo_aortic_mask[endo_aortic] = 1.0
        lvsurf.point_data[LV_MESH_DATA.ENDO_AORTIC_MASK.value] = endo_aortic_mask
        epi_aortic_mask = np.zeros(len(pts))
        epi_aortic_mask[epi_aortic] = 1.0
        lvsurf.point_data[LV_MESH_DATA.EPI_AORTIC_MASK.value] = epi_aortic_mask
        # select ids at the border of endo aortic mask using gradient method
        lvsurf = lvsurf.compute_derivative(LV_MESH_DATA.ENDO_AORTIC_MASK.value)
        lvsurf = lvsurf.compute_derivative("gradient")
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        ioi_atr_endo = np.where(grads_mag > 0)[0] 
        # select ids at the border of epi aortic mask using gradient method
        lvsurf = lvsurf.compute_derivative(LV_MESH_DATA.EPI_AORTIC_MASK.value)
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
        lvsurf.point_data[LV_MESH_DATA.BORDER_AORTIC_MASK.value] = border_aortic_mask
        # expand ids at aortic border with gradient method
        lvsurf = lvsurf.compute_derivative(LV_MESH_DATA.BORDER_AORTIC_MASK.value)
        lvsurf = lvsurf.compute_derivative("gradient")
        grads = lvsurf.get_array("gradient")
        grads_mag = np.linalg.norm(grads, axis=1)
        atr_border = np.where(grads_mag > 0)[0]
        # update LV_MESH_DATA aortic border info with latest values
        # -> this step is done just in case we need these values in another function
        border_aortic_mask = np.zeros(len(pts))
        border_aortic_mask[atr_border] = 1.0
        lvsurf.point_data[LV_MESH_DATA.BORDER_AORTIC_MASK.value] = epi_aortic_mask
        
        # -------------------------------
        # refine aortic at endocardio
        # select current pts at aortic border anc compute its center
        atr_border_pts = pts[atr_border]
        c_atr_border = centroid(atr_border_pts)
        r_atr_border = np.mean(np.linalg.norm(atr_border_pts - c_atr_border, axis=1))
        # c_atr_border = np.mean(atr_border_pts, axis=0)
        # select current pts at aortic border
        endo_aortic_pts = pts[endo_aortic]
        # compute distances between center of aortic border and endo aortic pts
        d_atr = np.linalg.norm(endo_aortic_pts - c_atr_border, axis=1)
        # filter by radius
        endo_aortic = endo_aortic[np.where(d_atr <= r_atr * (a3+1.0))[0]]
               
        # -------------------------------
        # set mask by layering values
        clustered = np.zeros(len(pts))
        clustered[epi_aortic] = LV_SURFS.EPI_AORTIC
        clustered[endo_aortic] = LV_SURFS.ENDO_AORTIC
        clustered[atr_border] = LV_SURFS.BORDER_AORTIC
        clustered[mtr] = LV_SURFS.MITRAL
        clustered[mtr_border] = LV_SURFS.BORDER_MITRAL
        clustered[its] = LV_SURFS.AM_INTERCECTION
        
        
        surf_to_global = np.array(lvsurf.point_data["vtkOriginalPointIds"], dtype=np.int64)
        
        # -------------------------------
        # transform ids from local surf values to global mesh ids
        mesh_clustered = np.zeros(self.mesh.n_points)
        mesh_clustered[surf_to_global] = clustered
        
        # -------------------------------
        # set atr and mitral as a mesh data
        atr = reduce(np.union1d, (endo_aortic, epi_aortic, atr_border))
        am_highlighted = np.zeros(len(pts))
        am_highlighted[atr] = LV_SURFS.AORTIC
        am_highlighted[mtr] = LV_SURFS.MITRAL
        am_highlighted[its] = LV_SURFS.AM_INTERCECTION
        mesh_am_highlighted = np.zeros(self.mesh.n_points)
        mesh_am_highlighted[surf_to_global] = am_highlighted
        

        # -------------------------------
        # save mesh data
        self._surface_mesh.point_data[LV_MESH_DATA.AM_DETAILED.value] = clustered
        self.mesh.point_data[LV_MESH_DATA.AM_DETAILED.value] = mesh_clustered
        
        self._surface_mesh.point_data[LV_MESH_DATA.AM_SURFS.value] = am_highlighted
        self.mesh.point_data[LV_MESH_DATA.AM_SURFS.value] = mesh_am_highlighted
        
        
        # -------------------------------
        # save virtual reference nodes
        self.add_virtual_node(LV_VIRTUAL_NODES.MITRAL, c_mtr, True)
        self.add_virtual_node(LV_VIRTUAL_NODES.AORTIC, c_atr, True)
        self.add_virtual_node(LV_VIRTUAL_NODES.AORTIC_BORDER, c_atr_border, True)       
        
        # -------------------------------
        # save aortic and mitral info
        
        
        self.aortic_info = {
            LV_AM_INFO.RADIUS.value: r_atr,
            LV_AM_INFO.CENTER.value: c_atr,
            LV_AM_INFO.SURF_IDS.value: atr, # ids at surface
            LV_AM_INFO.MESH_IDS.value: surf_to_global[atr], # ids at mesh
            
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
            LV_AM_INFO.BORDER_SURF_IDS.value: mtr_border, # ids at surface
            LV_AM_INFO.BORDER_MESH_IDS.value: surf_to_global[mtr_border] # ids at mesh     
        }
        
        return clustered, mesh_clustered
    
    def identify_surfaces(self, 
                          endo_epi_args={}, 
                          apex_base_args={}, 
                          aortic_mitral_args={},
                          create_nodesets=True,                          
                          ):
        
        endo_epi, mesh_endo_epi = self.identify_epi_endo_surfaces(**endo_epi_args)
        apex_base, mesh_apex_base = self.identify_base_and_apex_surfaces(**apex_base_args)
        aortic_mitral, mesh_aortic_mitral = self.identify_mitral_and_aortic_surfaces(**aortic_mitral_args)
        
        # To 'merge' result, we will overlay each info layer on top of each other
        # endo_epi will serve as backgroun  (will be lowest layer)
        # apex_base is the second merge   (will overwrite endo_epi)
        # aortic_mitral is the last merge (will be top-most layer and overwrite apex_base)
        
        # match indexes of interest at surface level
        ioi = np.where(apex_base!=LV_SURFS.OTHER.value)[0]
        endo_epi[ioi] = apex_base[ioi]
        ioi = np.where(aortic_mitral!=LV_SURFS.OTHER.value)[0]
        endo_epi[ioi] = aortic_mitral[ioi]
        
        # match indexes of interest at mesh (global) level
        ioi = np.where(mesh_apex_base!=LV_SURFS.OTHER.value)[0]
        mesh_endo_epi[ioi] = mesh_apex_base[ioi]
        ioi = np.where(mesh_aortic_mitral!=LV_SURFS.OTHER.value)[0]
        mesh_endo_epi[ioi] = mesh_aortic_mitral[ioi]
        
        # save results at surface and mesh levels
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS.value] = endo_epi.astype(np.int64)
        self.mesh.point_data[LV_MESH_DATA.SURFS.value] = mesh_endo_epi.astype(np.int64)
        
        # create nodesets
        if create_nodesets:
            self.create_nodesets_from_surfaces(mesh_data=LV_MESH_DATA.AM_SURFS.value, overwrite=True)
            self.create_nodesets_from_surfaces(mesh_data=LV_MESH_DATA.SURFS.value, overwrite=True)
        
    def create_nodesets_from_surfaces(self, 
                                      mesh_data=LV_MESH_DATA.SURFS.value,
                                      skip={},
                                      overwrite=False
                                      ):
        
        ids = self.mesh.point_data[mesh_data]
        for surf_enum in LV_SURFS:
            if surf_enum.name != "OTHER" and surf_enum.name not in skip:
                found_ids = np.where(ids==surf_enum.value)[0]
                if len(found_ids) > 0:
                    self.add_nodeset(surf_enum, found_ids, overwrite)
        
        


    def identify_surfaces2(self,
                          alpha_atr=0.5,  # coeff for radial distance computation
                          alpha_mtr=0.5,
                          beta_atr=0.9,  # coeff for second radial distance computation
                          #   beta2_atr=0.85,  # coeff for atr adjust
                          beta_mtr=0.9,
                          gamma_atr=89,
                          gamma_mtr=89,
                          gamma2_mtr=45,

                          phi_atr=80,
                          epi_angle=89,
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
            (ab_surf_regions, ab_mesh_regions) = self.identify_base_and_apex_regions(
                **kwargs)
            # ab_surf_regions = ab_surf_regions
            # ab_mesh_regions = ab_mesh_regions
        else:
            ab_surf_regions = self._surface_mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value]
            ab_mesh_regions = self.mesh.point_data[LV_MESH_DATA.APEX_BASE_REGION.value]
        # ensure that we are copying elements and not using original ones
        ab_surf_regions = np.copy(ab_surf_regions)
        ab_mesh_regions = np.copy(ab_mesh_regions)

        # extract surface mesh (use extract)
        lvsurf = self.get_surface_mesh()
        pts = lvsurf.points

        # ................
        # 1 - Start with an initial guess of endo-epi
        
        endo_epi_guess, mesh_endo_epi_guess = self.guess_epi_endo_based_on_surf_normals()

        # ................
        # 2 - Find initial guess of Mitral and Aortic clusters

        # compute gradients
        lvsurf.set_active_scalars(LV_MESH_DATA.EPI_ENDO_GUESS.value)
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

        # --> use mtr angles wrt normals as weights for next mtr center est.
        subcurr_mtr_mask = np.union1d(mtr_mask, its)
        mtr_pts = pts[subcurr_mtr_mask]
        mtr_vecs_1 = c_mtr - mtr_pts
        mtr_angles_1 = angle_between(
            surf_normals[subcurr_mtr_mask], mtr_vecs_1, check_orientation=False)**2

        # -- second pass --
        # Adjust mask: select new aortic and mitral pts
        new_atr_mask = np.union1d(atr_mask, its)
        new_mtr_mask = np.union1d(mtr_mask, its)
        atr_pts = pts[new_atr_mask]
        mtr_pts = pts[new_mtr_mask]
        # re-compute centers and radius
        c_atr = np.mean(atr_pts, axis=0)
        r_atr = np.mean(np.linalg.norm(atr_pts - c_atr, axis=1))
        c_mtr = np.average(mtr_pts, axis=0,
                           weights=1 - mtr_angles_1/mtr_angles_1.sum())
        r_mtr = np.mean(np.linalg.norm(mtr_pts - c_mtr, axis=1))
        # re-compute distance from pts to aortic and mitral centers
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        d_mtr = np.linalg.norm(pts - c_mtr, axis=1)
        # filter by radius; note that beta will reduce alpha
        atr = np.where(d_atr <= r_atr * (1.0+beta_atr))[0]
        mtr = np.where(d_mtr <= r_mtr * (1.0+beta_mtr))[0]

        # compute intersection between mitral and aortic values
        its = np.intersect1d(atr, mtr)  # intersection

        # compute final centers
        new_atr_mask = np.union1d(atr, its)
        new_mtr_mask = np.union1d(mtr, its)
        atr_pts = pts[new_atr_mask]
        mtr_pts = pts[new_mtr_mask]
        w_atr = np.zeros(len(pts))
        w_atr[atr] = 1.0
        w_atr[its] = 0.333
        w_atr = w_atr[w_atr > 0.0]
        w_atr /= w_atr.sum()

        # mtr_vecs_1 = c_mtr - mtr_pts
        # mtr_angles_1 = angle_between(
        # surf_normals[new_mtr_mask], mtr_vecs_1, check_orientation=False)**2
        w_mtr = np.zeros(len(pts))
        w_mtr[mtr] = 1.0
        w_mtr[its] = 0.4
        w_mtr = w_mtr[w_mtr > 0.0]
        # w_mtr += 0.5*(1 - mtr_angles_1/mtr_angles_1.sum())
        w_mtr /= w_mtr.sum()
        c_atr = np.average(atr_pts, axis=0, weights=w_atr)
        c_mtr = np.average(mtr_pts, axis=0, weights=w_mtr)

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

        # compute angles
        atr_angles = angle_between(
            surf_normals[new_atr_mask], atr_vecs, check_orientation=False)
        mtr_angles = angle_between(
            surf_normals[new_mtr_mask], mtr_vecs, check_orientation=False)

        # filter by angle
        atr = new_atr_mask[np.where(atr_angles <= np.radians(gamma_atr))[0]]
        mtr1 = new_mtr_mask[np.where(mtr_angles <= np.radians(gamma_mtr))[0]]
        mtr2 = new_mtr_mask[np.where(
            (mtr_angles <= np.radians(gamma2_mtr)))[0]]
        mtr = np.union1d(np.setdiff1d(mtr1, endo_ids), mtr2)

        mtr = np.setdiff1d(mtr, its)
        atr = np.setdiff1d(atr, its)

        # compute final centers
        new_atr_mask = np.union1d(atr, its)
        new_mtr_mask = np.union1d(mtr, its)
        atr_pts = pts[new_atr_mask]
        mtr_pts = pts[new_mtr_mask]
        w_atr = np.zeros(len(pts))
        w_atr[atr] = 1.0
        w_atr[its] = 0.333
        w_atr = w_atr[w_atr > 0.0]
        w_atr /= w_atr.sum()

        # mtr_vecs_1 = c_mtr - mtr_pts
        # mtr_angles_1 = angle_between(
        # surf_normals[new_mtr_mask], mtr_vecs_1, check_orientation=False)**2
        w_mtr = np.zeros(len(pts))
        w_mtr[mtr] = 1.0
        w_mtr[its] = 0.2
        w_mtr = w_mtr[w_mtr > 0.0]
        # w_mtr += 0.5*(1 - mtr_angles_1/mtr_angles_1.sum())
        w_mtr /= w_mtr.sum()
        c_atr = np.average(atr_pts, axis=0, weights=w_atr)
        c_mtr = np.average(mtr_pts, axis=0, weights=w_mtr)

        est_surfaces = np.copy(initial_guess)
        est_surfaces[epi_ids] = LV_SURFS.EPI
        est_surfaces[endo_ids] = LV_SURFS.ENDO
        est_surfaces[atr] = LV_SURFS.AORTIC
        est_surfaces[mtr] = LV_SURFS.MITRAL
        est_surfaces[its] = LV_SURFS.AM_INTERCECTION

        # ---
        # adjust aortic region
        d_atr = np.linalg.norm(pts - c_atr, axis=1)
        near_atr = np.where(d_atr <= r_atr * alpha_adj_atr)[0]
        # print("near_atr", near_atr)
        atr_pts = pts[near_atr]
        atr_vecs = c_atr - atr_pts
        # atr_angles = np.zeros(len(atr_vecs))
        atr_angles = angle_between(
            surf_normals[near_atr], atr_vecs, check_orientation=False)
        curr_vals = est_surfaces[near_atr]
        corr_epi = near_atr[np.where(
            (atr_angles > np.radians(phi_atr)) & ((curr_vals == LV_SURFS.AORTIC) | (curr_vals == LV_SURFS.ENDO)))[0]]

        # print("corr_epi", len(corr_epi))
        corr_epi_pts = pts[corr_epi]
        corr_epi_vecs = c_atr - corr_epi_pts
        corr_epi_angles = angle_between(
            surf_normals[corr_epi], corr_epi_vecs, check_orientation=False)
        corr_epi = corr_epi[np.where(
            corr_epi_angles > np.radians(epi_angle))[0]]
        # print("corr_epi", len(corr_epi))

        # epi_ids = np.union1d(epi_ids, corr_epi)
        # atr = np.setdiff1d(epi_ids, corr_epi)
        est_surfaces[corr_epi] = LV_SURFS.EPI

        # ................
        # 4 - Save data

        # 4.1 - Save data on surface mesh
        # save current est. without apex and base info on surface mesh
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS_EXPT_AB.value] = est_surfaces

        # merge surface apex and base info and save on surface mesh
        est_surfaces_ab = np.copy(est_surfaces)
        val_ids = np.where((ab_surf_regions != 0) & (est_surfaces_ab != LV_SURFS.AORTIC) & (
            est_surfaces_ab != LV_SURFS.MITRAL) & (est_surfaces_ab != LV_SURFS.AM_INTERCECTION))[0]
        est_surfaces_ab[val_ids] = ab_surf_regions[val_ids]
        self._surface_mesh.point_data[LV_MESH_DATA.SURFS.value] = est_surfaces_ab

        # 4.2 - Save data on global mesh
        # match surface idxs with global idxs
        surf_to_global = lvsurf.point_data["vtkOriginalPointIds"]
        epi_ids_mesh = surf_to_global[epi_ids]
        endo_ids_mesh = surf_to_global[endo_ids]
        atr_mesh = surf_to_global[atr]
        mtr_mesh = surf_to_global[mtr]
        its_mesh = surf_to_global[its]
        corr_epi_mesh = surf_to_global[corr_epi]

        mesh_est_surfs = np.zeros(self.mesh.n_points)
        mesh_est_surfs[epi_ids_mesh] = LV_SURFS.EPI
        mesh_est_surfs[endo_ids_mesh] = LV_SURFS.ENDO
        mesh_est_surfs[atr_mesh] = LV_SURFS.AORTIC
        mesh_est_surfs[mtr_mesh] = LV_SURFS.MITRAL
        mesh_est_surfs[its_mesh] = LV_SURFS.AM_INTERCECTION
        mesh_est_surfs[corr_epi_mesh] = LV_SURFS.EPI

        # save current est. without apex and base info on global mesh
        self.mesh.point_data[LV_MESH_DATA.SURFS_EXPT_AB.value] = mesh_est_surfs
        # merge global apex and base info and save on global mesh
        mesh_est_surfs_ab = np.copy(mesh_est_surfs)
        val_ids = np.where((ab_mesh_regions != 0) & (mesh_est_surfs_ab != LV_SURFS.AORTIC) & (
            mesh_est_surfs_ab != LV_SURFS.MITRAL) & (mesh_est_surfs_ab != LV_SURFS.AM_INTERCECTION))[0]
        mesh_est_surfs_ab[val_ids] = ab_mesh_regions[val_ids]
        self.mesh.point_data[LV_MESH_DATA.SURFS.value] = mesh_est_surfs_ab

        # 4.3 - Save data on local nodesets variable
        # save nodesets info to geometry obj -> only save refereces from large surface
        # self._nodesets[LV_SURFS.EPI] = surf_to_global[epi_ids]
        self._nodesets[LV_SURFS.ENDO.name] = np.array(
            endo_ids_mesh, dtype=np.int64)
        self._nodesets[LV_SURFS.AORTIC.name] = np.array(
            atr_mesh, dtype=np.int64)
        self._nodesets[LV_SURFS.MITRAL.name] = np.array(
            mtr_mesh, dtype=np.int64)
        self._nodesets[LV_SURFS.AM_INTERCECTION.name] = np.array(
            its_mesh, dtype=np.int64)
        self._nodesets[LV_SURFS.EPI.name] = np.array(np.union1d(
            epi_ids_mesh, corr_epi_mesh), dtype=np.int64)  # adjust for corrected epi ids

        # !!! hardcoded -> surface of intereste will be endocardio
        self._surfaces_oi[LV_SURFS.ENDO.name] = self.cells(
            mask=epi_ids_mesh, as_json_ready=True)

        # save virtual nodes (that not in mesh but are used in other computations)
        self._virtual_nodes[LV_SURFS.AORTIC.name] = np.array(
            c_atr, dtype=np.float64)  # represent AORTIC central node (x,y,z)
        self._virtual_nodes[LV_SURFS.MITRAL.name] = np.array(
            c_mtr, dtype=np.float64)

        self.aortic_info = {
            "R": np.mean(np.linalg.norm(pts[atr] - c_atr, axis=1)),
            "C": np.array(c_atr, dtype=np.float64),
        }
        self.mitral_info = {
            "R": np.mean(np.linalg.norm(pts[mtr] - c_mtr, axis=1)),
            "C": np.array(c_mtr, dtype=np.float64),
        }

        return (ab_surf_regions, ab_mesh_regions)

    # =============================================================================
    # Boundary conditions

    @staticmethod
    def get_springs_pts_for_plot(
            geo_pts: np.ndarray,
            rim_pts: np.ndarray,
            relations: np.ndarray,
            n_skip: int = 1):
        pts_a = geo_pts[relations[:, 0]][::n_skip]
        pts_b = rim_pts[relations[:, 1]][::n_skip]
        lines = None
        for a, b in zip(pts_a, pts_b):
            if lines is None:
                lines = lines_from_points(np.array([a, b]))
            else:
                lines = lines.merge(lines_from_points(np.array([a, b])))
        return lines

    def create_spring_rim_bc(self,
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
            surface = surface.value
        
        #  -- possible mitral BCs --
        if surface == LV_SURFS.MITRAL.value:
            if len(self.mitral_info) == 0:
                raise RuntimeError("Mitral info not found. Did you identify LV surfaces?")
            c = self.mitral_info[LV_AM_INFO.CENTER.value]
            r = self.mitral_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.BORDER_MITRAL.value:
            if len(self.mitral_info) == 0:
                raise RuntimeError("Mitral info not found. Did you identify LV surfaces?")
            c = self.mitral_info[LV_AM_INFO.BORDER_CENTER.value]
            r = self.mitral_info[LV_AM_INFO.BORDER_RADIUS.value]
        # -- possible aortic BCs --
        elif surface == LV_SURFS.AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError("Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.CENTER.value]
            r = self.aortic_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.ENDO_AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError("Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.CENTER.value]
            r = self.aortic_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.EPI_AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError("Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.CENTER.value]
            r = self.aortic_info[LV_AM_INFO.RADIUS.value]
        elif surface == LV_SURFS.BORDER_AORTIC.value:
            if len(self.aortic_info) == 0:
                raise RuntimeError("Aortic info not found. Did you identify LV surfaces?")
            c = self.aortic_info[LV_AM_INFO.BORDER_CENTER.value]
            r = self.aortic_info[LV_AM_INFO.BORDER_RADIUS.value]
        else:
            raise ValueError("Surface '{}' not valid or not yet implemented \
                for this boundary condition.".format(LV_SURFS(surface).name))
        
        # select pts at surface
        pts = self.points(mask=self.get_nodeset(surface))
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
        
        rim_data = {
                LV_RIM.NODES.value: rim,
                LV_RIM.CENTER.value: rim_center,
                LV_RIM.ELEMENTS.value: rim_el,
                LV_RIM.RELATIONS.value: nodes_rim_relations,
                LV_RIM.DISTS.value: nodes_rim_dists,
                LV_RIM.REF_NODESET.value: surface
            }
                
        return rim_data

    