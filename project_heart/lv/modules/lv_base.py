from project_heart.modules.container import BaseContainerHandler
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import create_rim_circunference, lines_from_points
from project_heart.utils.cloud_ops import *
from collections import deque

from project_heart.enums import *
from sklearn.cluster import KMeans

from functools import reduce

from pathlib import Path
import os

from project_heart.enums import LV_STATES, LV_SURFS, LV_VIRTUAL_NODES

default_lv_enums = {
    "STATES": LV_STATES,
    "REGIONS": LV_SURFS,
    "VIRTUAL_NODES": LV_VIRTUAL_NODES
}

import logging

logging.basicConfig()
logger = logging.getLogger('LV')

class LV_Base(BaseContainerHandler):
    def __init__(self, enums={}, *args, **kwargs):
        super(LV_Base, self).__init__(*args, **kwargs)

        self._rot_chain = deque()
        self._normal = None

        self.aortic_info = {}
        self.mitral_info = {}
        self.base_info = {}

        self._aligment_data = {}
        
        self._apex_ql = 0.05
        self._base_qh = 0.95

        self._long_line = None

        # ---- Enums
        self.REGIONS = LV_SURFS
        self.STATES = LV_STATES
        self.VIRTUAL_NODES = LV_VIRTUAL_NODES
        # overwrite enums if 'enums' dict is provided
        if len(enums) > 0:
            self.config_enums(enums, check_keys=default_lv_enums.keys())

        # self._centroid = self.est_centroid()
        # ------ Flags
        self._surfaces_identified_with_class_method = False

    # ----------------------------------------------------------------
    #  Estimation of apex and base methods
    # ----------------------------------------------------------------
    # These methods assume that there is no previous information available
    # for the current LV instance

    def est_centroid(self) -> np.ndarray:
        """Estimates the centroid of the geometry based on surface mesh.

        Returns:
            np.ndarray: [x,y,z] coordinates of center
        """
        lvsurf = self.get_surface_mesh()
        # center = np.mean(lvsurf.points, axis=0)
        return centroid(lvsurf.points)  # center

    @staticmethod
    def est_apex_ref(points, ql=0.05, **kwargs):
        # print("apex:", ql)
        zvalues = points[:, 2]
        # zmin = np.min(zvalues)
        zmin = np.min(zvalues)
        zmax = np.max(zvalues)
        thresh = np.interp(ql, [0, 1], [zmin, zmax])
        # if zmin > 0:
        #     thresh = zmin*(1+ql)
        #     # apex_region_idxs = np.where(zvalues <= thresh)[0]
        # else:
        #     thresh = zmin*(1-ql)
        apex_region_idxs = np.where(zvalues <= thresh)[0]
        # thresh = np.quantile(zvalues, ql)
        # apex_region_idxs = np.where(zvalues <= thresh)[0]
        apex_region_pts = points[apex_region_idxs]
        # return np.mean(apex_region_pts, 0), apex_region_idxs
        return centroid(apex_region_pts), apex_region_idxs
        
    @staticmethod
    def est_base_ref(points, qh=0.95, **kwargs):
        # print("base:", qh)
        zvalues = points[:, 2]
        zmin = np.min(zvalues)
        zmax = np.max(zvalues)
        thresh = np.interp(qh, [0, 1], [zmin, zmax])
        # if zmax > 0:
        #     base_region_idxs = np.where(zvalues >= thresh)[0]
        # else:
            # base_region_idxs = np.where(zvalues <= thresh)[0]
        # thresh = np.quantile(zvalues, qh)
        base_region_idxs = np.where(zvalues >= thresh)[0]
        base_region_pts = points[base_region_idxs]
        # print(len(base_region_pts))
        return centroid(base_region_pts), base_region_idxs

    @staticmethod
    def est_apex_and_base_refs(points, **kwargs):
        apex_ref, apex_region_idxs = LV_Base.est_apex_ref(
            points, **kwargs)
        base_ref, base_region_idxs = LV_Base.est_base_ref(
            points, **kwargs)
        info = {
            "apex_ref": apex_ref,
            "base_ref": base_ref,
            "apex_region": apex_region_idxs,
            "base_region": base_region_idxs}
        return np.vstack((base_ref, apex_ref)), info

    def est_apex_and_base_refs_iteratively(self, points, 
                                           n=5, rot_chain=None, 
                                           ql=None, qh=None,
                                           **kwargs):
        if rot_chain is None:
            rot_chain = deque()
        pts = np.copy(points)
        if ql is None:
            ql = self._apex_ql
        else:
            self._apex_ql = ql
        if qh is None:
            qh = self._base_qh
        else:
            self._base_qh = qh
        # check minimum n
        if n <= 0:
            n = 1
        # estimate pts aligment
        for _ in range(n):
            long_line, info = LV_Base.est_apex_and_base_refs(
                pts, ql=ql, qh=qh, **kwargs)
            lv_normal = unit_vector(long_line[0] - long_line[1])
            curr_rot = get_rotation(lv_normal, self._Z)
            pts = curr_rot.apply(pts)
            rot_chain.append(curr_rot)
            
        # we now have the data after aligment estimatiom.
        # we need to return info at initial state:
        pts = np.copy(points)
        apex_pt = centroid(pts[info["apex_region"]])
        base_pt = centroid(pts[info["base_region"]])
        long_line = np.vstack((base_pt, apex_pt))
        lv_normal = unit_vector(long_line[1] - long_line[0])
        info["normal"] = lv_normal
        info["long_line"] = long_line
        info["rot_chain"] = rot_chain
        info["apex"] = apex_pt
        info["base"] = base_pt

        return info
    
    def est_pts_aligment_with_lv_normal(self, points, 
                                        n=5, rot_chain=None, 
                                        ql=None, qh=None,
                                        **kwargs):
        if rot_chain is None:
            rot_chain = deque()
        pts = np.copy(points)
        if ql is None:
            ql = self._apex_ql
        else:
            self._apex_ql = ql
        if qh is None:
            qh = self._base_qh
        else:
            self._base_qh = ql
        for _ in range(n):
            long_line, _ = LV_Base.est_apex_and_base_refs(
                pts, ql=ql, qh=qh, **kwargs)
            lv_normal = unit_vector(long_line[0] - long_line[1])
            curr_rot = get_rotation(lv_normal, self._Z)
            pts = curr_rot.apply(pts)
            rot_chain.append(curr_rot)
        long_line, info = LV_Base.est_apex_and_base_refs(
            pts, ql=ql, qh=qh, **kwargs)
        lv_normal = unit_vector(long_line[0] - long_line[1])
        info["rot_pts"] = pts
        info["normal"] = lv_normal
        info["long_line"] = long_line
        info["rot_chain"] = rot_chain
        self._aligment_data = info
        return info

    # ----------------------------------------------------------------
    # Computation of apex and base methods
    # ----------------------------------------------------------------
    # These methods now assume that some information is available

    def compute_base_from_nodeset(self, base_nodeset:str=None) -> np.ndarray:
        """Computes the base virtual node as the centroid of provided nodeset.
           If no dataset is provided, it defaults to REGIONS.BASE_REGION. 
           
           Note: if enough information is provided, for instance BASE_BORDER
           or ENDO_BASE_BORDER, it is recommended to use these regions
           for base computation. Default value will not use these.

        Args:
            base_nodeset (str or Enum, optional): Nodeset to mask nodes. Defaults to None.

        Returns:
            np.ndarray: Apex virtual node
        """
        if base_nodeset is None:
            base_nodeset = self.REGIONS.BASE_REGION
        ref = centroid(self.nodes(mask=self.get_nodeset(base_nodeset)))
        self.add_virtual_node(self.VIRTUAL_NODES.BASE, ref, True)
        return self.get_virtual_node(self.VIRTUAL_NODES.BASE)

    def compute_apex_from_nodeset(self, apex_nodeset:str=None) -> np.ndarray:
        """Computes the apex virtual node as the centroid of provided nodeset. \n
           If no dataset is provided, it defaults to REGIONS.APEX_REGION

        Args:
            nodeset (str or Enum, optional): Nodeset to mask nodes. Defaults to None.

        Returns:
            np.ndarray: Apex virtual node
        """
        if apex_nodeset is None:
            apex_nodeset = self.REGIONS.APEX_REGION

        ref = centroid(self.nodes(mask=self.get_nodeset(apex_nodeset)))
        self.add_virtual_node(self.VIRTUAL_NODES.APEX, ref, True)
        return self.get_virtual_node(self.VIRTUAL_NODES.APEX)

    def compute_apex_from_base_vn(self, 
            d=5, 
            nodeset=None, 
            base_ref=None,
            log_level=logging.INFO
        ) -> np.ndarray:
        """Computes apex region based on node with longest distance form reference base and
        distance 'd' from such node. If nodeset is provided, will only use nodes from such nodeset.
        If reference base node is not provided, will try to use existing base virtual node.
        If requested, information will be saved as nodeset. 

        Args:
            d (int, optional): distance threshold from found apex id. Defaults to 5.
            nodeset (_type_, optional): nodeset in which search is performed. Defaults to None (uses entire mesh).
            base_ref ((str or enum) or (list, tuple or np.ndarray), optional): virtual reference node. Defaults to None.
            add_as_nodeset (bool, optional): _description_. Defaults to True.
            overwrite_nodeset (bool, optional): _description_. Defaults to True.

        Returns:
            np.ndarray: Region array
        """
        log = logger.getChild("compute_apex_from_base_vn")
        log.setLevel(log_level)

        
        if base_ref is None:
            base_ref = self.get_virtual_node(self.VIRTUAL_NODES.BASE)
        else:
            if isinstance(base_ref, (str, Enum)):
                base_ref = self.get_virtual_node(self.VIRTUAL_NODES.BASE)
                
            assert isinstance(base_ref, (list, tuple, np.ndarray)), (
                "base_ref must be list, tuple or np.ndarray representing [x,y,z] of reference node.")
            if not isinstance(base_ref, np.ndarray):
                base_ref = np.array(base_ref, dtype=np.float64)
            
            shape = base_ref.shape
            assert len(shape) == 1 and shape[0] == (3), (
                "base_ref must have shape of [3], representing [x,y,z] of reference node.")
        log.debug("using base_ref={}".format(base_ref))
        
        # get nodes
        if nodeset is None:
            log.debug("Using all nodes (nodeset is None)")
        else:
            log.debug("Using nodes from nodeset: {}".format(nodeset))
        mask = self.get_nodeset(nodeset) if nodeset is not None else None
        nodes = self.nodes(mask=mask)

        from project_heart.utils.cloud_ops import relate_closest
        _, dists = relate_closest(nodes, [base_ref])
        apex_id = np.argmax(dists)
        apex_pt = nodes[apex_id]
        log.debug("Apex id: {} -> {}".format(nodeset, apex_pt))

        _, dist_to_apex =  relate_closest(nodes, [apex_pt])
        apex_region_ids = np.where(dist_to_apex < d)
        log.debug("Number of nodes close to apex: {}".format(len(apex_region_ids)))

        if mask is not None:
            apex_region_ids = mask[apex_region_ids]
        
        ref = centroid(self.nodes(mask=apex_region_ids))
        log.debug("Apex: {}".format(ref))

        self.add_virtual_node(self.VIRTUAL_NODES.APEX, ref, True)
        return self.get_virtual_node(self.VIRTUAL_NODES.APEX), apex_region_ids

    def compute_apex_and_base_ref_from_nodesets(self, 
            apex_nodeset:str=None, 
            base_nodeset:str=None) -> dict:
        """Computes apex and base virtual nodes from provided nodesets. If no nodeset is provided for
        either region, default values are used (check individual methods for details). For consistency
        This method also computes longitudinal line and LV normal based on apex and base virtual nodes.

        Apex default nodeset: REGIONS.APEX_REGION
        Base default nodeset: REGIONS.BASE_REGION

        Args:
            apex_nodeset (str or Enum, optional): Nodeset used to mask nodes for apex. Defaults to None.
            base_nodeset (str or Enum, optional): Nodeset used to mask nodes for base. Defaults to None.

        Returns:
            dict: keys -> apex_region, base_region, long_line and normal.
        """
        # compute apex and base
        apex = self.compute_apex_from_nodeset(apex_nodeset)
        base = self.compute_base_from_nodeset(base_nodeset)
        # compute longitudinal axis and LV normal vector
        long_line = self.compute_long_line(apex, base)
        normal = self.compute_normal()
        # gather info (for consistency with previous code)
        info = {
                "apex_region": self.get_nodeset(apex_nodeset),
                "base_region": self.get_nodeset(base_region),
                "long_line": long_line,
                "normal": normal
            }
        return info



    # ----------------------------------------------------------------
    # Aortic and Mitral identification/set functions

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

    def set_base_info(self, base_id=LV_SURFS.BASE) -> None:
        """Sets base information for other computations, such as boundary conditions.\
           Creates 'base_info' dictionary with radius, center, and mesh ids.\
            If 'add_virtual_nodes' is set to true, it will create virtual node information\
            at center.

        Args:
            base_id (int, str or Enum, optional): identification of base nodeset. \
                Defaults to LV_SURFS.base.
            add_virtual_nodes (bool, optional): Whether to create virtual node information.
        """
        if base_id is not None:
            try:
                ioi = self.get_nodeset(base_id)
            except KeyError:
                raise KeyError("base_id not found within nodesets. Did you create it? \
                    You can use the function 'identify_surfaces' to automatically create it.")
            pts = self.nodes(ioi)
            center = centroid(pts)
            r = radius(pts, center)
            self.base_info = {
                LV_BASE_INFO.RADIUS.value: r,
                LV_BASE_INFO.CENTER.value: center,
                LV_BASE_INFO.SURF_IDS.value: None,  # ids at surface
                LV_BASE_INFO.MESH_IDS.value: ioi,  # ids at mesh
            }

    def create_nodesets_from_regions(self,
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

    
    # =============================================================================
    # Check methods

    def check_surf_initialization(self):
        return self._surfaces_identified_with_class_method

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
        elif surface == LV_SURFS.BASE.value:
            if len(self.base_info) == 0:
                raise RuntimeError(
                    "Base info not found. Did you identify LV surfaces?")
            c = self.base_info[LV_BASE_INFO.CENTER.value]
            r = self.base_info[LV_BASE_INFO.RADIUS.value]
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

    # =================================================================
    # Other

    def compute_normal(self):
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
                raise RuntimeError(
                    """Unable to compute normal. Are you explictly using LV_Base? Trye using a higher-level class instead (see docs), prooced with another method or manually set normal with: 'set_normal'.""")

    def set_long_line(self, line, dtype=np.float64) -> None:
        self._long_line = np.array(line, dtype=dtype)

    def compute_long_line(self, apex=None, base=None, **kwargs) -> None:
        if apex is not None and base is not None:
            self.set_long_line([apex, base])
        else:
            try:
                apex = self.get_virtual_node(LV_VIRTUAL_NODES.APEX)
                base = self.get_virtual_node(LV_VIRTUAL_NODES.BASE)
                self.set_long_line([apex, base])
            except:
                try:
                    self.compute_normal(**kwargs)
                    apex = self.get_virtual_node(LV_VIRTUAL_NODES.APEX)
                    base = self.get_virtual_node(LV_VIRTUAL_NODES.BASE)
                    self.set_long_line([apex, base])
                except:
                    raise RuntimeError(
                        "Unable to compute longitudinal line. Please, try setting it manually.")

    def get_long_line(self) -> np.ndarray:
        if self._long_line is None:
            self.compute_long_line()

        return self._long_line
