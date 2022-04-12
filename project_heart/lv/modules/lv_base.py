from project_heart.modules.container import BaseContainerHandler
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


class LV_Base(BaseContainerHandler):
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

    def est_pts_aligment_with_lv_normal(self, points, n=5, rot_chain=[], **kwargs):
        pts = np.copy(points)
        for _ in range(n):
            long_line, _ = LV_Base.est_apex_and_base_refs(
                pts, **kwargs)
            lv_normal = unit_vector(long_line[0] - long_line[1])
            curr_rot = get_rotation(lv_normal, self._Z)
            pts = curr_rot.apply(pts)
            rot_chain.append(curr_rot)
        long_line, info = LV_Base.est_apex_and_base_refs(
            pts, **kwargs)
        lv_normal = unit_vector(long_line[0] - long_line[1])
        info["rot_pts"] = pts
        info["normal"] = lv_normal
        info["long_line"] = long_line
        info["rot_chain"] = rot_chain
        self._aligment_data = info
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
                try:
                    self.identify_base_and_apex_surfaces()
                except:
                    raise RuntimeError(
                        """Unable to compute normal. Prooced with another method\
                           See 'identify_base_and_apex_surfaces' and 'set_normal'\
                           for details.
                        """)
