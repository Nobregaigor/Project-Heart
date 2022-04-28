from project_heart.modules.container import BaseContainerHandler
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import *
from project_heart.utils.cloud_ops import *
from .lv_region_identifier import LV_RegionIdentifier

from collections import deque

from project_heart.enums import *
from sklearn.cluster import KMeans

from functools import reduce

from pathlib import Path
import os

import logging
logger = logging.getLogger('LV.FiberEstimator')

from project_heart.enums import LV_FIBER_MODES, LV_FIBERS

default_lv_enums = {
    "FIBER_MODES": LV_FIBER_MODES,
    "FIBERS": LV_FIBERS,
}

class LV_FiberEstimator(LV_RegionIdentifier):
    def __init__(self, log_level=logging.INFO, enums={}, *args, **kwargs):
        super(LV_FiberEstimator, self).__init__(log_level=log_level, *args, **kwargs)

        logger.setLevel(log_level)
        # ------ default values
        self._default_fiber_markers = {
            "epi": LV_SURFS.EPI.value,
            "lv": LV_SURFS.ENDO.value,
            "base": LV_SURFS.BASE.value
        }
        
        self.FIBER_MODES = LV_FIBER_MODES
        self.FIBERS = LV_FIBERS
        # overwrite enums if 'enums' dict is provided
        if len(enums) > 0:
            self.config_enums(enums, check_keys=default_lv_enums.keys())


    def identify_fibers_regions_ldrb_1(self) -> tuple:
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
        ldrb_mesh_region_ids[mitral] = LV_SURFS.BASE
        ldrb_mesh_region_ids[border_aortic] = LV_SURFS.BASE
        # get ids for surface
        lvsurf_map_id = self.get_surface_id_map_from_mesh()
        lvsurf = self.get_surface_mesh()
        # save data at surface and mesh level
        lvsurf[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids[lvsurf_map_id]
        self.mesh[LV_MESH_DATA.LDRB_1.value] = ldrb_mesh_region_ids

        return ldrb_mesh_region_ids, ldrb_mesh_region_ids

    def identify_fibers_regions_ldrb_2(self) -> tuple:
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
        ldrb_mesh_region_ids[mitral] = LV_SURFS.BASE
        ldrb_mesh_region_ids[border_aortic] = LV_SURFS.BASE
        # get ids for surface
        lvsurf_map_id = self.get_surface_id_map_from_mesh()
        lvsurf = self.get_surface_mesh()
        # save data at surface and mesh level
        lvsurf[LV_MESH_DATA.LDRB_2.value] = ldrb_mesh_region_ids[lvsurf_map_id]
        self.mesh[LV_MESH_DATA.LDRB_2.value] = ldrb_mesh_region_ids

        return ldrb_mesh_region_ids, ldrb_mesh_region_ids

    def identify_fibers_regions_ldrb_3(self) -> tuple:
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
        ldrb_mesh_region_ids[mitral] = LV_SURFS.BASE
        ldrb_mesh_region_ids[border_aortic] = LV_SURFS.BASE
        # get ids for surface
        lvsurf_map_id = self.get_surface_id_map_from_mesh()
        lvsurf = self.get_surface_mesh()
        # save data at surface and mesh level
        lvsurf[LV_MESH_DATA.LDRB_3.value] = ldrb_mesh_region_ids[lvsurf_map_id]
        self.mesh[LV_MESH_DATA.LDRB_3.value] = ldrb_mesh_region_ids

        return ldrb_mesh_region_ids, ldrb_mesh_region_ids

    def identify_fibers_regions_ldrb(self, mode: str) -> tuple:
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
            return self.identify_fibers_regions_ldrb_1()
        elif mode == LV_FIBER_MODES.LDRB_2.value:
            return self.identify_fibers_regions_ldrb_2()
        elif mode == LV_FIBER_MODES.LDRB_3.value:
            return self.identify_fibers_regions_ldrb_3()
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
                       surf_region_key: str,
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
            surf_region_key (str): _description_
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
        # check if given surf_region_key is Enum
        surf_region_key = self.check_enum(surf_region_key)
        # check for mesh types (required pure tetrahedral mesh)
        assert(self.check_tet4_mesh()
               ), "Mesh must be composed of pure tetrahedrons."
        assert(self.check_tri3_surfmesh()
               ), "Surface mesh must be composed of pure triangles."
        # check for surface region ids (values for each node in surface describing region id)
        (in_mesh, in_surf_mesh) = self.check_mesh_data(surf_region_key)
        if not in_surf_mesh:
            found_ids = False
            try:
                self.identify_fibers_regions_ldrb(surf_region_key)
                found_ids = True
            except NotImplementedError:
                raise ValueError("Surface regions ids '{}' not found in mesh data. "
                    "Tried to compute using 'identify_fibers_regions_ldrb', but method "
                    "was not implemented. Please, check surf_region_key mesh data key.")
            if not found_ids: # check what type of error message to return
                if self.check_surf_initialization():
                    raise ValueError("Surface regions ids '{}' is not initialized within internal algorithm. "
                        "Did you add to surface mesh data?".format(surf_region_key))
                else:
                    raise ValueError("Surface regions ids '{}' not found in mesh data. "
                        "Did you identify surfaces? See 'LV.identify_surfaces'.")
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
        # cellregionIdsSurf = self.transform_point_data_to_cell_data(
        #     surf_region_key, method="median", surface=True)
        try:
            cellregionIdsSurf = self.get_facet_data(surf_region_key)
        except KeyError:
            try:
                cellregionIdsSurf = self.transform_region_to_facet_data(
                    surf_region_key)
            except:
                raise RuntimeError(
                    "Could not get facet data for region %s" % surf_region_key)

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

    def compute_fiber_angles(self, cell_data: bool = False):

        # ensure that normal was initialized
        try:
            self.get_normal()
        except:
            self.compute_normal()

        # prooced
        if not cell_data:
            fiber_pts_vec = self.mesh.point_data[LV_FIBERS.F0.value]
            sheet_pts_vec = self.mesh.point_data[LV_FIBERS.S0.value]
            sheet_normal_pts_vec = self.mesh.point_data[LV_FIBERS.N0.value]
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
        else:
            fiber_pts_vec = self.mesh.cell_data[LV_FIBERS.F0.value]
            sheet_pts_vec = self.mesh.cell_data[LV_FIBERS.S0.value]
            sheet_normal_pts_vec = self.mesh.cell_data[LV_FIBERS.N0.value]
            # Compute angles between normal and fiber vectors
            fiber_angles = np.degrees(self.compute_angles_wrt_normal(
                fiber_pts_vec, False, False) - np.pi*0.5)
            sheet_angles = np.degrees(self.compute_angles_wrt_normal(
                sheet_pts_vec, False, False) - np.pi*0.5)
            sheet_normal_angles = np.degrees(self.compute_angles_wrt_normal(
                sheet_normal_pts_vec, False, False) - np.pi*0.5)
            # add angles
            self.mesh.cell_data[LV_FIBERS.F0_ANGLES.value] = fiber_angles
            self.mesh.cell_data[LV_FIBERS.S0_ANGLES.value] = sheet_angles
            self.mesh.cell_data[LV_FIBERS.N0_ANGLES.value] = sheet_normal_angles

    def regress_fibers(self, other_LV,
                       container_loc=GEO_DATA.MESH_POINT_DATA,
                       compute_angles=True,
                       convert_to_cell_data=True,
                       **kwargs):
        if not issubclass(other_LV.__class__, BaseContainerHandler):
            raise ValueError(
                "Other LV object must be subclass of BaseContainerHandler.")

        # get list of fibers to regress
        to_regress = [LV_FIBERS.F0, LV_FIBERS.S0, LV_FIBERS.N0]
        # check if other LV contains such information
        try:
            for key in to_regress:
                _ = other_LV.get(container_loc, key)
        except:
            container_loc = self.check_enum(container_loc)
            key = self.check_enum(key)
            raise ValueError("Could not find '{}' in other LV object within '{}' container".format(
                key, container_loc))
        # apply regression
        reg_data = self.regress_from_other(
            other_LV, to_regress, container_loc=container_loc, **kwargs)

        if convert_to_cell_data:
            # Convert nodal data to cell data
            self.transform_point_data_to_cell_data(
                LV_FIBERS.F0, "mean", axis=0)
            self.transform_point_data_to_cell_data(
                LV_FIBERS.S0, "mean", axis=0)
            self.transform_point_data_to_cell_data(
                LV_FIBERS.N0, "mean", axis=0)

        if compute_angles:
            self.compute_fiber_angles()
            if convert_to_cell_data:
                # Convert nodal data to cell data
                self.transform_point_data_to_cell_data(
                    LV_FIBERS.F0_ANGLES, "mean", axis=0)
                self.transform_point_data_to_cell_data(
                    LV_FIBERS.S0_ANGLES, "mean", axis=0)
                self.transform_point_data_to_cell_data(
                    LV_FIBERS.N0_ANGLES, "mean", axis=0)

        return reg_data

    def interpolate_fibers(self, other_LV,
                       container_loc=GEO_DATA.MESH_POINT_DATA,
                       compute_angles=True,
                       convert_to_cell_data=True,
                       **kwargs):
        if not issubclass(other_LV.__class__, BaseContainerHandler):
            raise ValueError(
                "Other LV object must be subclass of BaseContainerHandler.")

        # get list of fibers to regress
        to_regress = [LV_FIBERS.F0, LV_FIBERS.S0, LV_FIBERS.N0]
        # check if other LV contains such information
        try:
            for key in to_regress:
                _ = other_LV.get(container_loc, key)
        except:
            container_loc = self.check_enum(container_loc)
            key = self.check_enum(key)
            raise ValueError("Could not find '{}' in other LV object within '{}' container".format(
                key, container_loc))
        # apply regression
        reg_data = self.interpolate_from_other(
            other_LV, to_regress, container_loc=container_loc, **kwargs)

        if convert_to_cell_data:
            # Convert nodal data to cell data
            self.transform_point_data_to_cell_data(
                LV_FIBERS.F0, "mean", axis=0)
            self.transform_point_data_to_cell_data(
                LV_FIBERS.S0, "mean", axis=0)
            self.transform_point_data_to_cell_data(
                LV_FIBERS.N0, "mean", axis=0)

        if compute_angles:
            self.compute_fiber_angles()
            if convert_to_cell_data:
                # Convert nodal data to cell data
                self.transform_point_data_to_cell_data(
                    LV_FIBERS.F0_ANGLES, "mean", axis=0)
                self.transform_point_data_to_cell_data(
                    LV_FIBERS.S0_ANGLES, "mean", axis=0)
                self.transform_point_data_to_cell_data(
                    LV_FIBERS.N0_ANGLES, "mean", axis=0)

        return reg_data