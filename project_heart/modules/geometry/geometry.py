from os import path
import pathlib
from turtle import st

import pyvista as pv

from .components import States
from project_heart.enums import *
from project_heart import utils
import numpy as np
from collections import deque
import json
import functools


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Geometry():
    def __init__(self,
                 mesh=None,
                 *args, **kwargs):
        if mesh is None:
            self.mesh = pv.UnstructuredGrid()
        else:
            self.mesh = mesh

        self._surface_mesh = pv.UnstructuredGrid()
        self.states = States()
        self._nodesets = {}  # {"nodeset_name": [ids...], ...}
        self._elemsets = {}  # {"elemset_name": [ids...], ...}
        self._surfaces_oi = {}
        self._normal = None

        # represent virtual nodes that are not in mesh but are used in other calculations
        self._virtual_nodes = {}
        # represent virtual elems that are not in mesh but are used in other calculations
        self._virtual_elems = {}

        # represent 'sets' of nodal relationships -> should defined as {"key": [[node_a, node_b]...]}
        self._discrete_sets = {}

        self._bcs = {}

        self._X = np.array([1., 0., 0.])
        self._Y = np.array([0., 1., 0.])
        self._Z = np.array([0., 0., 1.])

    def __print__(self):
        return self.mesh

    # ----------------------------------------------------------------
    # Build methods

    @classmethod
    def from_pyvista_dataset(cls, dataset):
        return cls(mesh=dataset)

    @classmethod
    def from_pyvista_read(cls, filename, identifier=None, threshold=None, **kwargs):
        geo = cls(mesh=pv.read(filename, **kwargs))
        if identifier is not None:
            geo.filter_mesh_by_scalar(identifier, threshold)
        return geo

    @classmethod
    def from_pyvista_wrap(cls, arr, **kwargs):
        return cls(mesh=pv.wrap(arr, **kwargs))

    @classmethod
    def from_nodes_elements(cls,
                            nodes: np.ndarray,
                            elements: np.ndarray,
                            el_offset: int = 0,
                            p_dtype: np.dtype = np.float32,
                            e_dtype: np.dtype = np.int64,
                            **kwargs):
        """Creates a mesh dataset from pre-defined nodes and elements. 
            Note: as it creates mesh purely based on nodes and elements, any additional information will be ignored.

        Args:
            nodes (np.ndarray): An array containing [x,y,z] coordinates (size should be nx3).
            elements (np.ndarray): A nested list of elements (expected as np.object array) indicating the node index of each element.
            el_offset (int): offset that indicates which element it should start counting from. Defaults to 0.
            p_dtype (np.type, optional): dtype of points. Defaults to np.float32.
            e_dtype (np.type, optional): dtype of elements. Defaults to np.int64.
        """
        points = np.array(
            nodes, dtype=p_dtype)  # ensure points is treated as a numpy array
        # create dictionary of cell types
        cells_dict = dict()
        for x in elements:
            nels = len(x)  # cell type is related to number of points
            # get VTK element type based on number of points in element
            vtk_eltype = N_PTS_TO_VTK_ELTYPE[nels].value
            # ensure we are dealing with numpy arrays
            if not isinstance(x, np.ndarray):
                x = np.asarray(x, dtype=e_dtype)
            # add element node indexes (x) to cells dictionary
            if vtk_eltype in cells_dict:
                cells_dict[vtk_eltype].append(x-el_offset)
            else:
                cells_dict[vtk_eltype] = deque([x-el_offset])
        # stack each cell type as numpy arrays
        for key, value in cells_dict.items():
            cells_dict[key] = np.vstack(value).astype(e_dtype)
        # create mesh
        return cls(mesh=pv.UnstructuredGrid(cells_dict, points))

    @classmethod
    def from_xplt(cls, xplt, **kwargs):
        if isinstance(xplt, pathlib.Path):
            from febio_python.xplt import read_xplt
            xplt = read_xplt(str(xplt))
        if isinstance(xplt, str):
            from febio_python.xplt import read_xplt
            xplt = read_xplt(xplt)

        # create mesh dataset from nodes and elements in xplt file
        geo = cls.from_nodes_elements(
            xplt["nodes"], xplt["elems"][0][:, 1:], **kwargs)  # WARNING: xplt["elems"][0] is temporary, I will modify to xplt["elems"] later (this is due an error in xplt_parser)
        # add timesteps
        geo.states.set_timesteps(xplt["timesteps"])
        # add states
        n = len(xplt["timesteps"])
        for i, (key, data_format) in enumerate(zip(xplt["data_keys"], xplt["data_format"])):
            data = xplt["data"]
            shape = data[0][i].shape
            # data_sequence = np.array(data[:, i])
            # _data = np.zeros((n, shape[0], shape[1]), dtype=np.float32)
            # for step in range(n):
            #     _data[step] = data_sequence[step]
            _data = np.vstack(np.array(data[:, i])).reshape(
                (n, shape[0], shape[1]))  # optimized version
            geo.states.add(key, _data, DATA_FORMAT(data_format))
        return geo

    # ----------------------------------------------------------------
    # Write methods (TO DO)

    def to_xml(self, filename, mode="feb", **kwargs):
        raise NotImplementedError()

    def to_vtk(self, filename, **kwargs):
        self.mesh.save(filename, **kwargs)

    def to_csv(self, filename, **kwargs):
        """ Write states to csv file """
        raise NotImplementedError()

    def to_dict(self, **kwargs) -> dict:

        _d = dict()
        _d[GEO_DICT.NODES.value] = np.array(
            self.points(), dtype=np.float64)  # xyz
        _d[GEO_DICT.ELEMENTS.value] = self.elements(as_json_ready=True)
        _d[GEO_DICT.NODESETS.value] = self._nodesets
        _d[GEO_DICT.ELEMTSETS.value] = self._elemsets
        _d[GEO_DICT.SURFACES.value] = self._surfaces_oi

        _d[GEO_DICT.VIRTUAL_NODES.value] = self._virtual_nodes
        _d[GEO_DICT.DISCRETE_SETS.value] = self._discrete_sets

        _d[GEO_DICT.BC.value] = self._bcs

        return _d

    def to_json(self, filename, **kwargs) -> None:
        non_serialized_d = self.to_dict(**kwargs)
        # serialized_d = dict()
        # for key, value in non_serialized_d.items():
        #     if isinstance(value, dict):
        #         serialized_d[key] = dict()
        #         for subkey, subvalue in value.items():
        #             serialized_d[key][subkey] = subvalue.tolist()
        #     else:
        #         serialized_d[key] = value.tolist()

        with open(filename, "w") as outfile:
            # json.dump(serialized_d, outfile, indent=indent, sort_keys=sort_keys, **kwargs)
            json.dump(non_serialized_d, outfile, sort_keys=False,
                      cls=NumpyEncoder, **kwargs)

    # ----------------------------------------------------------------
    # Reference methods

    # reference to points or nodes

    def points(self, mask: np.ndarray = None) -> np.ndarray:
        """Returns a pointer to the list of points in the mesh.

        Args:
            mask (np.ndarray, optional): An index or boolean mask. Defaults to None.

        Returns:
            np.ndarray: pointer to array of points [[x,y,z]...] (nx3).
        """
        if mask is not None:
            return self.mesh.points[mask]
        else:
            return self.mesh.points

    def nodes(self, mask: np.ndarray = None) -> np.ndarray:
        """Alias of self.points method: returns a pointer to the list of points in the mesh.

        Args:
            mask (np.ndarray, optional): An index or boolean mask. Defaults to None.

        Returns:
            np.ndarray: pointer to array of points [[x,y,z]...] (nx3).
        """
        return self.points(mask=mask)

    # reference to cells or elements
    def cells(self, key: VTK_ELEMENTS = None, mask: np.ndarray = None, as_json_ready=False, **kwargs) -> dict or np.ndarray:
        """
        Returns a pointer to the dictionary of cells in the mesh. 
        If key is provided, it returns the array of cells of given key. 
        If key and mask are provided, it returns the specified range of cells of given key and mask.

        Args:
            key (VTK_ELEMENTS, optional): int corresponding to VTK_ELEMENTS number (ex: 12-> HEXAHEDRON). Defaults to None.
            mask (np.ndarray, optional): An index or boolean mask. Defaults to None.

        Returns:
            dict or np.ndarray: A dictionary containing {VTK_ELEMENTS: np.ndarray([]), ...} pairs or a specified range of cells (ndarray (nxm)).
        """
        if key is not None:
            data = self.mesh.cells_dict[key]
            if mask is not None:
                return data[mask]
            else:
                return data
        else:
            if as_json_ready:
                new_cell_dict = dict()
                for key, value in self.mesh.cells_dict.items():
                    new_cell_dict[VTK_ELEMENTS(key).name] = value
                return new_cell_dict
            else:
                return self.mesh.cells_dict

    def elements(self, key: VTK_ELEMENTS = None, mask: np.ndarray = None, **kwargs) -> dict or np.ndarray:
        """
        Alias of self.cells method.
        Returns a pointer to the dictionary of cells in the mesh. 
        If key is provided, it returns the array of cells of given key. 
        If key and mask are provided, it returns the specified range of cells of given key and mask.

        Args:
            key (VTK_ELEMENTS, optional): int corresponding to VTK_ELEMENTS number (ex: 12-> HEXAHEDRON). Defaults to None.
            mask (np.ndarray, optional): An index or boolean mask. Defaults to None.

        Returns:
            dict or np.ndarray: A dictionary containing {VTK_ELEMENTS: cells...} pairs or a specified range of cells (ndarray (nxm)).
        """
        return self.cells(key=key, mask=mask, **kwargs)

    def timesteps(self) -> list:
        """Returns a pointer to the timesteps array contained in self.States.

        Returns:
            list: list of timesteps [0.0, t1, t2 ... t]
        """
        return self.States.timesteps

    def states(self) -> dict:
        """
        Returns a pointer to the the states dictionary in self.States. 
        Only use this method if you want to directly modify states information. 
        Otherwise use 'self.get'.

        Returns:
            dict: {DATA_FIELDS: np.ndarray[], ...} pairs
        """
        return self.States.data

    # ----------------------------------------------------------------
    # Get methods -> returns point, cell or state(s) data

    def get(self, what: str = GEO_DATA.STATES,
            key: str or None = None,
            mask: np.ndarray or None = None,
            i: int or None = None,
            t: float or None = None) -> np.ndarray:
        """
        This method is convinient way to retrieve specified data from the Geometry object. 
        The argument 'what' must be specified, it determines the location in which the
        method will look for the data. The 'mask' argument can be used to retrieve only
        specified range; it can be boolean or index array. If data is to be retrieved from
        States, a key must be specified. The arguments 'i' and 't' can be used to specify
        a given state timestep; 'i' determines the timestep index, while 't' determines
        the timestep value (if t is not in timesteps, it will look for the closest valid timestep). 

        Note: if 'i' and 't' are specified, 'i' will be used and 't' will be 
        ignored. 

        Args:
            what (GEO_DATA, optional): A string inferreing to GEO_DATA (where to look for the data). Defaults to GEO_DATA.STATES.
            key (str, optional): Identifies the state data to be retrieved. Defaults to None.
            mask (np.ndarray or None, optional): A boolean or index array. Defaults to None.
            i (int or None, optional): Timestep index. Defaults to None.
            t (float or None, optional): Timestep value. Defaults to None.

        Raises:
            ValueError: If 'what' is not int or GEO_DATA.
            ValueError: If States is requested but 'key' is not specified (is None).
            ValueError: If GEO_DATA is invalid.

        Returns:
            np.ndarray: Array of requested data.
        """
        if not (isinstance(what, int) or isinstance(what, GEO_DATA)):
            raise ValueError(
                "Should specify what to get by using a GEO_DATA value or its respective integer.")

        if what == GEO_DATA.NODES:
            return self.nodes()
        elif what == GEO_DATA.ELEMS:
            return self.elements()
        elif what == GEO_DATA.STATES:
            if key is None:
                raise ValueError("State Data 'key' must be specified.")
            return self.states.get(key, mask=mask, i=i, t=t)
        else:
            raise ValueError(
                "Not sure where to get data from: 'what', %s, should be one of the GEO_DATA values." % what)

    def set_normal(self, normal: np.ndarray, dtype: np.dtype = np.float64) -> np.ndarray:
        self._normal = np.asarray(normal, dtype=dtype)
        return self._normal

    def get_normal(self) -> np.ndarray:
        if self._normal is None:
            raise RuntimeError(
                "Normal was not initialized. Either set it manually or use a class method to do so.")
        return self._normal

    # ----------------------------------------------------------------
    # add methods

    def add_nodeset(self, name: str, ids: np.ndarray, overwrite: bool = False, dtype: np.dtype = np.int64) -> None:
        """Adds a list of indexes as a nodeset. 

        Args:
            name (str): nodeset name. 
            ids (np.ndarray): list of indexes referencing the nodes of given nodeset.
            overwrite (bool, optional): _description_. Defaults to False.
            dtype (np.dtype, optional): _description_. Defaults to np.int64.

        Raises:
            ValueError: if ids is not an ndarray
            ValueError: if ids is not only composed of integers
            ValueError: if max id is greater than number of nodes/points
            KeyError: _description_
        """
        if isinstance(name, Enum):
            name = name.value
        # check if ids is numpy array
        if not isinstance(ids, np.ndarray):
            raise ValueError("ids must be np.ndarray of integers")
        # check if ids is a list of integers only
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("ids must be integers.")
        # check if maximum id within n_nodes
        if np.max(ids) > self.mesh.n_points:
            raise ValueError(
                "maximum reference id is greater than number of nodes. \
                    Surfaces are list of integers referencing the indexes of nodes array.")
        # check if key is already used (if overwrite is False)
        if overwrite == False and name in self._nodesets:
            raise KeyError(
                "Nodeset '%s' already exists. Please, set overwrite flag to True if you want to replace it." % name)
        # add nodeset to the dictionary
        self._nodesets[name] = ids.astype(dtype)

    def get_nodeset(self, name: str):
        if isinstance(name, Enum):
            name = name.value
        if name not in self._nodesets:
            raise KeyError("Nodeset '%s' does not exist." % name)
        else:
            return self._nodesets[name]

    def add_virtual_node(self, name, node, replace=False) -> None:
        if isinstance(name, Enum):
            name = name.value
        if replace == False and name in self._virtual_nodes:
            raise KeyError(
                "Virtual node '%s' already exists. If you want to replace it, set replace flag to true." % name)
        self._virtual_nodes[name] = node

    def get_virtual_node(self, name: str) -> np.ndarray:
        if isinstance(name, Enum):
            name = name.value
        if name not in self._virtual_nodes:
            raise KeyError(
                "Virtual node '%s' does not exist. Did you create it?" % name)
        return self._virtual_nodes[name]

    def add_virtual_elem(self, name, elems, replace=False) -> None:
        if isinstance(name, Enum):
            name = name.value
        if replace == False and name in self._virtual_elems:
            raise KeyError(
                "Virtual elem '%s' already exists. If you want to replace it, set replace flag to true." % name)
        self._virtual_elems[name] = elems

    def get_virtual_elem(self, name: str) -> np.ndarray:
        if isinstance(name, Enum):
            name = name.value
        if name not in self._virtual_elems:
            raise KeyError(
                "Virtual elem '%s' does not exist. Did you create it?" % name)
        return self._virtual_elems[name]

    def add_discrete_set(self, name, discrete_set: np.ndarray, replace: bool = False, dtype: np.dtype = np.int64) -> None:
        """Adds a discrete set to the current object. Discreset sets, in this case, are defined as relationships\
            between nodes. It should be a np.ndarray of shape [nx2], in which n refers to the number of relations\
            in the set, and the columns represent node ids of two different set of nodes (relations will be defined\
            rowise).

        Args:
            name (dict key acceptable): Key reference to discrete set. 
            discrete_set (np.ndarray): Array of set relationships defined in rowise fashion \
                (each row of col 0 will be related to each row in col 1). Must be of shape [nx2].
            replace (bool, optional): Flag that prevents overwrites. Defaults to False.
            dtype (np.dtype, optional): If discrete_set is not a np.ndarray, it will be converted\
                to one and dtype will be used to determine it's type. Defaults to np.int64.

        Raises:
            KeyError: If replace is invalid and name is already present in self._discrete_sets'
            ValueError: If discrete_set is not a np.ndarray and it could not be converted to one;\
                or if shape of array is not [nx2].
        """
        if replace == False and name in self._discrete_sets:
            raise KeyError(
                "Discrete set '%s' already exists. If you want to replace it, set replace flag to true." % name)
        # check for discreset set shape
        if not isinstance(discrete_set, np.ndarray):
            try:
                discrete_set = np.array(discrete_set, dtype)
            except:
                ValueError(
                    "Could not transform discrete_set into a np.ndarray.")
        if len(discrete_set) > 2 or discrete_set.shape[1] != 2:
            raise ValueError(
                "discrete_set must have shape of [nx2], where n refers to number of node relations")
        self._discrete_sets[name] = discrete_set

    def add_bc(self, name, bc, replace=False):
        if replace == False and name in self._bcs:
            raise KeyError(
                "Boundary condition '%s' already exists. If you want to replace it, set replace flag to true." % name)
        self._bcs[name] = bc

    # -------------------------------
    # Mesh wrapped functions

    def filter_mesh_by_scalar(self, identifier: str, threshold: list, **kwargs) -> None:
        self.mesh.set_active_scalars(identifier)
        self.mesh = self.mesh.threshold(threshold, **kwargs)
        self._surface_mesh = self.mesh.extract_surface()

    def extract_surface_mesh(self, **kwars):
        self._surface_mesh = self.mesh.extract_surface(**kwars)
        return self._surface_mesh

    def get_surface_mesh(self, **kwargs):
        if self._surface_mesh.n_points == 0:
            return self.extract_surface_mesh(**kwargs)
        else:
            return self._surface_mesh

    # -------------------------------
    # other functions

    def map_surf_ids_to_global_ids(self, surf_ids, dtype=np.int64):
        lvsurf = self.get_surface_mesh()
        surf_to_global = lvsurf.point_data["vtkOriginalPointIds"]
        return np.array(surf_to_global[surf_ids], dtype=dtype)

    def get_node_ids_for_each_cell(self, surface=False, **kwargs):

        if surface:
            mesh = self.get_surface_mesh().copy().cast_to_unstructured_grid()
        else:
            mesh = self.mesh

        faces = deque()
        i, offset = 0, 0
        cc = mesh.cells  # fetch up front
        while i < mesh.n_cells:
            nn = cc[offset]
            faces.append(cc[offset+1:offset+1+nn])
            offset += nn + 1
            i += 1
        return faces

    def transform_point_data_to_cell_data(self,
                                          data_key,
                                          method="max",
                                          surface=False,
                                          dtype=None,
                                          axis=-1,
                                          **kwargs):

        if surface:
            mesh = self.get_surface_mesh()
        else:
            mesh = self.mesh

        if method == "max":
            fun = functools.partial(np.max, axis=axis)
        elif method == "min":
            fun = functools.partial(np.min, axis=axis)
        elif method == "mean":
            fun = functools.partial(np.mean, axis=axis)
        else:
            if not callable(method):
                raise ValueError(
                    "method must be 'max' or 'min' or a callable function.")
            else:
                fun = method

        # get the node id map for each cell
        cell_node_ids = self.get_node_ids_for_each_cell(surface=surface)
        # get current data at points array
        pt_data = mesh.get_array(data_key, "points")
        # create new array with same length as number of cells
        cells_data = np.zeros((mesh.n_cells, pt_data.shape[-1]))
        # for each cell, apply function based on values from all cell nodes
        for cell_index, node_ids in enumerate(cell_node_ids):
            cells_data[cell_index] = fun(pt_data[node_ids])

        # set dtype as same from point data if not specified.
        if dtype is None:
            dtype = pt_data.dtype

        # save new data at cell level
        mesh.cell_data[data_key] = cells_data.astype(dtype)

        # return pointer to saved data
        return mesh.cell_data[data_key]

    def smooth_surface(self, **kwargs):
        """ Adjust point coordinates using Laplacian smoothing.\n
            Uses Pyvista smoothing method: https://bit.ly/37ee3hU \n
            WARNING: This function applies smoothing at surface level \n
            by modifying their coordinates directly, which does not imply \n
            that cell volumes (if volumetric mesh) will be adjusted.
        """
        self._surface_mesh = self.get_surface_mesh().smooth(**kwargs)
        surf_to_global = self._surface_mesh.point_data["vtkOriginalPointIds"]
        self.mesh.points[surf_to_global] = self._surface_mesh.points.copy()

    # -------------------------------
    # plot

    def plot(self,
             mode="mesh",
             scalars=None,
             re=False,
             vnodes=[],
             vcolor="red",
             cat_exclude_zero=False,
             **kwargs):

        plot_args = dict(cmap="Set2",
                         opacity=1.0,
                         show_edges=False,
                         ambient=0.2,
                         diffuse=0.5,
                         specular=0.5,
                         specular_power=90
                         )

        plot_args.update(kwargs)
        plotter = pv.Plotter(lighting='three lights')
        plotter.background_color = 'w'
        plotter.enable_anti_aliasing()
        plotter.enable_shadows()

        if mode == "mesh":
            mesh = self.mesh
        elif mode == "surface":
            mesh = self.get_surface_mesh()

        if cat_exclude_zero:
            vals = np.copy(mesh.get_array(scalars))
            vals[vals == 0] = np.min(vals[vals > 0])-1
            if len(vals) == mesh.n_points:
                mesh.point_data["cat_exclude_zero_FOR_PLOT"] = vals
            else:
                mesh.cell_data["cat_exclude_zero_FOR_PLOT"] = vals
            scalars = "cat_exclude_zero_FOR_PLOT"

        plotter.add_mesh(mesh, scalars=scalars, **plot_args)

        if len(vnodes) > 0:
            for i, vn in enumerate(vnodes):
                if isinstance(vn, str) or isinstance(vn, Enum):
                    vname = vn
                    cvcolor = vcolor
                elif isinstance(vn, tuple) or isinstance(vn, list):
                    vname = vn[0]
                    cvcolor = vn[1]
                else:
                    raise ValueError("Invalid virtual node type information to plot. \
                        Must be '(either str or Enum)' or '(tuple or list)'.")
                node = self.get_virtual_node(vname)
                plotter.add_points(node, color=cvcolor,
                                   point_size=350, render_points_as_spheres=True)

        if re:
            return plotter
        else:
            plotter.show()

    # -------------------------------
    # Check Methods

    def check_enum(self, name):
        if isinstance(name, Enum):
            name = name.value
        return name

    def check_mesh_data(self, mesh_data: str) -> tuple:
        """Check whether given mesh_data is in mesh or surface mesh

        Args:
            mesh_data (str): Mesh data name (or Enum corresponding to mesh data name).

        Returns:
            tuple (bool, bool): (in_mesh, in_surf_mesh)
        """

        mesh_data = self.check_enum(mesh_data)

        in_mesh = mesh_data in self.mesh.array_names
        in_surf_mesh = mesh_data in self.get_surface_mesh().array_names

        return (in_mesh, in_surf_mesh)

    def check_tet4_mesh(self) -> bool:
        """Checks if mesh is composed of pure simple tetrahedrons (4 nodes).

        Returns:
            bool
        """
        cells_dict = self.cells()
        if len(cells_dict) > 1:  # If mesh is not composed of pure tet4 elements
            return False
        if VTK_ELEMENTS.TETRA.value in cells_dict:
            return True
        return False

    def check_tri3_surfmesh(self) -> bool:
        """Checks if surface mesh is composed of pure simple triangles (3 nodes).

        Returns:
            bool
        """
        surf = self.get_surface_mesh().copy()
        cells_dict = surf.cast_to_unstructured_grid().cells_dict
        if len(cells_dict) > 1:  # If mesh is not composed of pure tri elements
            return False
        if VTK_ELEMENTS.TRIANGLE.value in cells_dict:
            return True
        return False

    def check_hex8_mesh(self) -> bool:
        """Checks if mesh is composed of pure simple hexahedrons (8 nodes).

        Returns:
            bool
        """
        cells_dict = self.cells()
        if len(cells_dict) > 1:  # If mesh is not composed of pure tet4 elements
            return False
        if VTK_ELEMENTS.HEXAHEDRON.value in cells_dict:
            return True
        return False

    def check_vtkElements_mesh(self, vtk_elem_value: int) -> bool:
        """Checks if mesh is composed of pure 'vtk_elem_value'.

        Args:
            vtk_elem_value (int or Enum): Vtk element number representation. 

        Returns:
            bool
        """
        vtk_elem_value = self.check_enum(vtk_elem_value)
        cells_dict = self.cells()
        if len(cells_dict) > 1:  # If mesh is not composed of pure tet4 elements
            return False
        if vtk_elem_value in cells_dict:
            return True
        return False

    def check_vtkElements_surfmesh(self, vtk_elem_value: int) -> bool:
        """Checks if surface mesh is composed of pure 'vtk_elem_value'.

        Args:
            vtk_elem_value (int or Enum): Vtk element number representation. 

        Returns:
            bool
        """
        vtk_elem_value = self.check_enum(vtk_elem_value)
        surf = self.get_surface_mesh().copy()
        cells_dict = surf.cast_to_unstructured_grid().cells_dict
        if len(cells_dict) > 1:  # If mesh is not composed of pure tet4 elements
            return False
        if vtk_elem_value in cells_dict:
            return True
        return False
