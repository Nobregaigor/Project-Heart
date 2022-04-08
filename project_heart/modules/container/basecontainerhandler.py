import os
from os import path

import pathlib
from turtle import st

import pyvista as pv

from .components import States
from project_heart.enums import *
from project_heart.utils.vector_utils import angle_between
from project_heart import utils
import numpy as np
from collections import deque
import json
import functools

from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BaseContainerHandler():
    """Data storage obj.
    """

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

        # other into
        self._ref_file = None
        self._ref_dir = None

        self._cell_list = None
        self._surf_cell_list = None

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
        # save reference file info
        geo._ref_file = Path(filename)
        geo._ref_dir = geo._ref_file.parents[0]
        return geo

    @classmethod
    def from_pyvista_wrap(cls, arr, **kwargs):
        return cls(mesh=pv.wrap(arr, **kwargs))

    @staticmethod
    def set_pv_UnstructuredGrid_from_nodes_and_elements(
        nodes,
        elements,
        el_offset: int = 0,
        p_dtype: np.dtype = np.float32,
        e_dtype: np.dtype = np.int64
    ):

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

        return pv.UnstructuredGrid(cells_dict, points)

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
        mesh = BaseContainerHandler.set_pv_UnstructuredGrid_from_nodes_and_elements(
            nodes=nodes,
            elements=elements,
            el_offset=el_offset,
            p_dtype=p_dtype,
            e_dtype=e_dtype,
        )
        # create mesh
        return cls(mesh=mesh)

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
        # save reference file info
        geo._ref_file = Path(xplt)
        geo._ref_dir = geo._ref_file.parents[0]
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

    def to_dict(self,
                mesh_point_data={},
                mesh_cell_data={},
                export_all_mesh_data=False,
                nodeset_enums: list = None,
                surfaces_oi_enums: list = None,
                ** kwargs) -> dict:

        _d = dict()
        _d[GEO_DICT.NODES.value] = np.array(
            self.points(), dtype=np.float64)  # xyz
        _d[GEO_DICT.ELEMENTS.value] = self.elements(as_json_ready=True)

        # nodesets
        if nodeset_enums:
            if not isinstance(nodeset_enums, list):
                raise ValueError(
                    "nodeset_enums must be list of enum-like values for nodesets.")
            _d[GEO_DICT.NODESETS.value] = dict()
            for enumlike in nodeset_enums:
                _d[GEO_DICT.NODESETS.value].update(
                    self.get_nodesets_from_enum(enumlike))
        else:
            _d[GEO_DICT.NODESETS.value] = self._nodesets

        _d[GEO_DICT.ELEMTSETS.value] = self._elemsets

        # surfaces
        if surfaces_oi_enums:
            if not isinstance(surfaces_oi_enums, list):
                raise ValueError(
                    "surfaces_oi_enums must be list of enum-like values for nodesets.")
            _d[GEO_DICT.SURFACES.value] = dict()
            for enumlike in nodeset_enums:
                _d[GEO_DICT.SURFACES.value].update(
                    self.get_surface_oi_from_enum(enumlike))
        else:
            _d[GEO_DICT.SURFACES.value] = self._surfaces_oi

        _d[GEO_DICT.VIRTUAL_NODES.value] = self._virtual_nodes
        _d[GEO_DICT.DISCRETE_SETS.value] = self._discrete_sets

        _d[GEO_DICT.BC.value] = self._bcs

        # Export mesh data
        _d[GEO_DICT.MESH_POINT_DATA.value] = {}
        _d[GEO_DICT.MESH_CELL_DATA.value] = {}
        if export_all_mesh_data:
            mesh_point_data = self.mesh.point_data.keys()
            mesh_cell_data = self.mesh.cell_data.keys()
        for key in mesh_point_data:
            _d[GEO_DICT.MESH_POINT_DATA.value][key] = self.mesh.point_data[key].tolist()
        for key in mesh_cell_data:
            _d[GEO_DICT.MESH_CELL_DATA.value][key] = self.mesh.cell_data[key].tolist()

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
                      cls=NumpyEncoder)

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
            key: str = None,
            mask: np.ndarray = None,
            i: int = None,
            t: float = None) -> np.ndarray:
        """
        This method is convinient way to retrieve specified data from the BaseContainerHandler object. 
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
            key (str, enum or list/tuple of keys, optional): Identifies the state data to be retrieved.\
                If list is provided, will try to vertically stack data. Defaults to None.
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

        if isinstance(key, (list, tuple, np.ndarray)):
            try:
                return np.hstack([self.get(what, subkey, mask, i, t) for subkey in key])
            except:
                raise ValueError("Could stack data from list of keys. \
                    Try calling this function one key at a time.")

        what = self.check_enum(what)
        key = self.check_enum(key)

        if what == GEO_DATA.NODES:
            return self.nodes()
        elif what == GEO_DATA.ELEMS:
            return self.elements()
        elif what == GEO_DATA.STATES:
            if key is None:
                raise ValueError("State Data 'key' must be specified.")
            return self.states.get(key, mask=mask, i=i, t=t)
        elif what == GEO_DATA.MESH_POINT_DATA:
            if mask is not None:
                return self.mesh.point_data[key][mask]
            else:
                return self.mesh.point_data[key]
        elif what == GEO_DATA.MESH_CELL_DATA:
            if mask is not None:
                return self.mesh.cell_data[key][mask]
            else:
                return self.mesh.cell_data[key]
        elif what == GEO_DATA.SURF_POINT_DATA:
            if mask is not None:
                return self.get_surface_mesh().point_data[key][mask]
            else:
                return self.get_surface_mesh().point_data[key]
        elif what == GEO_DATA.SURF_CELL_DATA:
            if mask is not None:
                return self.get_surface_mesh().cell_data[key][mask]
            else:
                return self.get_surface_mesh().cell_data[key]
        else:
            try:
                what_enum = GEO_DATA(what)
                raise ValueError("%s not found" % what_enum.name)
            except:
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

    def get_bc(self, bc_name: str) -> tuple:
        try:
            return self._bcs[bc_name]
        except KeyError:
            raise ValueError(
                "bc_name '%s' not found. Did you create it?" % bc_name)

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
        name = self.check_enum(name)
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

    def get_surface_oi(self, name: str):
        if isinstance(name, Enum):
            str_val = name.name
            name = name.value
        else:
            str_val = name
        if name not in self._nodesets:
            raise KeyError(
                "Surface of interest '%s' does not exist." % str_val)
        else:
            return self._surfaces_oi[name]

    def add_virtual_node(self, name, node, replace=False) -> None:
        name = self.check_enum(name)
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
        if len(discrete_set.shape) > 2 or discrete_set.shape[1] != 2:
            raise ValueError(
                "discrete_set must have shape of [nx2], where n refers to number of node relations")
        self._discrete_sets[name] = discrete_set

    def add_bc(self, name, bc_type, bc, replace=False):
        if replace == False and name in self._bcs:
            raise KeyError(
                "Boundary condition '%s' already exists. If you want to replace it, set replace flag to true." % name)
        self._bcs[name] = (bc_type, bc)

    # -------------------------------
    # Mesh wrapped functions

    def filter_mesh_by_scalar(self, identifier: str, threshold: list, **kwargs) -> None:
        self.mesh.set_active_scalars(identifier)
        self.mesh = self.mesh.threshold(threshold, **kwargs)
        self._surface_mesh = self.mesh.extract_surface()

    def extract_surface_mesh(self, **kwars):
        self._surface_mesh = self.mesh.extract_surface(**kwars)
        return self._surface_mesh

    def get_surface_mesh(self, force_extract=False, **kwargs):
        if force_extract:
            return self.extract_surface_mesh(**kwargs)
        else:
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

    def get_surface_id_map_from_mesh(self):
        lvsurf = self.get_surface_mesh()
        return lvsurf.point_data["vtkOriginalPointIds"]

    def get_node_ids_for_each_cell(self, surface=False, **kwargs):

        # prep
        if surface:
            if self._surf_cell_list is not None:
                return self._surf_cell_list
            mesh = self.get_surface_mesh().copy().cast_to_unstructured_grid()
        else:
            if self._cell_list is not None:
                return self._cell_list
            mesh = self.mesh

        # compute - old (slow) method
        # faces = deque()
        # i, offset = 0, 0
        # cc = mesh.cells  # fetch up front
        # while i < mesh.n_cells:
        #     nn = cc[offset]
        #     faces.append(cc[offset+1:offset+1+nn])
        #     offset += nn + 1
        #     i += 1

        # compute
        cells_ids_list = deque()
        for cell_array in mesh.cells_dict.values():
            for cell in cell_array:
                cells_ids_list.append(cell)

        # save
        if surface:
            self._surf_cell_list = cells_ids_list
        else:
            self._cell_list = cells_ids_list

        return cells_ids_list

    def transform_point_data_to_cell_data(self,
                                          data_key,
                                          method="max",
                                          surface=False,
                                          dtype=None,
                                          axis=-1,
                                          **kwargs):
        # cehck if data_key is Enum representation of data key value
        data_key = self.check_enum(data_key)
        # cehck if data_key is present in mesh or surface mesh
        (in_mesh, in_surf_mesh) = self.check_mesh_data(data_key)
        if surface:
            if not in_surf_mesh:
                raise ValueError(
                    "data key '{}' not found in surface mesh data.".format(data_key))
            mesh = self.get_surface_mesh()
        else:
            if not in_mesh:
                raise ValueError(
                    "data key '{}' not found in mesh data.".format(data_key))
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
        if len(pt_data.shape) == 1:
            cells_data = np.zeros(mesh.n_cells)
        else:
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

    def merge_mesh_and_surface_mesh(self) -> pv.UnstructuredGrid:
        """Combines mesh and surface mesh into a single mesh dataset. This is often\
            required for some FEA solvers or other libraries (such as LDRB).

        Returns:
            pv.UnstructuredGrid: Merged mesh.
        """
        mesh = self.mesh.copy()
        mesh = mesh.merge(self.get_surface_mesh().copy())
        return mesh

    def prep_for_gmsh(self,
                      cellregionIds: np.ndarray,
                      mesh: pv.UnstructuredGrid = None,
                      ) -> pv.UnstructuredGrid:
        """Prepares a given mesh for gmsh meshion export. Includes gmsh:physical\
            and gmsh:geometrical data with 'cellregionIds' data.

        Args:
            cellregionIds (np.ndarray): Integer list identifying regions.
            mesh (pv.UnstructuredGrid, optional): Mesh object to export. Defaults to None (uses self.mesh).

        Returns:
            pv.UnstructuredGrid: Mesh dataset with "gmsh:physical" and "gmsh:geometrical" in cell_data.
        """
        if mesh is None:
            mesh = self.mesh.copy()

        mesh.cell_data["gmsh:physical"] = cellregionIds
        mesh.cell_data["gmsh:geometrical"] = cellregionIds
        return mesh

    def compute_angles_wrt_normal(self, vec_arr, check_orientation=False, degrees=False):
        normal_vec = np.repeat(np.expand_dims(
            self.get_normal(), 1), len(vec_arr), axis=1).T
        angles = angle_between(vec_arr, normal_vec,
                               check_orientation=check_orientation)
        if degrees:
            angles = np.degrees(angles)
        return angles

    def get_nodesets_from_enum(self, enum_like):
        if not isinstance(enum_like, object):
            raise ValueError(
                "enum_like must be Enum related to saved nodesets.")
        data = dict()
        for item in enum_like:
            try:
                data[item.name] = self.get_nodeset(item.value)
            except KeyError:
                continue
                # print("Unknown node set: %s" % item.name)
        return data

    def create_surface_oi_from_surface(self, surf_name):

        surf_map = self.get_surface_id_map_from_mesh()
        cell_id_list = self.get_node_ids_for_each_cell(surface=True)
        surf_name = self.check_enum(surf_name)
        try:
            surf = self.get_surface_mesh()
            index_map = surf.cell_data[surf_name]
        except KeyError:
            self.transform_point_data_to_cell_data(
                surf_name, "max", surface=True)
            surf = self.get_surface_mesh()
            index_map = surf.cell_data[surf_name]

        unique_vals = np.unique(index_map)
        for val in unique_vals:
            ioi = np.where(index_map == val)[0]
            surf_oi = deque()
            for i in ioi:
                surf_oi.append(surf_map[cell_id_list[i]])
            self._surfaces_oi[val] = list(surf_oi)

        return self._surfaces_oi

    def get_surface_oi_from_enum(self, enum_like):
        if not isinstance(enum_like, object):
            raise ValueError(
                "enum_like must be Enum related to saved nodesets.")
        data = dict()
        for item in enum_like:
            try:
                data[item.name] = self.get_surface_oi(item.value)
            except KeyError:
                continue
                # print("Unknown node set: %s" % item.name)
        return data

    def tetrahedralize(self,
                       backend=TETRA_BACKEND.TETGEN,
                       make_manifold=True,
                       **kwargs):
        """DOC PENDING.

        This is a wrap method for the Wildmeshing library: https://wildmeshing.github.io/python/\
        and the tetgen library https://tetgen.pyvista.org/. \
        Credits should be to the owners of the original libraries.

        Args:
            backend (_type_, optional): _description_. Defaults to TETRA_BACKEND.TETGEN.
            make_manifold (bool, optional): _description_. Defaults to True.

        Raises:
            ImportError: _description_
            ImportError: _description_
            NotImplementedError: _description_
        """

        backend = self.check_enum(backend)
        if backend == TETRA_BACKEND.WILDMESHING:
            try:
                import wildmeshing as wm
            except ImportError:
                raise ImportError(
                    "Wildmeshing library is required to tetrahedralize mesh with 'wildmeshing' backend. See https://wildmeshing.github.io/python/ for details.")

            # get Vertices and faces (surface mesh)
            surf = self.get_surface_mesh()
            V = surf.points
            F = surf.cast_to_unstructured_grid().cells_dict[5]
            # apply tetrahedralization
            tetra = wm.Tetrahedralizer(**kwargs)
            tetra.set_mesh(V, F)
            tetra.tetrahedralize()
            VT, TT = tetra.get_tet_mesh()
            mesh = BaseContainerHandler.set_pv_UnstructuredGrid_from_nodes_and_elements(
                nodes=VT,
                elements=TT,
                el_offset=0,
                p_dtype=np.float32,
                e_dtype=np.int64,
            )
            self.mesh = mesh            # save new mesh
            self.extract_surface_mesh()  # force extra new surface mesh
            os.remove("__tracked_surface.stl")  # delete generated file

        elif backend == TETRA_BACKEND.TETGEN:
            try:
                import tetgen
            except ImportError:
                raise ImportError(
                    "tetgen library is required to tetrahedralize mesh with 'tetgen' backend. See https://tetgen.pyvista.org/ for details.")

            tet = tetgen.TetGen(self.get_surface_mesh().triangulate())
            if make_manifold:
                tet.make_manifold()
            tet.tetrahedralize(**kwargs)
            self.mesh = tet.grid         # save new mesh
            self.extract_surface_mesh()  # force extra new surface mesh

        else:
            raise NotImplementedError(
                "We current support 'wildmeshing' and 'tetgen' backends. Check TETRA_BACKEND enum for details.")

    @staticmethod
    def regress(X, Y, XI,
                hidden_layer_sizes: tuple = (100,),
                early_stopping: bool = True,
                validation_fraction: float = 0.25,
                **kwargs):
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError:
            raise ImportError(
                "sklearn is required to perform regression. See https://bit.ly/3NX9bhR for details.")

        if XI.shape[-1] != X.shape[-1]:
            raise ValueError("Number of features must match. Please, verify.")

        reg = MLPRegressor(hidden_layer_sizes,
                           early_stopping=early_stopping,
                           validation_fraction=validation_fraction,
                           **kwargs)
        reg.fit(X, Y)
        return reg.predict(XI)

    def regress_from_array(self,
                           arr: np.ndarray,
                           new_xyz_domain: np.ndarray,
                           **kwargs):

        # try to match xyz with given arr size
        if len(arr) == self.mesh.n_points:
            xyz = self.points()
        elif len(arr) == self.get_surface_mesh().n_points:
            xyz = self.get_surface_mesh().points
        else:
            raise ValueError(
                """Could not match arr length with mesh points, or surface mesh points.\
                Regression is only supported for point arr, as number of features for\
                training and predicting must match exactly.""")

        return BaseContainerHandler.regress(xyz, arr, new_xyz_domain, **kwargs)

    def regress_from_data(self,
                          data_key: str,
                          new_xyz_domain: np.ndarray,
                          container_loc: int = GEO_DATA.MESH_POINT_DATA,
                          **kwargs):
        if container_loc != GEO_DATA.MESH_POINT_DATA and container_loc != GEO_DATA.SURF_POINT_DATA:
            raise ValueError(
                """Regression is only supported for point data, as number of features for\
                training and predicting must match exactly.""")

        # get data -> will raise error if could not retrieve data
        data = self.get(container_loc, data_key)
        return self.regress_from_array(data, new_xyz_domain, **kwargs)

    def regress_from_other(self,
                           other: object,
                           data_key: str,
                           container_loc: int = GEO_DATA.MESH_POINT_DATA,
                           **kwargs):
        if not issubclass(other.__class__, BaseContainerHandler):
            raise ValueError(
                "other object must be derived from BaseContainerHandler class.")

        data_key = self.check_enum(data_key)

        if container_loc == GEO_DATA.MESH_POINT_DATA:
            xyz = self.points()
            container = self.mesh.point_data
        elif container_loc == GEO_DATA.SURF_POINT_DATA:
            xyz = self.get_surface_mesh().points
            container = self.get_surface_mesh().point_data
        else:
            raise ValueError(
                """
                We only support MESH_POINT_DATA or SURF_POINT_DATA.\
                Regression is only supported for point data, as number of features for\
                training and predicting must match exactly. \
                If data is avaiable as point data, you can perform regression over\
                point data and then transform the result to cell data. \
                See documentation for further details.
                """)

        # apply regression
        reg_data = other.regress_from_data(
            data_key, xyz, container_loc=container_loc, **kwargs)

        if isinstance(data_key, (list, tuple, np.ndarray)):
            for i, key in enumerate(data_key):
                offset = 0
                shape = other.get(container_loc, key).shape
                d = reg_data[:, offset:offset+shape[-1]]
                container[self.check_enum(key)] = d
        else:
            container[data_key] = reg_data

        return reg_data

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

    def plot_streamlines(self,
                         vectors: str,
                         scalars=None,
                         decimate_boundary=0.90,
                         max_steps=1000,
                         surface_streamlines=False,
                         streamline_args={}
                         ):

        vectors = self.check_enum(vectors)
        seed_mesh = self.mesh.decimate_boundary(decimate_boundary)
        stream = self.mesh.streamlines_from_source(seed_mesh,
                                                   vectors,
                                                   surface_streamlines=surface_streamlines,
                                                   max_steps=max_steps,
                                                   integration_direction="both",
                                                   **streamline_args
                                                   )
        if scalars is None:
            scalars = vectors
        else:
            scalars = self.check_enum(scalars)
        p = pv.Plotter()
        p.enable_anti_aliasing()
        p.enable_shadows()
        p.background_color = 'w'
        # p.add_mesh(lv.mesh, color="beige")
        p.add_mesh(stream, scalars=scalars, lighting=True)
        p.show()

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
