from os import path
import pathlib

import pyvista as pv

from .components import States
from project_heart.enums import *
from project_heart import utils
import numpy as np
from collections import deque


class Geometry():
    def __init__(self, *args, **kwargs):
        self.mesh = pv.UnstructuredGrid()
        self.states = States()
        self._masks = {}

    def __print__(self):
        return self.mesh

    # ----------------------------------------------------------------
    # Build methods

    def from_pyvista_dataset(self, dataset):
        self.mesh = dataset

    def from_pyvista_read(self, filename, **kwargs):
        self.mesh = pv.read(filename, **kwargs)

    def from_pyvista_wrap(self, arr, **kwargs):
        self.mesh = pv.wrap(arr, **kwargs)

    def from_nodes_elements(self,
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
        self.mesh = pv.UnstructuredGrid(cells_dict, points)

    def from_xplt(self, xplt, **kwargs):
        if isinstance(xplt, pathlib.Path):
            from febio_python.xplt import read_xplt
            xplt = read_xplt(str(xplt))
        if isinstance(xplt, str):
            from febio_python.xplt import read_xplt
            xplt = read_xplt(xplt)

        # create mesh dataset from nodes and elements in xplt file
        self.from_nodes_elements(
            xplt["nodes"], xplt["elems"][0][:, 1:], **kwargs)  # WARNING: xplt["elems"][0] is temporary, I will modify to xplt["elems"] later (this is due an error in xplt_parser)
        # add timesteps
        self.states.set_timesteps(xplt["timesteps"])
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
            self.states.add(key, _data, DATA_FORMAT(data_format))

    # ----------------------------------------------------------------
    # Write methods (TO DO)

    def to_xml(self, filename, mode="feb", **kwargs):
        raise NotImplementedError()

    def to_vtk(self, filename, **kwargs):
        self.mesh.save(filename, **kwargs)

    def to_csv(self, filename, **kwargs):
        """ Write states to csv file """
        raise NotImplementedError()

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
    def cells(self, key: VTK_ELEMENTS = None, mask: np.ndarray = None) -> dict or np.ndarray:
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
            return self.mesh.cells_dict

    def elements(self, key: VTK_ELEMENTS = None, mask: np.ndarray = None) -> dict or np.ndarray:
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
        return self.cells(key=key, mask=mask)

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
