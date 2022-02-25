import unittest
import pyvista as pv
from pyvista import examples
import numpy as np
from datetime import datetime

from project_heart.enums import *
from .geometry import Geometry


from os import path
import pathlib
FILE_PATH = pathlib.Path(path.dirname(path.realpath(__file__)))
TEST_FILES_DIR = FILE_PATH/"test_files"


def runtime(start_time, end_time):
    time_diff = (end_time - start_time)
    return time_diff.total_seconds() * 1000


def build_sample_grid():
    sample_points = np.array(
        [
            [-0.5, -0.5,  0.],
            [-0.5, -0.5,  0.5],
            [-0.5, -0.5,  1.],
            [-0.5,  0.,  0.],
            [-0.5,  0.,  0.5],
            [-0.5,  0.,  1.],
            [-0.5,  0.5,  0.],
            [-0.5,  0.5,  0.5],
            [-0.5,  0.5,  1.],
            [0., -0.5,  0.],
            [0., -0.5,  0.5],
            [0., -0.5,  1.],
            [0.,  0.,  0.],
            [0.,  0.,  0.5],
            [0.,  0.,  1.],
            [0.,  0.5,  0.],
            [0.,  0.5,  0.5],
            [0.,  0.5,  1.],
            [0.5, -0.5,  0.],
            [0.5, -0.5,  0.5],
            [0.5, -0.5,  1.],
            [0.5,  0.,  0.],
            [0.5,  0.,  0.5],
            [0.5,  0.,  1.],
            [0.5,  0.5,  0.],
            [0.5,  0.5,  0.5],
            [0.5,  0.5,  1.]
        ], dtype=np.float32
    )
    sample_cells = np.array(
        [
            [0, 9, 12, 3, 1, 10, 13, 4],
            [1, 10, 13, 4, 2, 11, 14, 5],
            [3, 12, 15, 6, 4, 13, 16, 7],
            [4, 13, 16, 7, 5, 14, 17, 8],
            [9, 18, 21, 12, 10, 19, 22, 13],
            [10, 19, 22, 13, 11, 20, 23, 14],
            [12, 21, 24, 15, 13, 22, 25, 16],
            [13, 22, 25, 16, 14, 23, 26, 17]
        ]
    )
    sample_cells_dict = {12: sample_cells}
    return sample_points, sample_cells, pv.UnstructuredGrid(sample_cells_dict, sample_points)


class test_Geometry(unittest.TestCase):
    def test_from_nodes_elements(self):
        sample_points, sample_cells, grid = build_sample_grid()

        start_time = datetime.now()
        geo = Geometry()
        geo.from_nodes_elements(sample_points, sample_cells)
        end_time = datetime.now()
        exe_time = runtime(start_time, end_time)

        # check for expected results
        # total number of points
        self.assertEqual(grid.n_points, geo.mesh.n_points)
        # total number of cells
        self.assertEqual(grid.n_cells, geo.mesh.n_cells)
        # check for expected elapsed time
        self.assertLessEqual(exe_time, 2.0)

    def test_from_xplt(self):
        _, _, grid = build_sample_grid()

        start_time = datetime.now()
        geo = Geometry()
        geo.from_xplt(TEST_FILES_DIR/"sample.xplt")
        end_time = datetime.now()
        exe_time = runtime(start_time, end_time)

        # check for expected results
        # total number of points
        self.assertEqual(grid.n_points, geo.mesh.n_points)
        # total number of cells
        self.assertEqual(grid.n_cells, geo.mesh.n_cells)
        # total number of states
        self.assertEqual(geo.states.n(), 11)
        # avaiable states (keys)
        self.assertSetEqual({DATA_FIELDS.DISPLACEMENT, DATA_FIELDS.STRESS},
                            geo.states.keys())
        # state data format
        self.assertEqual(
            geo.states.data_format[DATA_FIELDS.DISPLACEMENT], DATA_FORMAT.NODES)
        self.assertEqual(
            geo.states.data_format[DATA_FIELDS.STRESS], DATA_FORMAT.CELLS)
        # state data shape
        self.assertTupleEqual(geo.states.get(
            DATA_FIELDS.DISPLACEMENT).shape, (11, 27, 3))
        self.assertTupleEqual(geo.states.get(
            DATA_FIELDS.STRESS).shape, (11, 8, 6))

        # check for expected elapsed time
        self.assertLessEqual(exe_time, 20.0)

    def test_get(self):
        # start timer
        start_time = datetime.now()
        # ---
        # build geometry
        geo = Geometry()
        geo.from_xplt(TEST_FILES_DIR/"sample.xplt")
        # extract nodes, cells and state data
        nodes = geo.get(GEO_DATA.NODES)
        cells = geo.get(GEO_DATA.CELLS)
        disp = geo.get(GEO_DATA.STATES, DATA_FIELDS.DISPLACEMENT)
        stress = geo.get(GEO_DATA.STATES, DATA_FIELDS.STRESS)
        # extract step data
        disp_2 = geo.get(GEO_DATA.STATES, DATA_FIELDS.DISPLACEMENT, i=2)
        disp_3 = geo.get(GEO_DATA.STATES, DATA_FIELDS.DISPLACEMENT, i=-1)
        stress_2 = geo.get(GEO_DATA.STATES, DATA_FIELDS.STRESS, t=0)
        stress_3 = geo.get(GEO_DATA.STATES, DATA_FIELDS.STRESS, t=3)
        # ---
        # end timer
        end_time = datetime.now()
        exe_time = runtime(start_time, end_time)

        # check for expected results
        self.assertEqual(len(nodes), geo.mesh.n_points)
        self.assertEqual(len(cells[12]), geo.mesh.n_cells)
        # state data shape
        self.assertTupleEqual(disp.shape, (11, 27, 3))
        self.assertTupleEqual(stress.shape, (11, 8, 6))
        # step data shape
        self.assertTupleEqual(disp_2.shape, (27, 3))
        self.assertTupleEqual(disp_3.shape, (27, 3))
        self.assertTupleEqual(stress_2.shape, (8, 6))
        self.assertTupleEqual(stress_3.shape, (8, 6))

        # check for NOT expected results
        with self.assertRaises(ValueError):
            geo.get(42)
        with self.assertRaises(ValueError):
            geo.get("42")
        with self.assertRaises(ValueError):
            geo.get(GEO_DATA.STATES, key=None)
        with self.assertRaises(TypeError):
            geo.get(GEO_DATA.STATES, key=928)
        with self.assertRaises(KeyError):
            geo.get(GEO_DATA.STATES, key="NOT_IN_STATES")
        with self.assertRaises(TypeError):
            geo.get(GEO_DATA.STATES, key=DATA_FIELDS.DISPLACEMENT, i="NOT_NUMBER")
        with self.assertRaises(TypeError):
            geo.get(GEO_DATA.STATES, key=DATA_FIELDS.DISPLACEMENT, i=[])
        with self.assertRaises(TypeError):
            geo.get(GEO_DATA.STATES, key=DATA_FIELDS.DISPLACEMENT, t="NOT_NUMBER")
        with self.assertRaises(TypeError):
            geo.get(GEO_DATA.STATES, key=DATA_FIELDS.DISPLACEMENT, t={})
        with self.assertRaises(ValueError):
            geo.get(GEO_DATA.STATES, key=DATA_FIELDS.DISPLACEMENT, t=-1)

        # check for expected elapsed time
        self.assertLessEqual(exe_time, 20.0)
