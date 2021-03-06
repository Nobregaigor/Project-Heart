from enum import IntEnum


class GEO_DATA(IntEnum):
    POINTS = 0
    POINT = 0
    NODES = 0
    NODE = 0

    ELEMS = 1
    ELEM = 1
    ELEMENTS = 1
    ELEMENT = 1
    CELLS = 1
    CELL = 1

    STATES = 3
    STATE = 3

    MESH_POINT_DATA = 4
    MESH_CELL_DATA = 5

    # surface mesh
    SURF_POINT_DATA = 6
    SURF_MESH_DATA = 7
    FACET_DATA = 7
