from enum import IntEnum


class DATA_FORMAT(IntEnum):
    UNKNOWN = -1

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

    OTHER = 3
