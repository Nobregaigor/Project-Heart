from enum import IntEnum

# =============================================================================
# VKT ELEMENTS


class VTK_ELEMENTS(IntEnum):
    # Linear cells
    EMPTY_CELL = 0
    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    PENTAGONAL_PRISM = 15
    HEXAGONAL_PRISM = 16

    # Quadratic, isoparametric cells
    QUADRATIC_EDGE = 21
    QUADRATIC_TRIANGLE = 22
    QUADRATIC_QUAD = 23
    QUADRATIC_POLYGON = 36
    QUADRATIC_TETRA = 24
    QUADRATIC_HEXAHEDRON = 25
    QUADRATIC_WEDGE = 26
    QUADRATIC_PYRAMID = 27
    BIQUADRATIC_QUAD = 28
    TRIQUADRATIC_HEXAHEDRON = 29
    TRIQUADRATIC_PYRAMID = 37
    QUADRATIC_LINEAR_QUAD = 30
    QUADRATIC_LINEAR_WEDGE = 31
    BIQUADRATIC_QUADRATIC_WEDGE = 32
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
    BIQUADRATIC_TRIANGLE = 34

    # Cubic, isoparametric cell
    CUBIC_LINE = 35

    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET = 41

    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON = 42

    # Higher order cells in parametric form
    PARAMETRIC_CURVE = 51
    PARAMETRIC_SURFACE = 52
    PARAMETRIC_TRI_SURFACE = 53
    PARAMETRIC_QUAD_SURFACE = 54
    PARAMETRIC_TETRA_REGION = 55
    PARAMETRIC_HEX_REGION = 56

    # Higher order cells
    HIGHER_ORDER_EDGE = 60
    HIGHER_ORDER_TRIANGLE = 61
    HIGHER_ORDER_QUAD = 62
    HIGHER_ORDER_POLYGON = 63
    HIGHER_ORDER_TETRAHEDRON = 64
    HIGHER_ORDER_WEDGE = 65
    HIGHER_ORDER_PYRAMID = 66
    HIGHER_ORDER_HEXAHEDRON = 67

    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    LAGRANGE_CURVE = 68
    LAGRANGE_TRIANGLE = 69
    LAGRANGE_QUADRILATERAL = 70
    LAGRANGE_TETRAHEDRON = 71
    LAGRANGE_HEXAHEDRON = 72
    LAGRANGE_WEDGE = 73
    LAGRANGE_PYRAMID = 74

    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE = 75
    BEZIER_TRIANGLE = 76
    BEZIER_QUADRILATERAL = 77
    BEZIER_TETRAHEDRON = 78
    BEZIER_HEXAHEDRON = 79
    BEZIER_WEDGE = 80
    BEZIER_PYRAMID = 81

# =============================================================================
# Number of points in each element type --> NEED FURTHER IMPLEMENTATION


class N_PTS_IN_ELEMENT(IntEnum):
    EMPTY = 0
    TRIANGLE = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 8
    QUADRATIC_HEXAHEDRON = 20

# =============================================================================
# Conversion between length(element) to its type


N_PTS_TO_VTK_ELTYPE = dict(
    (
        (N_PTS_IN_ELEMENT.EMPTY, VTK_ELEMENTS.EMPTY_CELL),
        (N_PTS_IN_ELEMENT.TRIANGLE, VTK_ELEMENTS.TRIANGLE),
        (N_PTS_IN_ELEMENT.TETRAHEDRON, VTK_ELEMENTS.TETRA),
        (N_PTS_IN_ELEMENT.HEXAHEDRON, VTK_ELEMENTS.HEXAHEDRON),
        (N_PTS_IN_ELEMENT.QUADRATIC_HEXAHEDRON, VTK_ELEMENTS.QUADRATIC_HEXAHEDRON),
    )
)
