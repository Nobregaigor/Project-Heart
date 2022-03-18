import numpy as np
from project_heart.utils.vector_utils import *
from collections import deque
import pyvista as pv


def create_rim_circunference(
    center: np.ndarray,
    radius: float,
    height: float,
    normal: np.ndarray,
    cross_normal: np.ndarray,
    radial_resolution: int = 64,
    rim_angles: np.ndarray = None
) -> tuple:
    """Creates a set of points organized as a 'rim' structure (two stacked circles) \
        at given center, radius, height, normal (normal to rim plane) and cross_normal \
        (orthonomal vector to normal describing the direction of rim plane). \
        If rim_angles is provided, radial_resolution will be ignored.

    Args:
        center (np.ndarray): Coordinates [x,y,z] of rim's center.

        radius (float): Rim's radius.

        height (float): Height of rim. Top and bottom rim planes will be 0.5 * height \
          above and below center respectively.

        normal (np.ndarray): Normal to rim's plane.

        cross_normal (np.ndarray): Orthonomal vector to normal (parallel to rim's plane).
        radial_resolution (int, optional): Number of points along the rim's circumference \
          (for top and bottom circles). If rim_angles is provided, radial_resolution will \
            be ignored. Defaults to 64.

        rim_angles (np.ndarray, optional): List of angles in which rim points will be generates. \
          If not provided (set to None), will compute radial_resolution. Defaults to None.

    Raises:
        ValueError: If rim_angle is provided but is not a numpy array or \
          if it cannot be converted to a numpy array.

    Returns:
        tuple: (nodes (np.ndarray), center (np.ndarray), elements (np.ndarray))
    """

    # set top and bottom centers
    c1 = center - normal*height*0.5
    # c2 = center + normal*height*0.5

    # create angles
    if rim_angles is None:
        rim_angles = np.linspace(0, 2*np.pi, radial_resolution+1)[:-1]
    else:
        if not isinstance(rim_angles, np.ndarray):
            try:
                rim_angles = np.asarray(rim_angles, dtype=np.float64)
            except:
                raise ValueError(
                    "rim_angles must be able to be converted to np.ndarray.")
        else:
            raise ValueError(
                "If rim_angles is provided, it must be either a list of float or a np.ndarray of shape [n]")

    # create points around circle at given conditions
    rim = generate_circle_by_vectors(rim_angles, c1, radius,
                                     normal, cross_normal)

    # create second circle by 'extruding' from first set of points
    rim_2 = rim + height*normal

    # set elements
    elements = deque()
    n_nodes = len(rim)
    for i, j in zip(range(0, n_nodes), range(n_nodes, n_nodes*2-1)):
        elements.append((
            i, j, j+1, i+1
        ))
    elements = np.array(elements, dtype=np.int64)

    # stack nodes
    rim = np.vstack([rim, rim_2])

    return rim, center, elements


def lines_from_points(points: np.ndarray) -> pv.PolyData:
    """Given an array of points as [[x1,y1,z1], [x2,y2,z2], ...], make a line set.

    Args:
        points (np.ndarray): Array of points as [[x1,y1,z1], [x2,y2,z2], ...].

    Returns:
        pv.PolyData: Pyvista PolyData dataset.
    """
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells

    return poly
