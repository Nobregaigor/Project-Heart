from scipy.spatial.transform import Rotation as tr
from project_heart.utils.vector_utils import *


def get_rotation(from_vec, to_vector):
    # USING APPROACH FROM:
    # https://bit.ly/2W9gNb5

    # copy vectors so we dont modify them directly
    from_vec = np.copy(from_vec)
    to_vector = np.copy(to_vector)

    # Make unit vector
    to_vector = unit_vector(to_vector)
    from_vec = unit_vector(from_vec)

    v = np.cross(from_vec, to_vector)  # cross product
    s = np.abs(v)  # sine of angle
    c = np.dot(from_vec, to_vector)  # cosine of angle
    # get skew-symmetric cross-product matrix of v
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    # compute rotation matrix
    rot_matrix = np.identity(3) + vx + vx**2 * (1/(1+c))

    # create rotation object from rotation matrix
    rot = tr.from_matrix(rot_matrix)
    return rot


def centroid(points, ql=0.01, qh=0.99):

    # remove outliers in x, y and z directions
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    low_x = np.quantile(x, ql)
    high_x = np.quantile(x, qh)
    low_y = np.quantile(y, ql)
    high_y = np.quantile(y, qh)
    low_z = np.quantile(z, ql)
    high_z = np.quantile(z, qh)
    filter = np.where((x >= low_x) & (x <= high_x) &
                      (y >= low_y) & (y <= high_y) &
                      (z >= low_z) & (z <= high_z)
                      )[0]
    bound_points = points[filter]

    # compute centroid based on mean of extremas
    x = bound_points[:, 0]
    y = bound_points[:, 1]
    z = bound_points[:, 2]

    c = np.zeros(3)
    c[0] = (np.max(x) + np.min(x)) * 0.5
    c[1] = (np.max(y) + np.min(y)) * 0.5
    c[2] = (np.max(z) + np.min(z)) * 0.5

    return c
