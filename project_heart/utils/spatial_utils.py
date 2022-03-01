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
