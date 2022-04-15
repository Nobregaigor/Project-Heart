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


def radius(points, center=None):
    if center is None:
        center = centroid(points)
    return np.mean(np.linalg.norm(points - center, axis=1))


def get_p_along_line(k:float, line):
    """
      Returns a point (x,y,z) at a relative distance k from the
      start of a line defined with two boundary points by interpolation. 

      The default settings returns a point in a relative distance from the 
      apex in the along the longitudinal axis.

      Example: k=0.5 returns the midpoint of a given line.
    """
    if not isinstance(k, float):
        raise TypeError('k must be a float')
    if not isinstance(line, np.ndarray):
        raise TypeError('line must be a numpy array')
        
    # allocate array data
    data = np.zeros(3, dtype=np.float32)
    # interpolate x, y and z in k from 0 to 1 between two boundaries
    # for i in range(3):
        # data[i] = np.interp(k, [0.0, 1.0], [line[0][i], line[1][i]])
    data[0] = np.interp(k, [0.0, 1.0], [line[0][0], line[1][0]])
    data[1] = np.interp(k, [0.0, 1.0], [line[0][1], line[1][1]])
    data[2] = np.interp(k, [0.0, 1.0], [line[0][2], line[1][2]])
    return data
