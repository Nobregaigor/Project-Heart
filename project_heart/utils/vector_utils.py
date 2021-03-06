import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def unit_vector_arr(arr):
    """Returns the unit vector of each row of an nx3 array

    Args:
        arr ([array]): [n rows x 3 columns array corresponding to (x,y,z)]

    Returns:
        [array]: [array of shape shape as input with each row a unit vector]
    """
    return np.divide(arr.T, np.linalg.norm(arr, axis=1) + 1e-10).T


def check_angle_orientation(angle, a, b, zaxis=[0., 0., 1.]):
    # make sure angle is in range [0, 2*pi)
    # https://bit.ly/3nUrr0U
    z = np.asarray(zaxis)
    det = np.linalg.det(np.vstack((a.T, b.T, z.T)))
    if det < 0:
        angle = 2*np.pi - angle
    return angle


def angle_between(a, b, assume_unit_vector=False, check_orientation=True, zaxis=[0., 0., 1.]):

    if isinstance(a, list):
        a = np.asarray(a)

    if isinstance(b, list):
        b = np.asarray(b)

    if a.shape != b.shape:
        if b.shape == (3,) or b.shape == (2,):
            b = np.repeat(np.expand_dims(b, 1), len(a), axis=1).T
        # if a.shape[1] != b.shape[1]:
        else:
            raise ValueError(
                "a and b must be of same size at second dimension -> will compute the row-wise angle in between the two vector arrays.")

    single_element = False
    if len(a.shape) == 1:
        single_element = True
        a = np.expand_dims(a, 0)
        b = np.expand_dims(b, 0)

    if not assume_unit_vector:
        a = unit_vector_arr(a)
        b = unit_vector_arr(b)

    dot = np.sum(a*b, axis=1)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    if check_orientation:
        zaxis = np.asarray(zaxis)
        if zaxis.shape == (3,) or zaxis.shape == (2,):
            zaxis = np.repeat(np.expand_dims(zaxis, 1), len(a), axis=1).T
        angle = np.array(list(map(
            check_angle_orientation,
            angle, a, b, zaxis)
        ))

    if not single_element:
        return angle
    else:
        return angle[0]


def fit_plane(points: np.ndarray) -> tuple:
    """Creates a plane from a list of points.

    Args:
        points (np.ndarray): A numpy array of points of [[x,y,z]...] coordinates.

    Returns:
        tuple: [normal, plane d constant]
    """
    if not isinstance(points, np.ndarray):
        try:
            points = np.asarray(points, dtype=np.float64)
        except:
            raise(ValueError("Points must be able to be converted to a np.array"))
    P_mean = points.mean(axis=0)
    P_centered = points - P_mean
    U, s, V = np.linalg.svd(P_centered)
    normal = V[2, :]
    d = -np.dot(P_mean, normal)
    return (normal, d)


def generate_circle_by_vectors(t: np.ndarray, C: np.ndarray, r: float, n: np.ndarray, u: np.ndarray, dtype: np.dtype = np.float64) -> np.ndarray:
    """Creates a set of points based on 't' angle list (one point for each angle in t) at center \
        'C' with radius 'r' and normal 'n' and direction 'u' (othogonal to 'n')

    Args:
        t (np.ndarray): List of angles in which points should be generated (in radians).
        C (np.ndarray): Center of circle. Array should be [x,y,z]
        r (float): Radius of circle.
        n (np.ndarray): Vector normal of circle [x,y,z].
        u (np.ndarray): Vector cross-normal of circle. Orthogonal vector to 'n' [x,y,z].
        dtype (np.dtype): Numpy dtype of the returned array.

    Returns:
        np.ndarray: List of points as [[x,y,z]...] np.darray
    """
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:, np.newaxis]*u + r * \
        np.sin(t)[:, np.newaxis]*np.cross(n, u) + C
    return P_circle.astype(dtype)


def calc_plane_d(normal, v):
    """
      Returns the "d" constant of the equation of the plane
      with a reference point 'v' by using dot product.
    """
    # # get point 'v' if not defined
    # if v is None:
    #   v = get_p_along_line(k, line)
    d = -np.dot(normal[:3], v)
    return d

def distance(p1, p2, dtype=np.float64):
    if not isinstance(p1, np.ndarray):
        p1 = np.array(p1, dtype=dtype)
    if not isinstance(p2, np.ndarray):
        p2 = np.array(p2, dtype=dtype)
    return np.linalg.norm(p1-p2).astype(dtype)

def dist_from_plane(pts, normal, d, abs=True):
    """ Returns the perpendicular distance from point p and plane"""
    # print(p)
    a = np.dot(pts, normal)
    if abs:
        return np.abs(a + d) / np.linalg.norm(normal)
    else:
        return a + d / np.linalg.norm(normal)

def project_pts_onto_plane(pts, plane_normal, plane_d):
    plane_normal = unit_vector(plane_normal)
    dists = dist_from_plane(pts, plane_normal, plane_d, abs=False).reshape((-1,1))
    return pts - dists*plane_normal

def project_vec_a_onto_vec_b(a,b, unit=False):
    b_norm = np.sqrt(sum(b**2))    
    pr_a = (np.dot(a, b)/b_norm**2)*b
    if not unit:
        return pr_a
    else:
        return unit_vector(pr_a)

def dist_from_line(p1:np.ndarray, p2:np.ndarray, p3:np.ndarray, dtype:np.dtype=np.float64) -> np.ndarray:
    """Computes the perpendicular distance between one or multiple points [p1] to a line (or lines) \
        defined by [p2, p3]. p2 and p3 must have same shape. points are defined as: [x,y,z].

    Args:
        p1 (np.ndarray): Reference point(s)
        p2 (np.ndarray): First point on line(s)
        p3 (np.ndarray): Second point on line(s)
        dtype (np.dtype, optional): Values are converted to numpy array. This denies the output type. Defaults to np.float64.

    Returns:
        float or np.ndarray: Perpendicular distance(s). If singular point and line, will return float. Otherwise will return an array.
    """
    assert len(p2) == len(p3), "p2 and p3 must have same number of points as they represent lines. Received: p2 ({}), p3 ({})".format(len(p2), len(p3))
    if not isinstance(p1, np.ndarray):
        p1 = np.asarray(p1, dtype=dtype)
        assert p1.shape[-1] == 3, "Point must be composed of [x,y,z]. Received shape (p1): {}".format(p1.shape)
    if not isinstance(p2, np.ndarray):
        p2 = np.asarray(p2, dtype=dtype)
        assert p2.shape[-1] == 3, "Point must be composed of [x,y,z]. Received shape (p2): {}".format(p2.shape)
    if not isinstance(p3, np.ndarray):
        p3 = np.asarray(p3, dtype=dtype)
        assert p3.shape[-1] == 3, "Point must be composed of [x,y,z]. Received shape (p3): {}".format(p3.shape)

    dists = np.linalg.norm(np.cross(p3-p2, p2-p1, axis=-1), axis=-1)/np.linalg.norm(p3-p2, axis=-1)
    # dists = np.linalg.norm(np.cross(p2-p1, p1-p3, axis=-1), axis=-1)/np.linalg.norm(p2-p1, axis=-1)
    return dists.astype(dtype)


def get_pts_close_to_plane(points, maxd, normal, v, return_mask=False):
    """
      Returns points close to a distance 'maxd' along a plane defined
      by a 'normal' vector and a point 'v' on plane
    """
    # compute constant 'd' of a plane
    d = calc_plane_d(normal, v)

    # compute dist
    dists = dist_from_plane(points, normal, d)

    return np.where(dists <= maxd)[0]
