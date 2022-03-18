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

    if a.shape != b.shape:
        raise ValueError(
            "a and b must be of same size -> will compute the row-wise angle in between the two vector arrays.")

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
        angle = np.array(list(map(
            check_angle_orientation,
            angle, a, b)
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
