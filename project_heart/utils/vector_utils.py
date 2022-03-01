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
