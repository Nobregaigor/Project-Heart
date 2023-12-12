import numpy as np


def convert_to_cylindrical_coordinates(
    data: np.ndarray, 
    centers: np.ndarray, 
    ref_pt:np.ndarray=None, 
    x_axis:np.ndarray=None, 
    z_axis:np.ndarray=None,
    dtype:np.dtype=np.float64) -> np.ndarray:
    """Converts data from standard cartesian coordinates to cylindrical coordinates using 
    rotation matrix P as described by https://bit.ly/3BBxlaS, using in-plane angle between
    the vector from reference point to element center and the x axis.

    Args:
        data (np.ndarray): data to be converted to cylindrical coordinate. Must contain 6 elements at last axis.
        centers (np.ndarray): element centers. Must match number of data entry (one center per element)
        ref_pt (np.ndarray, optional): reference point. Defaults to None -> [0,0,0].
        x_axis (np.ndarray, optional): reference x-axis. Defaults to None -> [1,0,0].
        dtype (np.dtype, optional): data type. Defaults to np.float64.

    Returns:
        np.ndarray: data converted into cylindrical coordinates.
    """
    from project_heart.utils.vector_utils import angle_between

    # check data size
    assert data.shape[-1] == 6, AssertionError("Data must contain 6 element at last axis. "
                                               "Received: {}".format(data.shape[-1]))
    assert len(data) == len(centers), AssertionError(
        "Number of element centers must match number of data. "
        "Expected one center per data entry. "
        "Expected: {}, received: {}".format(len(data), len(centers)))

    if ref_pt is None:
        ref_pt = np.asarray([0,0,0])
    if x_axis is None:
        x_axis = np.asarray([1,0,0])
    if z_axis is None:
        z_axis = np.asarray([0,0,1])
    
    # set return variable
    cy_data = np.zeros(data.shape, dtype=dtype)
    
    # convert ref_pt to array
    ref_pt = np.array(ref_pt, dtype=dtype)
    x_axis = np.array(x_axis, dtype=dtype)
    
    u = centers - ref_pt
    
    thetas = angle_between(u[:, :2], x_axis[:2], check_orientation=True)
    # thetas = angle_between(u, x_axis, check_orientation=True, zaxis=z_axis)
    # thetas = angle_between(u, x_axis, check_orientation=False)
    
    sin_t = np.sin(thetas)          # sin(theta)
    cos_t = np.cos(thetas)          # cos(theta)
    sin_t2 = sin_t ** 2             # sin(theta)**2
    cos_t2 = cos_t ** 2             # cos(theta)**2
    sin_cos_t = sin_t * cos_t       # sin(theta)*cos(theta)

    cy_data[:, 0] = data[:, 0]*cos_t2 + data[:, 1]*sin_t2 + 2*data[:, 3]*sin_cos_t          # a_rr
    cy_data[:, 1] = data[:, 0]*sin_t2 + data[:, 1]*cos_t2 - 2*data[:, 3]*sin_cos_t          # a_tt
    cy_data[:, 2] = data[:, 2]                                                              # a_zz
    cy_data[:, 3] = (data[:, 1] - data[:, 0])*sin_cos_t + data[:, 3]*(cos_t2 - sin_t2)      # a_rt = a_tr
    cy_data[:, 4] = -data[:, 5]*sin_t + data[:, 4]*cos_t                                    # a_tz = a_zt
    cy_data[:, 5] = data[:, 5]*cos_t + data[:, 4]*sin_t                                     # a_rz = a_zr
        
    return cy_data