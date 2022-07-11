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

def align_vec_axis(vec:np.ndarray, ref_axis:np.ndarray, new_axis:np.ndarray, unit:bool=False) -> np.ndarray:
    """Aligns a vector defined at standard axis (X: [1,0,0], Y: [0,1,0], Z: [0,0,1]) to \
        nont-standard coordinates based on a nre reference axis. 
   
    Args:
        vec (np.ndarray): _description_
        ref_axis (np.ndarray): _description_
        unit (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    rot = get_rotation(ref_axis, new_axis)
    if not unit:
        return rot.apply(vec)
    else:
        return unit_vector(rot.apply(vec))

def centroid(points, filter=True, ql=0.01, qh=0.99):

    if filter:
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
        if len(bound_points) == 0:
            bound_points = points
    else:
        bound_points = points
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


def get_p_along_line(k: float, line, extrapolate=False):
    """
      Returns a point (x,y,z) at a relative distance k from the
      start of a line defined with two boundary points by interpolation. 

      The default settings returns a point in a relative distance from the 
      apex in the along the longitudinal axis.

      Example: k=0.5 returns the midpoint of a given line.
    """
    
    if not isinstance(k, (int, float)):
        raise TypeError('k must be a float')
    if not isinstance(line, np.ndarray):
        # raise TypeError('line must be a numpy array')
        line = np.asarray(line)

    # allocate array data
    data = np.zeros(3, dtype=np.float64)
    # interpolate x, y and z in k from 0 to 1 between two boundaries
    # for i in range(3):
    # data[i] = np.interp(k, [0.0, 1.0], [line[0][i], line[1][i]])
    if not extrapolate:
        data[0] = np.interp(k, [0.0, 1.0], [line[0][0], line[1][0]])
        data[1] = np.interp(k, [0.0, 1.0], [line[0][1], line[1][1]])
        data[2] = np.interp(k, [0.0, 1.0], [line[0][2], line[1][2]])
    else:
        from scipy.interpolate import interp1d
        data[0] = interp1d([0.0, 1.0], [line[0][0], line[1][0]], fill_value="extrapolate")(k)
        data[1] = interp1d([0.0, 1.0], [line[0][1], line[1][1]], fill_value="extrapolate")(k)
        data[2] = interp1d([0.0, 1.0], [line[0][2], line[1][2]], fill_value="extrapolate")(k)
    return data

# ==================================
# cartesian/polar transformation
# ==================================


def cart2pol(xy, dtype=np.float32):
    rho = np.sqrt(xy[0]**2 + xy[1]**2)
    phi = np.arctan2(xy[1], xy[0])
    return np.array([rho, phi], dtype=dtype)


def pol2cart(rho_phi, dtype=np.float32):
    x = rho_phi[0] * np.cos(rho_phi[1])
    y = rho_phi[0] * np.sin(rho_phi[1])
    return np.array([x, y], dtype=dtype)

# ==================================
# Sorting
# ==================================


def sort_by_angles_2D(xy, vec=[1., 0.], return_indexes=False):
    xy_mean = np.mean(xy)
    angles = angle_between(xy - xy_mean, vec, check_orientation=False)
    # angles = np.asarray([angle_between_2D(p[:2] - xy_mean, vec) for p in xy])

    max_angle, min_anlge = np.max(angles), np.min(angles)
    if max_angle >= 0.75*np.pi and min_anlge <= -0.75*np.pi:
        # to_sort = np.sin(angles)
        angles += 2*np.pi
    # to_sort = np.cos(angles)

    to_sort = angles
    if return_indexes:
        return np.argsort(to_sort)
    else:
        return xy[np.argsort(to_sort)]


def sort_by_polar(xy):
    polar = np.array([cart2pol(v) for v in xy], dtype=np.float32)
    # check for angle discontinuity
    phis = polar[:, 1]
    max_angle, min_anlge = np.max(phis), np.min(phis)
    if max_angle > 0.95*np.pi and min_anlge < -0.95*np.pi:
        phis[phis < 0] += 2*np.pi
    polar[:, 1] = phis
    # sort by angle, magnitude
    lexsort = np.lexsort((polar[:, 1], polar[:, 0]))
    return xy[lexsort]


def sort_circumferential_2D(xy, vec=np.asarray([1.0, 0.0])):
    xy_mean = np.mean(xy)
    angles = angle_between(xy - xy_mean, vec, check_orientation=False)
    # angles = np.asarray([angle_between_2D(p[:2] - xy_mean, vec) for p in xy])
    idx = np.lexsort([angles, xy[:, 1], xy[:, 0]])
    return xy[idx]


def sort_by_spherical(xyz, order="phi_theta_r"):
    # transform to spherical coordinates
    rs = np.linalg.norm(xyz, axis=1)
    thetas = np.arccos(xyz[:, 2]/rs)
    phis = np.arctan2(xyz[:, 1],xyz[:, 0])
    # sort by columns
    if order == "phi_theta_r":
        ids = np.lexsort((phis, thetas, rs))
    elif order == "phi_r_theta":
        ids = np.lexsort((phis, rs, thetas))
    elif order == "r_phi_theta":
        ids = np.lexsort((rs, phis, thetas))
    elif order == "r_theta_phi":
        ids = np.lexsort((rs, thetas, phis))
    elif order == "theta_r_phi":
        ids = np.lexsort((thetas, rs, phis))
    elif order == "theta_phi_r":
        ids = np.lexsort((thetas, phis, rs))
    else:
        raise ValueError("Order not understood. Options are: "
                         "phi_theta_r, phi_r_theta, "
                         "r_phi_theta, r_theta_phi ",
                         "theta_r_phi, theta_phi_r")
    return xyz[ids], ids

# ==================================
# GROUPING
# ==================================

def grouping_by_distance(xyz, w=1.0, assume_sorted=False, decay=0.95, min_g=4, sort_mode="spherical", 
                         max_trials=100):
    """
      This function simplifies a xyz curve by grouping elements based on their
      distance. 
      The heuristic method is group all elements that are within w * eucledean
      distance between element i and further elements
    """
    # This algorithm needs elements to be sorted by [y,x,z]
    if not assume_sorted:
        if sort_mode == "vertical":
            xyz = xyz[np.lexsort([xyz[:, 1], xyz[:, 0], xyz[:, 2]])]
        if sort_mode == "spherical":
            xyz, _ = sort_by_spherical(xyz)
        elif sort_mode == "circumferential":
            xyz = sort_circumferential_2D(xyz)

    def find_groups(w):
        # set list for grouping elements
        groups = []
        ii = 0
        # start seach
        while ii != len(xyz):
            # for each element, compare the distance with all other elements
            dists = np.linalg.norm(xyz[ii+1:]-xyz[ii], axis=1)
            # search for all elements that are relatively close to given element
            # distance should be <= distance betwen element ii and element ii+1
            search = np.argwhere(dists > dists[0]*w)
            if len(search) > 0:
                # if distance is found, group elements from ii to jj
                jj = search[0][0]
                groups.append(xyz[ii:jj+ii])
                ii += jj
            else:
                break
        # append last group (from lat ii to end of xyz list)
        groups.append(xyz[ii:])
        return groups

    trial = 0
    n_groups = 1
    groups = []
    failed_grouping = False
    if min_g == 0:
        min_g = 1
    while n_groups <= min_g and trial < max_trials:
        groups = find_groups(w)
        w = w*decay
        if w < 1:
            failed_grouping = True
            break
        n_groups = len(groups)
        trial += 1

    if failed_grouping:
        return xyz

    if len(groups) <= min_g:
        raise ValueError(
            "Length of groups is {}. Needed: {}.".format(n_groups, min_g))

    # compute mean elements of data
    data = np.array([np.average(val, axis=0) for val in groups if len(val) > 0],
                    dtype=np.float32)

    # mean_last_two = np.mean(xyz[-2:], axis=0)
    # if np.linalg.norm(mean_last_two-data[-1]) <= np.linalg.norm(data[-1]-data[-2])*w:
    #     data = np.vstack((data, mean_last_two))

    # if np.linalg.norm(xyz[0]-data[0]) <= np.linalg.norm(data[0]-data[1])*w:
    #     data = np.vstack((xyz[0], data))
    return data


# ==================================
# 2d SEG. LENGTH
# ==================================


def compute_segment_length_2D(xyz_data, nps=10, xaxis=1, yaxis=2,
                              sort="yaxis",
                              stack_first=False,
                              stack_last=False,
                              connect_endpoints=False):
    """ Calculates the total segment length based on curve 
        having shape similar to y=a*x^2. 
    """
    # get xy data with specified x and y axis
    xy = np.hstack(
        [xyz_data[:, xaxis].reshape((-1, 1)),
         xyz_data[:, yaxis].reshape(-1, 1)])
    if len(xy) <= 1:
        return 0.0

    # sort based on yaxis
    if sort == "yaxis":
        # xy[:, 1] += -np.min(xy[:, 1]) # make sure yvalues are positive
        # xy = xy[np.argsort(np.sign(xy[:,0])*xy[:,1])]
        _m = np.mean((xy[:, 0], xy[:, 1]), axis=1)
        _d = abs(xy - _m.T)
        _r = np.mean(_d, axis=0)
        if _r > 1.5:
            s = np.lexsort((xy[:, 0], xy[:, 1]))
        else:
            s = np.lexsort((xy[:, 1], xy[:, 0]))
        xy = xy[s]
    elif sort == "polar":
        xy = sort_by_polar(xy)
    elif sort == "angles":
        xy = sort_by_angles_2D(xy)
    else:
        raise ValueError("Sorting method not value.")

    # get first and last data points
    first = xy[0]
    last = xy[-1]
    # split data into segments and compute mean values
    splits = np.array_split(xy, len(xy)//nps)
    # compute mean for each split
    xy = np.array([np.mean(val, axis=0) for val in splits if len(val) > 0],
                  dtype=np.float32)

    # stack first and last element
    if stack_first:
        xy = np.vstack((first, xy))
    if stack_last:
        xy = np.vstack((xy, last))
    if connect_endpoints:
        xy = np.vstack((xy, xy[0]))

    # compute total length
    return np.sum(np.linalg.norm((xy[1:]-xy[:-1]), axis=1))

# ==================================
# OTHER UTILS
# ==================================


def simplify(xy, nps):
    if nps == 0:
        return xy
    if len(xy)//nps == 0 or len(xy) <= 1:
        return xy
    splits = np.array_split(xy, len(xy)//nps)
    return np.array([np.average(val, axis=0) for val in splits if len(val) > 0],
                    dtype=np.float32)


def simplify_on_z(xyz, nps):
    """
      This algorithm computes the means of chunks of xyz data sorted on z axis
    """
    if nps == 0:
        return xyz
    if len(xyz)//nps == 0 or len(xyz) <= 1:
        return xyz
    xyz = xyz[np.argsort(xyz[:, -1])]
    nsplits = len(xyz)//nps
    if nsplits <= 1:
        return xyz
    splits = np.array_split(xyz, len(xyz)//nps)
    # compute mean for each split
    return np.array([np.average(val, axis=0) for val in splits if len(val) > 0],
                    dtype=np.float32)


def swap_xy(xyz, ax1, ax2):
    return np.hstack((xyz[:, ax1].reshape((-1, 1)), xyz[:, ax2].reshape((-1, 1))))


def line_sum(coords, join_ends=False):
    if not join_ends:
        return np.sum(np.linalg.norm(coords[1:]-coords[:-1], axis=1))
    else:
        _coords = np.vstack((coords, coords[0]))
        return np.sum(np.linalg.norm(_coords[1:]-_coords[:-1], axis=1))

# ==================================
# 3D Seg. circunferential length
# ==================================


def compute_circumferential_length(xyz, nps=0.12, nps2=0.25, w=3.0, decay=0.98, fast_decision=True):
    if len(xyz) == 1:
        return 0.0

    if fast_decision:
        d_x = np.max(xyz[:, 0]) - np.min(xyz[:, 0])
        d_y = np.max(xyz[:, 1]) - np.min(xyz[:, 1])
        xy = swap_xy(xyz, 0, 1) if d_x > d_y else swap_xy(xyz, 1, 0)
        xy = sort_circumferential_2D(xy)
    else:
        # try first attempt
        xy1 = swap_xy(xyz, 0, 1)
        xy1 = sort_circumferential_2D(xy1)
        d1 = line_sum(xy1)
        # try second attempt
        xy2 = swap_xy(xyz, 1, 0)
        xy2 = sort_circumferential_2D(xy2)
        d2 = line_sum(xy2)
        # decide best approach
        xy = xy1 if d1 < d2 else xy2

    xy = simplify(xy, int(np.floor(nps*len(xyz))))
    min_g = int(np.ceil(nps2*len(xy)))
    if len(xy) <= min_g:
        return line_sum(xy)

    xy = grouping_by_distance(
        xy, w, decay=decay, min_g=min_g, assume_sorted=True)
    return line_sum(xy)


# ==================================
# 3D Seg. longitudinal length
# ==================================

def compute_longitudinal_length(xyz, nps=0.075, nps2=0.33, w=3.0, decay=0.95, **kwargs):
    if len(xyz) == 0:
        return 0.0

    n_pts_split = int(np.floor(nps*len(xyz)))
    if n_pts_split < 3:
        n_pts_split = 3

    xyz = simplify_on_z(xyz, n_pts_split)
    min_g = int(np.ceil(nps2*len(xyz)))

    xyz = grouping_by_distance(
        xyz, w, decay=decay, min_g=min_g, sort_mode="vertical")
    return line_sum(xyz)


def compute_length_by_clustering(xyz, n_clusters=6, random_state=0, batch_size=5120, **kwargs):
  
    # divide into clusters
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                batch_size = batch_size,
                                random_state=random_state, **kwargs).fit(xyz)

    # get centers
    centers = kmeans.cluster_centers_

    # transform to spherical coordinates
    rs = np.linalg.norm(centers, axis=1)
    thetas = np.arccos(centers[:, 2]/rs)
    phis = np.arctan2(centers[:, 1],centers[:, 0])

    # sort by columns
    ids = np.lexsort((phis, thetas,rs))
    centers = centers[ids]
    
    #compute length
    return line_sum(centers)

def apply_filter_on_line_segment(xyz, 
                            mfilter_ws=0, 
                           sfilter_ws=0, 
                           sfilter_or=0,
                           keep_first=False,
                           keep_last=False):
    def _apply_filter(seq):
        # reduce noise with filters
        new_seq = np.copy(seq)
        if mfilter_ws > 0 and len(seq) > mfilter_ws:
            from scipy import signal
            new_seq = signal.medfilt(seq, mfilter_ws)
            if keep_first:
                new_seq[0] = seq[0]
            if keep_last:
                new_seq[-1] = seq[-1]
        if sfilter_ws > 0 and len(seq) > sfilter_ws:
            from scipy import signal
            new_seq = signal.savgol_filter(seq, sfilter_ws, sfilter_or)
            if keep_first:
                new_seq[0] = seq[0]
            if keep_last:
                new_seq[-1] = seq[-1]
        return seq
            
    xyz[:, 0] = _apply_filter(xyz[:, 0])
    xyz[:, 1] = _apply_filter(xyz[:, 1])
    xyz[:, 2] = _apply_filter(xyz[:, 2])
    
    return xyz


def compute_length_from_predefined_cluster_list(xyz:np.ndarray, 
                                            clusters:list=None, 
                                            assume_sorted:bool=False,
                                            use_centroid=False,
                                            filter_args=None,
                                            join_ends=False,
                                            dtype=np.float64,
                                            **kwargs) -> float:
    assert clusters is not None, "clusters must be provided."
    # compute centers from list of clusters
    if not use_centroid:
        centers = [np.mean(xyz[kids], axis=0) for kids in clusters]
    else:
        centers = [centroid(xyz[kids]) for kids in clusters]
    # transform list of centers to array
    centers = np.asarray(centers, dtype=dtype)    
    # for optmization, we can assume list of clusters is sorted
    if not assume_sorted:
        # transform to spherical coordinates
        rs = np.linalg.norm(centers, axis=1)
        thetas = np.arccos(centers[:, 2]/rs)
        phis = np.arctan2(centers[:, 1],centers[:, 0])
        # sort by columns
        ids = np.lexsort((phis, thetas,rs))
        centers = centers[ids]
    # apply filters (if requested)
    if filter_args is not None:
        centers = apply_filter_on_line_segment(centers, **filter_args)
    #compute length
    return line_sum(centers, join_ends=join_ends)


# ===================================
# Projections
# ===================================

def project_pt_on_line(pt, p1, p2):
    
    l2 = np.sum((p1-p2)**2) # distance between p1 and p2
    t = np.sum((pt - p1) * (p2 - p1)) / l2
    return p1 + t * (p2 - p1)