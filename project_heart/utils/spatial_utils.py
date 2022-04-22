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


def get_p_along_line(k: float, line):
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


# ==================================
# GROUPING
# ==================================

def grouping_by_distance(xyz, w=1.0, assume_sorted=False, decay=0.95, min_g=4, sort_mode="vertical", max_trials=100):
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

    mean_last_two = np.mean(xyz[-2:], axis=0)
    if np.linalg.norm(mean_last_two-data[-1]) <= np.linalg.norm(data[-1]-data[-2])*w:
        data = np.vstack((data, mean_last_two))

    if np.linalg.norm(xyz[0]-data[0]) <= np.linalg.norm(data[0]-data[1])*w:
        data = np.vstack((xyz[0], data))
    return data


def group_by_z(xyz, zaxis=np.asarray([0., 0., 1.])):

    buckets = deque([deque() for _ in range(n_subsets)])

    # set values to be digitized. The final values will have the similar meaning
    # as a 2D view on xz as: y = sign(x)*y. This method allows for quickly
    # approximate left and right sides of the plane without the need to sort
    # note: y values must be positive.
    zs = pts[:, 2]
    angle_y = angle_between_2D(np.cross(zaxis, normal)[:2], YAXIS[:2])
    if abs(angle_y) <= np.radians(45) or abs(angle_y) >= np.radians(135):
        ys = pts[:, 1]
        arr = np.sign(ys)*(zs-np.min(zs))
    else:
        xs = pts[:, 0]
        arr = np.sign(xs)*(zs-np.min(zs))
    # get ranges of of modified z-axis and compute bins
    min_z, max_z = np.min(arr), np.max(arr)
    bins = np.digitize(arr, np.linspace(min_z, max_z+1, n_subsets+1))

    # The previous method works for most scenarios. However, there are some
    # limitation when deciding which bin the bottom nodes belong to.
    # Let's check if any modification needs to be made. The heuristic will
    # be based on the distance between a given node and the median value of
    # its bin and the opposite bin. If the distance between the opposite bin
    # is less than of the one to its bin's median, it probably should be
    # in the opposite bin.

    # simply get the left and right ids (bucket idexes) of bottom bins
    # They sould be the two middle bins.
    right_id = n_subsets//2
    left_id = n_subsets//2+1

    # get indexes of right and left ids (so that we can refer to them later)
    right_idexes = np.argwhere(bins == right_id).reshape((-1,))
    left_idexes = np.argwhere(bins == left_id).reshape((-1,))

    # get right and left pts
    right_pts = pts[right_idexes][:, :2]
    left_pts = pts[left_idexes][:, :2]

    # compute median of each bin
    right_median = np.median(right_pts, axis=0)
    left_median = np.median(left_pts, axis=0)

    # compute the distance from each node to it's current bin's median
    d_r_to_m = np.linalg.norm((right_pts-right_median), axis=1)
    d_l_to_m = np.linalg.norm((left_pts-left_median), axis=1)

    # compute the distance from each node to the opposite bin's median
    d_r_to_l = np.linalg.norm((right_pts-left_median), axis=1)
    d_l_to_r = np.linalg.norm((left_pts-right_median), axis=1)

    # For each distance, if the distance between the opposite bin and the
    # current bin is shorter than the one from the current bin, the node
    # belongs to the opposite bin.
    for i, (drm, drl) in enumerate(zip(d_r_to_m, d_r_to_l)):
        if drl < drm:
            bins[right_idexes[i]] = left_id
    for i, (dlm, dlr) in enumerate(zip(d_l_to_m, d_l_to_r)):
        if dlr < dlm:
            bins[left_idexes[i]] = right_id

    # add ids to each bucket
    for i, pool_idx in enumerate(bins):
        buckets[pool_idx-1].append(valid_ids[i])

    return buckets

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


def line_sum(coords):
    return np.sum(np.linalg.norm(coords[1:]-coords[:-1], axis=1))

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
    