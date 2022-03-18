import scipy
import numpy as np


def query_kdtree(A: np.ndarray, B: np.ndarray) -> tuple:
    """Performs a kd-query of B on A; in which A is the kd-tree. \
        Returns the distances and indexes of closest points of B w.r.t. A.

    Args:
        A (np.ndarray): Could points where search will be performed.
        B (np.ndarray): Cloud points used to search.

    Returns:
        tuple: (distances, indexes)
    """
    tree = scipy.spatial.cKDTree(A)
    return tree.query(B)


def relate_closest(A: np.ndarray, B: np.ndarray) -> tuple:
    """Relates closest points of A w.r.t. to B by performing a kd-query.

    Args:
        A (np.ndarray): Reference nodes (where relation will start).
        B (np.ndarray): Other nodes (where relationw will end).

    Returns:
        tuple: (Idxs relationships [A, B], (np.ndarray), distances (np.ndarray))
    """
    from_nodes = np.arange(len(A)).reshape(-1, 1)
    dists, to_nodes = query_kdtree(B, A)
    to_nodes = to_nodes.reshape(-1, 1)
    return np.hstack((from_nodes, to_nodes)), dists
