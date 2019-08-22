from numba import jit
import numpy as np

@jit
def in_bounding_box(coords, bounding_box):
    """
    Loop version with numba

    Parameters
    ----------
    coords : np.array (N x 3)
    bounding_box : tuple (np.array, np.array)
        center coordinate and edge lengths of bounding box

    Returns
    -------
    np.array of bool
        inlying coordinates are indicated as true
    """
    edge_sizes = bounding_box[1] / 2
    coords = np.array(coords) - bounding_box[0]
    inlier = np.zeros((len(coords)), dtype=np.bool)
    for i in range(len(coords)):
        x_cond = (coords[i, 0] > -edge_sizes[0]) & (coords[i, 0] < edge_sizes[0])
        y_cond = (coords[i, 1] > -edge_sizes[1]) & (coords[i, 1] < edge_sizes[1])
        z_cond = (coords[i, 2] > -edge_sizes[2]) & (coords[i, 2] < edge_sizes[2])
        inlier[i] = x_cond & y_cond & z_cond
    return inlier
