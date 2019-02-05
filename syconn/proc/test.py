import numpy as np
#import numba
from numba import jit
import timeit


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




def create_toy_data(size1, size2, moduloo):

    matrix = np.zeros((size1,size2), dtype=int)
    for i in range(size1):
        for j in range(size2):
            matrix[i, j] = np.random.randint(low = 0, high = moduloo)
    return matrix


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


coords = create_toy_data(512,3,100)
bounding_box = create_toy_data(2,3, 10)

print ("in_bounding_box__Numba")
wrapped = wrapper(in_bounding_box, coords, bounding_box)
print (timeit.timeit(wrapped, number=1000000))