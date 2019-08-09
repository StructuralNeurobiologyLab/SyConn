from numba import jit
import numpy as np
from numba import types
#from numba.typed import Dict



@jit(nopython=True)
def semseg2mesh_sub2(index_views, semseg_view, index_background, length, classes, arr):
    for ii in range(len(index_views)):
        vertex_ix = index_views[ii]
        if vertex_ix == index_background:
            continue
        l = semseg_view[ii]  # vertex label
        arr[vertex_ix][l] += 1
    return arr