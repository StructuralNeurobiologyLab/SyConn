from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views, render_sso_coords_generic
import time
import itertools
from syconn import global_params
from syconn.handler.basics import chunkify_successive


class data:
    ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')
    ssv = ssc.get_super_segmentation_object(29753344)
    def __init__(self):
        ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')
        ssv = ssc.get_super_segmentation_object(29753344)
        #self.ssc = ssc.get_super_segmentation_object(29753344)
        exloc = np.array([5602, 4173, 4474]) * ssv.scaling
        self.exlocs = np.concatenate(ssv.sample_locations())


def sort_ssv_descending(ssv_array):
    length_array = []
    index_array = []
    for ssv_object in ssv_array:
        length_array.append(len(np.concatenate(ssv_object.sample_locations())))
    index_array = sorting_in_descending_order(length_array)
    return index_array


def sorting_in_descending_order(array):
    index_array = []
    for i in range(len(array)):
        index_array.append(i)
    for iter_num in range(len(array) - 1, 0, -1):
        for idx in range(iter_num):
            if array[idx] < array[idx + 1]:
                temp = array[idx]
                array[idx] = array[idx + 1]
                array[idx + 1] = temp
                temp = index_array[idx]
                index_array[idx] = index_array[idx + 1]
                index_array[idx + 1] = temp
    print(array)
    print(index_array)
    return index_array

if __name__=='__main__':
    # TODO: use toy data and improve logging, see test_backend.py
    working_dir = "/wholebrain/scratch/areaxfs3/"
    location_array = [29753344, 29833344, 30393344, 30493344, 32393344, 34493344, 39553344, 42173344]
    ssv_array = []
    length_array = []
    render_indexview = True
    now = time.time()
    ssc = SuperSegmentationDataset(working_dir)
    ix = 0
    for loc in location_array:
        ssv_array.append(ssc.get_super_segmentation_object(loc))
        exlocs = np.concatenate(ssv_array[ix].sample_locations())
        length_array.append(len(exlocs))
        ix = ix + 1
    #exlocs = exlocs[::30]
    print("Example location array:", exlocs.shape)
    print(working_dir)
    now2 = time.time()
    print("time for reading data")
    print(now2-now)

    array = [19,2,31,45,6,11,121,27]
    index_array = sorting_in_descending_order(array)
    print(array)
    print(index_array)
    index_array = sort_ssv_descending(ssv_array)
    print(index_array)
    print(length_array)
    """
    i = 0
    exlocs = chunkify_successive(exlocs, 10)
    params = []
    for ex in exlocs:
        params.append([exlocs, i])
        i=i+1
        print(i)
        print(params[i-1])
    """
    """
   # view = render_sso_coords_multiprocessing(ssv, working_dir, exlocs, render_indexviews=render_indexview,
                                   n_jobs=20, verbose=True)

    now1 = time.time()

    print(now1)
    print("")
    print(now1-now2)

#global_params.PYOPENGL_PLATFORM = 'osmesa'
    now2 = time.time()
    print(now2)

    if render_indexview:
        views = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views = render_sso_coords(ssv, exlocs, verbose=True)
        
    """
    now3 = time.time()
    print(now3-now2)
    #views = render_sso_coords_generic(ssv, working_dir, exlocs, render_indexviews=render_indexview, verbose=True)

#now2 = time.time()

# print(now2)
#print(now2-now1)

#print(np.sum((views-viewsos)**2))




#imsave('/u/atultm/test_2dproj.png', views[0, 0, 0])           #running on cluster
#imsave('/home/atulm/test_2dproj.png', views[0, 0, 0])        #running on local machine
#imageio.imwrite('/u/atultm/test_2dpro1j.png', views[0, 0, 0])



