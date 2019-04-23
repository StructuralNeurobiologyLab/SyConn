from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views
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


if __name__=='__main__':
    # TODO: use toy data and improve logging, see test_backend.py
    working_dir = "/wholebrain/scratch/areaxfs3/"
    render_indexview = False
    now = time.time()
    ssc = SuperSegmentationDataset(working_dir)
    ssv = ssc.get_super_segmentation_object(29753344)
    exlocs = np.concatenate(ssv.sample_locations())
    #exlocs = exlocs[::10]
    example = np.arange(30).reshape((10, 3))
    print(example)
    print(example.shape)
    example = np.array_split(example, 4)
    print(example)
    print("Example location array:", exlocs.shape)
    print(working_dir)
    now2 = time.time()
    print("time for reading data")
    print(now2-now)
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
    view = render_sso_coords_multiprocessing(ssv, working_dir, exlocs, n_jobs=5,
                                verbose=True, render_indexviews=render_indexview)
    view1 = render_sso_coords(ssv, exlocs, verbose=True)
    now1 = time.time()
    print(np.array_equal(view, view1))
    #print(np.sum((view - view1)**2))
    print(now1)
    print("")
    print(now1-now)
    print(view[0])
    print(view1[0])
    print(np.array_equal(view[0], view1[0]))
    print(len(view[0]))
    print('hello_atul')
    print(len(view1[0]))
    print('real trial')
    print(np.array_equiv(view, view1))
    print(np.array_equiv(view[1001], view1[1001]))
#global_params.PYOPENGL_PLATFORM = 'osmesa'
    now2 = time.time()
    print(now2)
    """
    if render_indexview:
        views = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views = render_sso_coords(ssv, exlocs, verbose=True)
    now3 = time.time()
    print(now3-now2)
    """

#now2 = time.time()

# print(now2)
#print(now2-now1)

#print(np.sum((views-viewsos)**2))




#imsave('/u/atultm/test_2dproj.png', views[0, 0, 0])           #running on cluster
#imsave('/home/atulm/test_2dproj.png', views[0, 0, 0])        #running on local machine
#imageio.imwrite('/u/atultm/test_2dpro1j.png', views[0, 0, 0])



