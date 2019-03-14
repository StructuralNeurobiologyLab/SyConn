from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views
import time
import itertools
from syconn import global_params

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
    render_indexview = True
    now = time.time()
    ssc = SuperSegmentationDataset(working_dir)
    ssv = ssc.get_super_segmentation_object(29753344)
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[::10]
    print("Example location array:", exlocs.shape)
    print(working_dir)
    now2 = time.time()
    print("time for reading data")
    print(now2-now)
    render_sso_coords_multiprocessing(ssv, working_dir, exlocs, render_indexviews=render_indexview,
                                      n_jobs=2, verbose=True)
    now1 = time.time()

    print(now1)
    print("")
    print(now1-now)

#global_params.PYOPENGL_PLATFORM = 'osmesa'
    now2 = time.time()
    print(now2)

    if render_indexview:
        views = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views = render_sso_coords(ssv, exlocs, verbose=True)
    now3 = time.time()
    print(now3-now2)


#now2 = time.time()

# print(now2)
#print(now2-now1)

#print(np.sum((views-viewsos)**2))




#imsave('/u/atultm/test_2dproj.png', views[0, 0, 0])           #running on cluster
#imsave('/home/atulm/test_2dproj.png', views[0, 0, 0])        #running on local machine
#imageio.imwrite('/u/atultm/test_2dpro1j.png', views[0, 0, 0])



