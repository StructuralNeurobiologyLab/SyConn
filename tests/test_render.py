from syconn.reps.super_segmentation import  SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, render_sso_coords_multiprocessing
import time

class data:
    ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')
    ssv = ssc.get_super_segmentation_object(29753344)
    def __init__(self):
        ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')
        ssv = ssc.get_super_segmentation_object(29753344)
        #self.ssc = ssc.get_super_segmentation_object(29753344)
        exloc = np.array([5602, 4173, 4474]) * ssv.scaling
        self.exlocs = np.concatenate(ssv.sample_locations())


    def merge(self, param):




if __name__=='__main__':
    # TODO: use toy data and improve logging, see test_backend.py
    now = time.time()
    print(now)
    n=1
    ssv = []
    exloc = []
    exlocs = []
    #render_sso_coords_multiprocessing()
    """"
    for i in range(5):
        ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')  # running on cluster
        # ssc=SuperSegmentationDataset('/home/atulm/mount/wb/wholebrain/scratch/areaxfs3/')    #running on local machine
        ssv.append(ssc.get_super_segmentation_object(29753344))

        exloc.append(np.array([5602, 4173, 4474]) * ssv.scaling)
        exlocs.append(np.concatenate(ssv.sample_locations())
        entry = [ssv, exlocs]
        params.append(entry)
        
    """
    params = []
    for i in range(n):
        params.append(data())
        print(params[i].exlocs)
        print(params[i].ssc)

    render_sso_coords_multiprocessing(params, n_job=n, verbose=True)
    now1 = time.time()
    print(now1)
    print(now1-now)

    #global_params.PYOPENGL_PLATFORM = 'osmesa'


    #viewsos = render_sso_coords(ssv, exlocs[::10], verbose=True )

    #now2 = time.time()

   # print(now2)
    #print(now2-now1)

    #print(np.sum((views-viewsos)**2))




    #imsave('/u/atultm/test_2dproj.png', views[0, 0, 0])           #running on cluster
    #imsave('/home/atulm/test_2dproj.png', views[0, 0, 0])        #running on local machine
    #imageio.imwrite('/u/atultm/test_2dpro1j.png', views[0, 0, 0])



