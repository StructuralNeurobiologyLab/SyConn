from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, render_sso_coords_multiprocessing
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


    #def merge(self, param):
        #


if __name__=='__main__':
    # TODO: use toy data and improve logging, see test_backend.py
    now = time.time()
    print(now)
    print(global_params.wd)
    n = 1
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
    """
        ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')  # running on cluster
        # ssc=SuperSegmentationDataset('/home/atulm/mount/wb/wholebrain/scratch/areaxfs3/')    #running on local machine
        ssv = ssc.get_super_segmentation_object(29753344)

        exloc = np.array([5602, 4173, 4474]) * ssv.scaling
        exlocs = np.concatenate(ssv.sample_locations())

        views = render_sso_coords(ssv, exlocs[::10], verbose=True)
    """
    params = []
    ssv1 = []
    real_params = []
    real_params2 = []
    j = 0
    k = 0
    ln = []
    ln.append(0)
    #for i in range(n):
     #   params.append(data())
      #  print(params[i].exlocs)
       # print(params[i].ssc)

    for i in range(n):
        ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')
        ssv = ssc.get_super_segmentation_object(29753344)
        #print(i)
        ssv1.append(ssv)
        exloc = np.array([5602, 4173, 4474]) * ssv.scaling
        exlocs = np.concatenate(ssv.sample_locations())
        exlocs = exlocs[::10]
        l = 0
        l = len(exlocs)
        #for i in range(l):
         #   ssv1.append(ssv)
         #   j = (j+1)
            #print(j)
        params.extend(exlocs)
        k = (k+l)
        ln.append(k)
        print(k)
        print(ln)
        print(len(params))
    for i in range(n):
        real_params.append(0)
    for i in range(n):
        real_params[i] = {'sso' : ssv1[i], 'coords' : params[ln[i]:(ln[i+1]-1)]}
    #for i in range(k):
     #   real_params[i] = {'sso': ssv1[i], 'coords': params[i]}

    #print(len(params))
    #print(len(ssv1))

    #for i in range(j):
    #    real_params[i] = {'sso' : ssv1[i], 'coords' : params[i]}
    """
    for i in range(n):
        ssc = SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')
        ssv = ssc.get_super_segmentation_object(29753344)
        #print(i)
        #ssv1.append(ssv)
        exloc = np.array([5602, 4173, 4474]) * ssv.scaling
        exlocs = np.concatenate(ssv.sample_locations())
        for kk, coords in enumerate(exlocs, k):

            params_thread = [ssv, coords]
            real_params2.append(params_thread)
        l = 0
        l = len(exlocs)
        k = (k+l)
        print(k)
        print("Reading data so far")

    #print(len(params))
    #print(len(ssv1))
    print(len(real_params2))
    print(real_params2[3000])
    
    for i in range(k):
        real_params.append(0)
    for i in range(k):
        real_params[i] = (ssv1[i], params[i])
    """
    working_dir = "/wholebrain/scratch/areaxfs3/"
    print(working_dir)
    now2 = time.time()
    print("time for reading data")
    print(now2-now)
    render_sso_coords_multiprocessing(ssv1, working_dir, params, n_jobs=n, verbose=True)
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



