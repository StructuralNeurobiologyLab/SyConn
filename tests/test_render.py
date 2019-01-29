from syconn.reps.super_segmentation import  SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords
import time

if __name__=='__main__':
    # TODO: use toy data and improve logging, see test_backend.py
    now = time.time()
    print(now)
    ssc=SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')   #running on cluster
    #ssc=SuperSegmentationDataset('/home/atulm/mount/wb/wholebrain/scratch/areaxfs3/')    #running on local machine
    ssv = ssc.get_super_segmentation_object(29753344)

    exloc = np.array([5602, 4173, 4474])*ssv.scaling
    exlocs = np.concatenate(ssv.sample_locations())

    views = render_sso_coords(ssv, exlocs[::10], verbose=True )
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



