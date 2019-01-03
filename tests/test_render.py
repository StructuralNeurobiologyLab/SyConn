from syconn.reps.super_segmentation import  SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords
import matplotlib.pyplot as plt
from scipy.misc import imsave

if __name__=='__main__':
   ssc=SuperSegmentationDataset('/wholebrain/scratch/areaxfs3/')   #running on cluster
    #ssc=SuperSegmentationDataset('/home/atulm/mount/wb/wholebrain/scratch/areaxfs3/')    #running on local machine
    ssv = ssc.get_super_segmentation_object(29753344)

    exloc=np.array([5602, 4173, 4474])*ssv.scaling
    more_exlocs = np.concate(ssv.sample_locations())

    views = render_sso_coords(ssv, [exloc], )
    imsave('/u/atultm/test_2dproj.png', views[0, 0, 0])           #running on cluster
    #imsave('/home/atultm/test_2dproj.png', views[0, 0, 0])        #running on local machine
    # imageio.imwrite('/u/atultm/test_2dpro1j.png', views[0, 0, 0])



