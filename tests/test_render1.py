from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views
import time
from matplotlib.pyplot import imsave
import imageio
import pathlib


def test_multiprocessed_vs_serial_rendering():
    # TODO: use example data and improve logging, see test_backend.py
    working_dir = "/wholebrain/scratch/areaxfs3/"
    render_indexview = False

    ssc = SuperSegmentationDataset(working_dir)
    ssv = ssc.get_super_segmentation_object(29753344)
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:100]
    views = render_sso_coords_multiprocessing(
        ssv, working_dir, rendering_locations=exlocs,
        render_indexviews=render_indexview, n_jobs=5, verbose=True)
    #raise()
    if render_indexview:
        views2 = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views2 = render_sso_coords(ssv, exlocs, verbose=True)
    # imsave('/u/atultm/test_2dproj_multi.png', views[0, 0, 0])
    # imsave('/u/atultm/test_2dproj.png', views2[0, 0, 0])
    for i in range(len(views)):
        # if not os.path.exists(i):
        #     os.makedirs(i)
        for j in range(2):
            # if not os.path.exists(j):
            #     os.makedirs(j)
    # print(shape.views[0, 0, 0])
    # print(shape.views2[0, 0, 0])
            file_name = '/u/atultm/image_data/' + str(j) +'_'+ str(i)+'test_2dpro1j_multi.png'
            file_name2 = '/u/atultm/image_data/' + str(j) +'_'+ str(i) + 'test_2dpro1j.png'
            imageio.imwrite(file_name, views[i, 0, j])
            imageio.imwrite(file_name2, views2[i, 0, j])

    assert np.all(views == views2)

def render_index_comparison():
    working_dir = "/wholebrain/scratch/areaxfs3/"
    ssc = SuperSegmentationDataset(working_dir)
    ssv = ssc.get_super_segmentation_object(29753344)
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:100]
    views1 = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    #views2 = render_sso_coords_index_views_new(ssv, exlocs, verbose=True)
    return views1
if __name__ == '__main__':
    test_multiprocessed_vs_serial_rendering()

