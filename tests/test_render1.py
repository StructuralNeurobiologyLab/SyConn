from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views
import time
from multiprocessing import cpu_count


def test_multiprocessed_vs_serial_rendering():
    # TODO: use example data and improve logging, see test_backend.py
    working_dir = "/wholebrain/scratch/areaxfs3/"
    render_indexview = True

    ssc = SuperSegmentationDataset(working_dir)
    ssv = ssc.get_super_segmentation_object(29753344)
    ssv.nb_cpus = cpu_count()
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:1000]
    views = render_sso_coords_multiprocessing(
        ssv, working_dir, rendering_locations=exlocs,
        render_indexviews=render_indexview, n_jobs=10, verbose=True)

    # overwrite any precomputed caches by re-initialization of SSV
    ssv = ssc.get_super_segmentation_object(29753344)
    ssv.nb_cpus = cpu_count()
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:1000]
    if render_indexview:
        views2 = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views2 = render_sso_coords(ssv, exlocs, verbose=True)

    print('Fraction of different index values in index-views: {:.4f}'
          ''.format(np.sum(views != views2) / np.prod(views.shape)))
    assert np.all(views == views2)


if __name__ == '__main__':
    test_multiprocessed_vs_serial_rendering()

