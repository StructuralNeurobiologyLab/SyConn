from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views, \
    render_sso_coords_label_views
import time


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

    if render_indexview:
        views2 = render_sso_coords_label_views(ssv, exlocs, verbose=True)
    else:
        views2 = render_sso_coords(ssv, exlocs, verbose=True)

    assert np.all(views == views2)
    raise()

if __name__ == '__main__':
    test_multiprocessed_vs_serial_rendering()

