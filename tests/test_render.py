import pytest
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


# def test_multiprocessed_vs_serial_rendering():
# from syconn.proc.ssd_assembly import init_sso_from_kzip
# from syconn.reps.super_segmentation import SuperSegmentationDataset
# import numpy as np
# from syconn.proc.rendering import render_sso_coords, \
#     render_sso_coords_multiprocessing, render_sso_coords_index_views
# import time
# import os
# from multiprocessing import cpu_count
#     # TODO: use example data cube and improve logging
#     working_dir = "/wholebrain/scratch/areaxfs3/"
#     render_indexview = True
#     if not os.path.isdir(working_dir):
#         return
#     ssc = SuperSegmentationDataset(working_dir)
#     ssv = ssc.get_super_segmentation_object(29753344)
#     ssv.nb_cpus = cpu_count()
#     exlocs = np.concatenate(ssv.sample_locations())
#     exlocs = exlocs[:1000]
#     views = render_sso_coords_multiprocessing(
#         ssv, working_dir, rendering_locations=exlocs,
#         render_indexviews=render_indexview, n_jobs=10, verbose=True)
#
#     # overwrite any precomputed caches by re-initialization of SSV
#     ssv = ssc.get_super_segmentation_object(29753344)
#     ssv.nb_cpus = cpu_count()
#     exlocs = np.concatenate(ssv.sample_locations())
#     exlocs = exlocs[:1000]
#     if render_indexview:
#         views2 = render_sso_coords_index_views(ssv, exlocs, verbose=True)
#     else:
#         views2 = render_sso_coords(ssv, exlocs, verbose=True)
#
#     print('Fraction of different index values in index-views: {:.4f}'
#           ''.format(np.sum(views != views2) / np.prod(views.shape)))
#     assert np.all(views == views2)


@pytest.mark.filterwarnings("ignore:Modifying DynConfig items via")
def test_raw_and_index_rendering_osmesa():
    from syconn import global_params
    global_params.config['pyopengl_platform'] = 'osmesa'
    from syconn.proc.ssd_assembly import init_sso_from_kzip
    from syconn.proc.rendering import render_sso_coords, \
        render_sso_coords_index_views
    import os
    import numpy as np
    fname = os.path.dirname(__file__) + '/renderexample.k.zip'
    assert os.path.isfile(fname)
    ssv = init_sso_from_kzip(fname, sso_id=1)
    rendering_locations = np.concatenate(ssv.sample_locations())
    index_views = render_sso_coords_index_views(ssv, rendering_locations,
                                                verbose=True)
    raw_views = render_sso_coords(ssv, rendering_locations, verbose=True)
    assert len(index_views) == len(raw_views)
    # TODO: add further property tests, e.g. dtype, value range ...


@pytest.mark.filterwarnings("ignore:Modifying DynConfig items via")
def test_raw_and_index_rendering_egl():
    from syconn import global_params
    global_params.config['pyopengl_platform'] = 'egl'
    from syconn.proc.ssd_assembly import init_sso_from_kzip
    from syconn.proc.rendering import render_sso_coords, \
        render_sso_coords_index_views
    import os
    import numpy as np
    fname = os.path.dirname(__file__) + '/renderexample.k.zip'
    assert os.path.isfile(fname)
    ssv = init_sso_from_kzip(fname, sso_id=1)
    rendering_locations = np.concatenate(ssv.sample_locations())
    index_views = render_sso_coords_index_views(ssv, rendering_locations,
                                                verbose=True)
    raw_views = render_sso_coords(ssv, rendering_locations, verbose=True)
    assert len(index_views) == len(raw_views)
    # TODO: add further property tests, e.g. dtype, value range ...


if __name__ == '__main__':
    test_raw_and_index_rendering_osmesa()

