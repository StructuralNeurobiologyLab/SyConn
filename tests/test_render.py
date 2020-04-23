import pytest
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = f'{os.path.dirname(__file__)}/../data/renderexample.k.zip'


def test_multiprocessed_vs_serial_rendering():
    from syconn.proc.ssd_assembly import init_sso_from_kzip
    import numpy as np
    from syconn.proc.rendering import render_sso_coords, \
        render_sso_coords_multiprocessing, render_sso_coords_index_views
    from multiprocessing import cpu_count
    render_indexview = True

    ssv = init_sso_from_kzip(fname, sso_id=1)
    ssv.nb_cpus = cpu_count()
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:10]
    views = render_sso_coords_multiprocessing(
        ssv, rendering_locations=exlocs,
        render_indexviews=render_indexview, n_jobs=10, verbose=True)

    # overwrite any precomputed caches by re-initialization of SSV
    del ssv
    ssv = init_sso_from_kzip(fname, sso_id=1)
    ssv.nb_cpus = cpu_count()
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:10]
    if render_indexview:
        views2 = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views2 = render_sso_coords(ssv, exlocs, verbose=True)

    print('Fraction of different index values in index-views: {:.4f}'
          ''.format(np.sum(views != views2) / np.prod(views.shape)))
    assert np.all(views == views2)


@pytest.mark.filterwarnings("ignore:Modifying DynConfig items via")
def test_raw_and_index_rendering_osmesa():
    from syconn import global_params
    global_params.config['pyopengl_platform'] = 'osmesa'
    from syconn.proc.ssd_assembly import init_sso_from_kzip
    from syconn.proc.rendering import render_sso_coords, \
        render_sso_coords_index_views
    assert os.path.isfile(fname)
    ssv = init_sso_from_kzip(fname, sso_id=1)
    rendering_locations = np.concatenate(ssv.sample_locations())[:10]
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
    assert os.path.isfile(fname)
    ssv = init_sso_from_kzip(fname, sso_id=1)
    rendering_locations = np.concatenate(ssv.sample_locations())[:10]
    index_views = render_sso_coords_index_views(ssv, rendering_locations,
                                                verbose=True)
    raw_views = render_sso_coords(ssv, rendering_locations, verbose=True)
    assert len(index_views) == len(raw_views)
    # TODO: add further property tests, e.g. dtype, value range ...


@pytest.mark.filterwarnings("ignore:Modifying DynConfig items via")
def test_egl_and_osmesa_swap_and_equivalence():
    from syconn import global_params
    global_params.config['pyopengl_platform'] = 'egl'
    from syconn.proc.ssd_assembly import init_sso_from_kzip
    from syconn.proc.rendering import render_sso_coords, \
        render_sso_coords_index_views
    assert os.path.isfile(fname)
    ssv = init_sso_from_kzip(fname, sso_id=1)
    rendering_locations = np.concatenate(ssv.sample_locations())
    index_views = render_sso_coords_index_views(ssv, rendering_locations,
                                                verbose=True)
    raw_views = render_sso_coords(ssv, rendering_locations, verbose=True)

    global_params.config['pyopengl_platform'] = 'osmesa'
    index_views_osmesa = render_sso_coords_index_views(ssv, rendering_locations,
                                                verbose=True)
    raw_views_osmesa = render_sso_coords(ssv, rendering_locations, verbose=True)
    nb_of_pixels = np.prod(raw_views.shape)
    # fraction of different vertex indices must be below 1 out of 100k
    assert np.sum(index_views != index_views_osmesa) / nb_of_pixels < 1e-5
    # maximum deviation of depth value must be small
    # used to be 1 instead of 20, changed with commit 53ff49e0 (only 9 pixel in total with high deviation)
    # manual inspection of the one image causing this deviation yielded no qualitative difference
    assert np.max(np.abs(raw_views-raw_views_osmesa)) < 20
    assert np.sum(np.abs(raw_views - raw_views_osmesa) > 1) / nb_of_pixels < 1e-5
    # fraction of affected pixels must be below 0.05
    assert np.sum(raw_views != raw_views_osmesa) / nb_of_pixels < 0.05


if __name__ == '__main__':
    test_raw_and_index_rendering_osmesa()
    test_raw_and_index_rendering_egl()
    test_egl_and_osmesa_swap_and_equivalence()


