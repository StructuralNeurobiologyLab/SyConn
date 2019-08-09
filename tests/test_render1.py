from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
from syconn.proc.rendering import render_sso_coords, \
    render_sso_coords_multiprocessing, render_sso_coords_index_views, \
    render_sso_coords_label_views
import time


def test_multiprocessed_vs_serial_rendering():
    # TODO: use example data and improve logging, see test_backend.py
    working_dir = "/wholebrain/scratch/areaxfs3/"
    render_indexview = True

    ssc = SuperSegmentationDataset(working_dir)
    ssv = ssc.get_super_segmentation_object(29753344)
    exlocs = np.concatenate(ssv.sample_locations())
    exlocs = exlocs[:100]
    views = render_sso_coords_multiprocessing(
        ssv, working_dir, rendering_locations=exlocs,
        render_indexviews=render_indexview, n_jobs=5, verbose=True)

    if render_indexview:
        views2 = render_sso_coords_index_views(ssv, exlocs, verbose=True)
    else:
        views2 = render_sso_coords(ssv, exlocs, verbose=True)

    assert np.all(views == views2)
    raise()

def semseg2mesh_sub2(index_views, semseg_view, index_background, length, classes):
    arr = np.zeros((length+1, classes+1), dtype=int)
    for ii in range(len(index_views)):
        vertex_ix = index_views[ii]
        if vertex_ix == index_background:
            continue
        l = semseg_view[ii]  # vertex label
        arr[vertex_ix][l] += 1
    return arr

if __name__ == '__main__':
    t = time.time()
    #a2 = rat()
    t2 = time.time()
    # arr = np.ones((8, 3))
    # print(t2 - t)
    # print(arr)
    # print(arr[2])
    inde = [1225, 871352, 3127, 8132, 17633, 3127, 1873163, 873163, 871352,
            187279, 3127]
    sems = [256, 112, 126, 234, 12, 42, 71, 112, 112, 89, 55]
    arr = semseg2mesh_sub2(inde, sems, 40000000, 1873163, 256)
    print(arr[873163][112])
    print(arr[871352][112])
    mask = np.sum(arr, axis=1) == 0
    print(mask)
    pred = np.argmax(arr, axis=1)
    pred[mask] = 300
    print(pred)
    print(pred[100])
    # print(dc)
    # print(arr)
    # for ix in arr:
    #     print(dc[ix][0: pp[ix]])
    # test_multiprocessed_vs_serial_rendering()
    # numbers = [[1,3,2], [2,4,6], [3,9,0]]
    # letters = ["A", "B", "C"]
    # params = []
    # for numbers_value, letters_value in zip(numbers, letters):
    #     params.append([numbers_value, letters_value])
    #     #print(numbers_value, letters_value)
    # print(params)
    # # params = [[ix, par] for ix, par in
    # #           enumerate(numbers)]
    # #print(params)
    # numbers = [1, 3, 2, 2, 4, 6, 3, 9]
    # numbers = chunkify_successive_split(numbers,3)
    # params1 = []
    # for numbers_value, letters_value in zip(numbers[0], letters):
    #     params1.append([numbers_value, letters_value])
    # print(params1)
    # d = [[1, 2, 3],[0, 9, 5, 8],[],[],[0, 82, 52],[],[]]
    # c = [[0, 6],[1, 2],[5, 3, 5],[2, 5, 6],[],[],[]]
    # e = [[],[7, 8],[],[92, 21, 6],[],[0, 9, 100],[0, 6]]
    # #p = merge_multi_d_array(d, c)
    # #p1 = merge_multi_d_array(c, d)
    # q = [d, c, e]
    # #print(p)
    # #print(p1)
    # print(q)
    # print(q[0])
    # z = merge_array_of_multi_d_array(q)
    # print(z)