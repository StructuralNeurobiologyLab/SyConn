from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.handler.prediction import int2str_converter
from syconn.proc.meshes import mesh2obj_file_colors
from morphx.classes.hybridmesh import HybridCloud
from scipy.spatial import cKDTree
from collections import Counter

import os
import multiprocessing as mp
import tqdm as tqdm
import numpy as np

target_dir = os.path.expanduser(f'/wholebrain/scratch/amancu/mergeError/Nodes/AdditionalGT/')

GREY = np.array([180., 180., 180., 255.])
PINK = np.array([10., 255., 10., 255.])

def create_hcs(cell_ids, slice):
    for cell_id in tqdm.tqdm(cell_ids[slice]):
        curr_cell = ssd.get_super_segmentation_object(cell_id)
        curr_cell.load_skeleton()
        curr_cell.load_mesh('sv')
        curr_cell.skeleton['nodes']

        # setup verts and features
        verts = curr_cell.mesh[1].reshape(-1, 3)
        features = np.zeros(shape=(len(verts),), dtype=np.int32)

        # setup nodes and their labels
        nodes = curr_cell.skeleton['nodes'] * curr_cell.scaling
        node_labels = np.zeros(shape=(len(nodes),)) - 2
        node_tree = cKDTree(data=nodes)
        # set first the context extraction area
        # print(node_tree.query_ball_point(verts.mean(axis=0), r=int(40e3)))
        idcs = np.unique(node_tree.query_ball_point(verts.mean(axis=0), r=int(30e3)))
        node_labels[idcs] = float(-1)

        hc = HybridCloud(vertices=verts, features=features,
                             nodes=nodes, node_labels=node_labels,
                             edges=curr_cell.skeleton['edges'])

        # # for distances
        # colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
        # mask = np.where(hc.node_labels == -1)[0]
        # mask = np.array([[x] for x in mask])
        # try:
        #     np.put_along_axis(colors, mask, PINK, axis=0)
        # except:
        #     print("No foreground labels in original context.")
        #     pass
        # mesh2obj_file_colors(os.path.expanduser(
        #     f'/wholebrain/scratch/amancu/mergeError/Nodes/Examples/Additional/{cell_id}_extraction.ply'),
        #     [np.array([]), hc.nodes, np.array([])], colors)
        #
        # # for verts
        # colors = np.full(shape=(hc.vertices.shape[0], 4,), fill_value=GREY)
        # mesh2obj_file_colors(os.path.expanduser(
        #     f'/wholebrain/scratch/amancu/mergeError/Nodes/Examples/Additional/{cell_id}_verts.ply'),
        #     [np.array([]), hc.vertices, np.array([])], colors)

        if hc.save2pkl(os.path.expanduser(target_dir + f'{cell_id}.pkl')):
                    print(f'HybridCloud not written for {cell_id}')





if __name__ == '__main__':
    # get the better datset
    global_params.wd = '/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2'
    ssd = SuperSegmentationDataset(global_params.config.working_dir)

    # define wanted celltypes to insert
    msn_mask = ssd.load_numpy_data('celltype_cnn_e3') == 2
    msns = ssd.ssv_ids[msn_mask]

    lman_mask = ssd.load_numpy_data('celltype_cnn_e3') == 3
    lmans = ssd.ssv_ids[lman_mask]

    hvc_mask = ssd.load_numpy_data('celltype_cnn_e3') == 4
    hvcs = ssd.ssv_ids[hvc_mask]

    # define attributes of cells
    # get each size
    sizes = ssd.load_numpy_data('size')
    msn_sizes = sizes[msn_mask]
    lman_sizes = sizes[lman_mask]
    hvc_sizes = sizes[hvc_mask]

    # get wanted MSN cell ids
    eighty = np.percentile(msn_sizes, 85)
    ninety = np.percentile(msn_sizes, 95)
    msn_ids_to_insert = msns[(msn_sizes > eighty) & (msn_sizes < ninety)]
    msn_to_insert = np.random.choice(msn_ids_to_insert, 500)

    if len(list((Counter(msn_to_insert) - Counter(set(msn_to_insert))).keys())) != 0:
        print("MSN array has duplicate ids. Rectifying... ")
        a = list((Counter(msn_to_insert) - Counter(set(msn_to_insert))).keys())
        mask = np.invert(np.isin(msn_to_insert, a))
        msn_to_insert = msn_to_insert[mask]
        print(f"MSN has now {len(msn_to_insert)} cell ids")

    if len(list((Counter(msn_to_insert) - Counter(set(msn_to_insert))).keys())) != 0:
        raise ValueError("MSN array has duplicate ids.")

    # get wanted LMAN cell ids
    eighty = np.percentile(lman_sizes, 85)
    ninety = np.percentile(lman_sizes, 95)
    lman_ids_to_insert = lmans[(lman_sizes > eighty) & (lman_sizes < ninety)]
    lman_to_insert = np.random.choice(lman_ids_to_insert, 500)

    if len(list((Counter(lman_to_insert) - Counter(set(lman_to_insert))).keys())) != 0:
        print("LMAN array has duplicate ids. Rectifying... ")
        a = list((Counter(lman_to_insert) - Counter(set(lman_to_insert))).keys())
        mask = np.invert(np.isin(lman_to_insert, a))
        lman_to_insert = lman_to_insert[mask]
        print(f"LMAN has now {len(lman_to_insert)} cell ids")

    if len(list((Counter(lman_to_insert) - Counter(set(lman_to_insert))).keys())) != 0:
        raise ValueError("LMAN array has duplicate ids.")
    
    # get wanted HVC cell ids
    eighty = np.percentile(hvc_sizes, 85)
    ninety = np.percentile(hvc_sizes, 95)
    hvc_ids_to_insert = hvcs[(hvc_sizes > eighty) & (hvc_sizes < ninety)]
    hvc_to_insert = np.random.choice(hvc_ids_to_insert, 500)

    if len(list((Counter(hvc_to_insert) - Counter(set(hvc_to_insert))).keys())) != 0:
        print("HVC array has duplicate ids. Rectifying...")
        a = list((Counter(hvc_to_insert) - Counter(set(hvc_to_insert))).keys())
        mask = np.invert(np.isin(hvc_to_insert, a))
        hvc_to_insert = hvc_to_insert[mask]
        print(f"HVC now has {len(hvc_to_insert)} cell ids")
    
    if len(list((Counter(hvc_to_insert) - Counter(set(hvc_to_insert))).keys())) != 0:
        raise ValueError("HVC array has duplicate ids.")

    ids_list = np.concatenate([msn_to_insert, lman_to_insert, hvc_to_insert])

    n_proc = 18
    offset = 0
    chunksize = len(ids_list) // n_proc
    proc_slices = []

    for i_proc in range(n_proc):
        chunkstart = int(offset + (i_proc * chunksize))
        # make sure to include the division remainder for the last process
        chunkend = int(offset + (i_proc + 1) * chunksize) if i_proc < n_proc - 1 else int(offset + len(ids_list))
        proc_slices.append(np.s_[chunkstart:chunkend])

    with mp.Pool(processes=n_proc) as pool:
        multiple_results = [pool.apply_async(create_hcs, (ids_list, slicee)) for slicee in proc_slices]
        print([res.get() for res in multiple_results])
