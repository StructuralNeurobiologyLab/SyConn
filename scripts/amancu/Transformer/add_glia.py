# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Andrei Mancu, Philipp Schubert, Joergen Kornfeld

import numpy as np
import os
import argparse
import networkx as nx
import csv
import gc
import timeit
import multiprocessing as mp

try:
    import open3d as o3d
except ImportError:
    pass

from tqdm import tqdm
from scipy.spatial import cKDTree
from syconn import global_params
from syconn.handler.config import initialize_logging
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from syconn.proc.meshes import calc_contact_syn_mesh, mesh2obj_file_colors, merge_meshes, gen_mesh_voxelmask
from syconn.handler.basics import write_obj2pkl, load_pkl2obj
from syconn.proc.ssd_proc import merge_ssv
from syconn.backend.storage import AttributeDict, MeshStorage, VoxelStorage, VoxelStorageDyn, VoxelStorageLazyLoading
from morphx.classes.hybridmesh import HybridCloud

# paths
filtered_cs_ids_path = os.path.expanduser(f'/cajal/scratch/users/amancu/merge_error/transformer/GT/filtered_cs_ids[10k-100k]_glia.npy')
lookup_cellpair2cs_path = os.path.expanduser(
    f'/cajal/scratch/users/amancu/merge_error/transformer/GT/lookup_cellpair2cs[10k-100k]_glia.pkl')
ply_example_path = os.path.expanduser(f'/cajal/scratch/users/amancu/merge_error/transformer/GT/Examples/')
glia_path = '/wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v5_j0251_72_seg_20210127_agglo2_IDs.csv'

no_Skelmerger_radius = 20e3
skelmerger_radius = 2e3

# colors for labels
PINK = np.array([10., 255., 10., 255.])
BLUE = np.array([255., 125., 125., 255.])
GREY = np.array([180., 180., 180., 255.])

# for the classification task
# "-2" no interest zones, "-1" context zones, "0" no merger, "1" merger zone
# for the regression task task
# "-2" no interest zones, "-1" context zones -> 0
# interest zone [0,1] range, where 1 is a source node, and the distance from it decreases until 0

def get_labels_and_distances(merged_cell, cs_coord_list, source_node_idcs, threshold):
    ##############################
    # First nodes
    merged_cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling  # coordinates of all nodes
    merged_cell_edges = merged_cell.skeleton['edges']

    G = nx.from_edgelist(merged_cell_edges)

    # calculate weights of the graph as euclidean space distance between nodes
    for u, v in G.edges:
        arr = np.array([merged_cell_nodes[u], merged_cell_nodes[v]])
        G.edges[u, v]['weight'] = np.linalg.norm(np.diff(arr, axis=0))

    dist, path = nx.multi_source_dijkstra(G, sources=source_node_idcs, cutoff=threshold)

    # set features to be the distances
    node_features = np.zeros(shape=(len(merged_cell_nodes),)) - 2

    node_tree = cKDTree(data=merged_cell_nodes)
    # set first the context extraction area
    idcs = np.unique(np.concatenate(node_tree.query_ball_point(cs_coord_list, r=int(30e3))))
    node_features[idcs] = float(-1)

    for i in range(len(dist.keys())):
        node_features[list(dist.keys())[i]] = list(dist.values())[i]

    ###########################
    # Second vertices

    # combine cell vertices and build mesh
    cell_mesh = merged_cell.mesh
    cell_vertices = cell_mesh[1].reshape(-1, 3)

    # initialize cKTDTree with the combined cell vertices
    vert_NN = cKDTree(data=cell_vertices, )
    # find nearest neighbor of cell vertices to cs vertices
    vert_neighbors = vert_NN.query_ball_point(x=cs_coord_list, r=threshold, workers=2)
    # create single set of point neighbors
    vert_neighbors = np.unique(np.concatenate(vert_neighbors)).astype(int)

    # create labels and the corresponding colors
    # for type compatibility
    one = np.uint(1)
    vertex_labels = np.zeros(shape=(len(cell_vertices),), dtype=np.uint)
    np.put(vertex_labels, vert_neighbors, one)

    colors = np.full(shape=(len(cell_vertices), 4,), fill_value=GREY)
    vert_neighbors = np.array([[x] for x in vert_neighbors])
    np.put_along_axis(colors, vert_neighbors, PINK, axis=0)

    return merged_cell_nodes, node_features, cell_vertices, vertex_labels

def merge_ssv_and_get_source_node_idcs(cell_obj1, cell_obj2, cs_coord_list):
    """
        Merge two cell objects into one"

        Notes:
            Skeleton is in voxel coordinates

        Parameters
        ----------
        cell_obj1, cell_obj2 : SuperSegmentationObject
            Two cells to be merged.
        cs_coord_list : List of locations representing each area of contact site
    """

    merged_cell = SuperSegmentationObject(ssv_id=-1, working_dir=None, version='tmp')
    for mesh_type in ['sv']:                                  # 'sj' fails for current v3 dataset (Not Found)
        # if failure occurs, leave no foreground sample in set, so that true negatives are still inside
        try:
            mesh1 = cell_obj1.load_mesh(mesh_type)
        except:
            print(f'not able to load mesh for :{cell_obj1}')
            return None, None

        try:
            mesh2 = cell_obj2.load_mesh(mesh_type)
        except:
            print(f'not able to find mesh for: {cell_obj2}')
            return None, None
        ind_lst = [mesh1[0], mesh2[0]]
        vert_lst = [mesh1[1], mesh2[1]]

        merged_cell._meshes[mesh_type] = merge_meshes(ind_lst, vert_lst)
        merged_cell._meshes[mesh_type] += ([None, None],)  # add normals

    # merge skeletons
    merged_cell.skeleton = {}
    try:
        cell_obj1.load_skeleton()
    except:
        return None, None

    try:
        cell_obj2.load_skeleton()
    except:
        return None, None
    edge_idc_offset = len(cell_obj1.skeleton['nodes'])
    merged_cell.skeleton['edges'] = np.concatenate([cell_obj1.skeleton['edges'],
                                                    cell_obj2.skeleton['edges'] +
                                                    edge_idc_offset])  # additional offset

    # merge UNSCALED skeleton nodes
    merged_cell.skeleton['nodes'] = np.concatenate([cell_obj1.skeleton['nodes'],
                                                    cell_obj2.skeleton['nodes']])

    # Find the all 2 nodes that are the nearest to the contact site areas
    scaled_skeleton1 = cell_obj1.skeleton['nodes'] * merged_cell.scaling
    scaled_skeleton2 = cell_obj2.skeleton['nodes'] * merged_cell.scaling
    node_pairs = []

    node_tree1 = cKDTree(data=scaled_skeleton1)
    node_tree2 = cKDTree(data=scaled_skeleton2)
    # find first neighboring node from each skeleton
    for cs_coord in cs_coord_list:
        _, idcs1 = node_tree1.query(cs_coord, k=1, workers=2)
        _, idcs2 = node_tree2.query(cs_coord, k=1, workers=2)
        try:
            node_pairs.append([idcs1, idcs2 + edge_idc_offset])
        except Exception as e:
            log.error(f'Exception: {e}')
            continue  # if no neighbor was found in either nn searches

    # add remaining edges of the merge spot
    merged_cell.skeleton['edges'] = np.concatenate([merged_cell.skeleton['edges'], np.unique(node_pairs,
                                                                                             axis=0)])  # node_pairs should be unique, as some node connecttions repeat themselves
    merged_cell.skeleton['diameters'] = np.concatenate([cell_obj1.skeleton['diameters'],
                                                        cell_obj2.skeleton['diameters']])

    return merged_cell, set(np.unique(node_pairs))

def create_labeled_points(cell_pair2cs_ids, cell_pairs, slice, cs_dataset, ssv_set, radii):
    # process every cell pair
    for cellpair in tqdm(cell_pairs[slice], desc='Sample gen'):
        cell1 = cellpair[0]
        cell2 = cellpair[1]
        # if cells are same, skip
        if cell1 == cell2:
            log.warn(f'Dict cells are the same for: {cell1}')
            continue

        if dataset == 'test':
            truth_array = [os.path.exists(os.path.expanduser(
                f'/cajal/scratch/users/amancu/merge_error/transformer/GT/test/R{int(radius)}_downsample300/sso_{cell1}_{cell2}.pkl'))
                for radius in radii]
        else:
            truth_array = [os.path.exists(os.path.expanduser(
                f'/cajal/scratch/users/amancu/merge_error/transformer/GT/training/R{int(radius)}_downsample300/sso_{cell1}_{cell2}.pkl'))
                for radius in radii]
        if np.all(truth_array):
            continue

        # get partner cells, merge them, and get contact site mesh
        fstObj = ssd.get_super_segmentation_object(cell1)
        sndObj = ssd.get_super_segmentation_object(cell2)

        # cs coords for skeleton nodes
        cs_coord_list = []
        for cs_id in cell_pair2cs_ids[(cell1, cell2)]:
            cs = cs_dataset.get_segmentation_object(cs_id)
            # list of cs meshes [(indices, vertices, normals)]
            voxel_dc = VoxelStorage(cs.voxel_path, read_only=True, disable_locking=True)
            voxel_iter = voxel_dc.iter_voxelmask_offset(cs.id, overlap=1)
            cs_mesh = gen_mesh_voxelmask(voxel_iter, cs.scaling, overlap=1, compute_connected_components=True)
            for mesh in cs_mesh:
                area_mesh = mesh[1].reshape(-1, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(area_mesh)
                voxel_size = 300
                _, idcs, _ = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound())
                rep_coords = area_mesh[np.max(idcs, axis=1)]
                # choose the mean coords on each axis as the representative coordinate for the contact site
                # rep_coord = np.mean(area_mesh, axis=0)
                cs_coord_list.extend(rep_coords)

        if len(cs_coord_list) == 0:
            log.info(f'No cs found for given cell pair {cell1} and {cell2}')
            continue

        # pass one vertex from each contact site mesh too, so that the skeleton concatenation is done correctly
        merged_cell, source_node_idcs = merge_ssv_and_get_source_node_idcs(fstObj, sndObj, cs_coord_list)

        if merged_cell is None:
            print(f'Nonetype for cells {cell1} and {cell2}')
            continue

        merged_cell_verts = merged_cell.mesh[1].reshape(-1, 3)
        features = np.zeros(shape=(len(merged_cell_verts),), dtype=np.int32)

        for radius in radii:
            # look for nearest neighbors, merge cell meshes and label
            # cell_vertices, vertex_labels, colors = find_vertNearestNeighbor(merged_cell, cs_verts, radius)

            # look for nearby skeleton nodes
            merged_cell_nodes, node_labels, merged_cell_verts, vert_labels = get_labels_and_distances(merged_cell, cs_coord_list,
                                                                                        source_node_idcs, radius)

            # save mesh to .ply and mesh+skeleton with labels as HybridCloud .pkl
            #TODO put also labels
            hc = HybridCloud(vertices=merged_cell_verts, features=features, labels=vert_labels,
                             nodes=merged_cell_nodes, node_labels=node_labels,
                             edges=merged_cell.skeleton['edges'])

            if dataset == 'test':
                if hc.save2pkl(os.path.expanduser(
                        f'/cajal/scratch/users/amancu/merge_error/transformer/GT/test/R{int(radius)}_downsample300/sso_{cell1}_{cell2}.pkl')):
                    log.info('HybridCloud not written')
            else:
                if hc.save2pkl(os.path.expanduser(
                        f'/cajal/scratch/users/amancu/merge_error/transformer/GT/training/R{int(radius)}_downsample300/sso_{cell1}_{cell2}.pkl')):
                    log.info('HybridCloud not written')

            # # for distances
            # colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            # mask = np.where(hc.node_labels >= 0)[0]
            # mask = np.array([[x] for x in mask])
            # try:
            #     np.put_along_axis(colors, mask, PINK, axis=0)
            # except:
            #     print("No foreground labels in original context.")
            #     pass
            # mesh2obj_file_colors(os.path.expanduser(
            #     f'/wholebrain/scratch/amancu/mergeError/Nodes/Examples/Glia/sso_{cell1}_{cell2}_original_distance_nodes.ply'),
            #     [np.array([]), hc.nodes, np.array([])], colors)

            # # for source nodes
            # colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            # mask = np.where(hc.node_labels == 1)[0]
            # mask = np.array([[x] for x in mask])
            # try:
            #     np.put_along_axis(colors, mask, BLUE, axis=0)
            # except:
            #     print("No foreground labels in original context.")
            #     pass
            # mesh2obj_file_colors(os.path.expanduser(
            #     f'/wholebrain/scratch/amancu/mergeError/Nodes/Examples/Glia/sso_{cell1}_{cell2}_original_source_nodes.ply'),
            #     [np.array([]), hc.nodes, np.array([])], colors)

        del hc
        gc.collect()


def create_lookup_table_glia(filtered_contact_sites_ids, cs_dataset, glia_ids):
    """Loop through all contact_sites ids and if two corresponding cells are found,
    store the cell_id and corresponding cs_id into dictionary

    Returns
    -------
    cell_pair2cs_ids : dict
    cell_pairs : list
    """
    cell_pair2cs_ids = {}
    cell_pairs = []
    ssd = SuperSegmentationDataset()
    # only add 2000 samples for astrocyte mergers
    limit = 0

    for cs_id in tqdm(filtered_contact_sites_ids):
        sv_partner = cs_dataset.get_segmentation_object(cs_id).cs_partner
        if sv_partner[0] in glia_ids or sv_partner[1] in glia_ids:
            limit += 1
            print(f'FOUND PARTNER WITH GLIA: {sv_partner[0] if sv_partner[0] in glia_ids else sv_partner[1]}')
            # dict = ssd.sv2ssv_ids(sv_partner)
            c1 = sv_partner[0]
            c2 = sv_partner[1]

            if c1 == c2:
                continue
            if c1 < c2:
                if (c1, c2) in cell_pair2cs_ids:
                    cell_pair2cs_ids[(c1, c2)].append(cs_id)
                else:
                    cell_pair2cs_ids[(c1, c2)] = [cs_id]
                    cell_pairs.append((c1, c2))
            else:
                if (c2, c1) in cell_pair2cs_ids:
                    cell_pair2cs_ids[c2, c1].append(cs_id)
                else:
                    cell_pair2cs_ids[(c2, c1)] = [cs_id]
                    cell_pairs.append((c2, c1))
        if limit == 3000:
            break

    log.info("len cell_pairs2cs_ids: {}".format(len(cell_pair2cs_ids)))
    return cell_pair2cs_ids, cell_pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate samples for merger error')
    parser.add_argument('--r', nargs='+', help='Radius of merger',
                        default=[3000])
    parser.add_argument('--nproc', type=int, help='Number of processors to use',
                        default=20)
    parser.add_argument('--set', type=str, help='Training or test set generation.', default='training')
    args = parser.parse_args()
    # global cs_ptMerger_radius
    cs_merge_radii = [int(x) for x in args.r]

    n_proc = args.nproc
    dataset = args.set

    experiment_name = f'generate_mergeError_GT_glia'
    global log
    log = initialize_logging(experiment_name,
                             log_dir=os.path.expanduser('/wholebrain/scratch/amancu/mergeError/Nodes/logs/'))

    # setup datasets
    global_params.wd = '/ssdscratch/pschuber/songbird/j0251/j0251_72_seg_20210127_agglo2/'
    ssd = SuperSegmentationDataset()
    cs_dataset = SegmentationDataset(obj_type='cs')

    glia_ids = []

    glia_ids = [342564861, 236420399, 2211357026, 2110074037, 1185827504, 52230331, 2088339237, 2184701599, 471383931,\
                468147607, 616904585, 446586134, 1284891076, 1663451484, 1761534407, 1444345326, 1572357060, \
                1976729924, 1716896659, 2006421567, 1718483951, 644198848, 746433201]

    print(f'glia count: {len(glia_ids)}')

    log.info(f'Executing GT generation for: {dataset}')
    log.info(f'Datasets loaded')
    log.info(f'Cs merge radii: {cs_merge_radii}')

    # skip small and very large CS. Keep 10.000 < size < 1.000.000 -> ?
    if not os.path.exists(filtered_cs_ids_path):
        # mask to get the filtered contact sites by size
        mask = (cs_dataset.sizes > int(1e4)) & (cs_dataset.sizes < int(1e6))
        filtered_cs_ids = cs_dataset.ids[mask]
        np.save(filtered_cs_ids_path, filtered_cs_ids)
        log.info(f'Done writing filtered cs_ids')
    else:
        filtered_cs_ids = np.load(filtered_cs_ids_path)
        log.info('Filtered ids gotten from file')

    # create lookup table for cell ids to cs ids
    if not os.path.exists(lookup_cellpair2cs_path):
        cell_pair2cs_ids, cell_pairs = create_lookup_table_glia(filtered_cs_ids, cs_dataset, glia_ids)
        write_obj2pkl(lookup_cellpair2cs_path, cell_pair2cs_ids)
        log.info(f'Cell pairs written to pickle')
    else:
        try:
            cell_pair2cs_ids = load_pkl2obj(lookup_cellpair2cs_path)
            cell_pairs = []
            for key in cell_pair2cs_ids.keys():
                cell_pairs.append(key)
            log.info(f'Cell pairs and pair2cs dict loaded')
        except:
            log.info(f'Pickle file not found')
            cell_pair2cs_ids, cell_pairs = create_lookup_table_glia(filtered_cs_ids, cs_dataset, glia_ids)
            write_obj2pkl(lookup_cellpair2cs_path, cell_pair2cs_ids)
            log.info(f'Cell pairs written to pickle')

    del filtered_cs_ids
    gc.collect()

    if dataset == 'test':
        offset = 2000
        nr_samples = 1000
        log.info(f'Offset and nr_samples adapted to test set.')
    else:
        offset = 0
        nr_samples = 2000

    # setup parallelization parameters
    log.info(f'Using {n_proc} processors on {len(cell_pair2cs_ids)} objects')
    chunksize = nr_samples // n_proc
    proc_slices = []

    for i_proc in range(n_proc):
        chunkstart = int(offset + (i_proc * chunksize))
        # make sure to include the division remainder for the last process
        chunkend = int(offset + (i_proc + 1) * chunksize) if i_proc < n_proc - 1 else int(offset + nr_samples)
        proc_slices.append(np.s_[chunkstart:chunkend])

    log.info(proc_slices)
    # set of ssv_ids to find entries faster
    ssv_ids_set = set(ssd.ssv_ids)
    params = [(cell_pair2cs_ids, cell_pairs, slice, cs_dataset, ssv_ids_set, cs_merge_radii) for slice in proc_slices]
    log.info(len(params))

    start = timeit.default_timer()
    log.info(f'Sample generation started with {len(params)} tasks')

    # process_map(create_labeled_points, params, max_workers=n_proc)
    # with ProcessPoolExecutor(max_workers=n_proc) as executor:
    #     executor.map(create_labeled_points, params)

    running_tasks = [mp.Process(target=create_labeled_points, args=param) for param in params]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

    end = timeit.default_timer()
    log.info(f'The whole GT generation took {(end-start)/60}')
