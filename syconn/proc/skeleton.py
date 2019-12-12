
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.handler.basics import kd_factory
import numpy as np
import time
import kimimaro
from cloudvolume import PrecomputedSkeleton
from syconn.handler.basics import kd_factory
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode
import numpy as np
import os as os
import kimimaro
import tqdm
import time
from syconn.handler.basics import load_pkl2obj, write_obj2pkl
import networkx as nx

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

#code from https://pypi.org/project/kimimaro/
def kimimaro_skelgen(cube_size, cube_offset):
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    kd = kd_factory(global_params.config.kd_seg_path)
    seg = kd.from_overlaycubes_to_matrix(cube_size, cube_offset, mag=2)
    seg_cell = np.zeros_like(seg)
    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            for z in range(seg.shape[2]):
                try:
                    seg_cell[x, y, z] = ssd.mapping_dict_reversed[seg[x, y, z]]
                except KeyError:
                    seg_cell[x, y, z] = 0
    #kimimaro code

    skels = kimimaro.skeletonize(
        seg_cell,
        teasar_params={
            'scale': 4,
            'const': 100,  # physical units
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100,  # physical units
            'soma_acceptance_threshold': 3500,  # physical units
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300,  # physical units
            'max_paths': 50,  # default None
        },
        # object_ids=[ ... ], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=1000,  # skip connected components with fewer than this many voxels
        anisotropy=(20, 20, 40),  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        progress=True,  # default False, show progress bar
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )


    for ii in skels:
        cell = skels[ii]
        for i,v in enumerate(cell.vertices):
            c = cell.vertices[i]
            #add cube_offset in physical coordinates
            cell.vertices[i] = np.array([int(c[0]+cube_offset[0]*20), int(c[1]+cube_offset[1]*20), int(c[2]+cube_offset[2]*40)])
        # cloud_volume docu: " reduce size of skeleton by factor of 2, preserves branch and end points" link:https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeleton
        #cell = cell.downsample(2)
        # code from sparsify_skeleton_fast in syconn.procs.super_segmentation_helper
        # modify for kimimaro_skeletons
        """
        start = time.time()
        dot_prod_thresh: float = 0.8,
                           max_dist_thresh: Union[int, float] = 500,
                           min_dist_thresh: Union[int, float] = 50
        degree_dict = {i: 0 for i, iv in enumerate(cell.vertices)}
        neighbour_dict = {i: [] for i in list(degree_dict.keys())}
        edges = np.hstack(cell.edges)
        for for i, iv in enumerate(skel.vertices):
            amount = np.where(edges == i)[0]
            degree_dict[i] = amount.size
            for e in cell.edges:
                if i not in e:
                    continue
                if i == e[0]:
                    neighbour_dict[i].append(e[1])
                if i == e[1]:
                    neighbour_dict[i].append(e[0])
        n_nodes_start = len(cell.vertices)
    
        change = 1

        while change > 0:
            change = 0
            
            
            visiting_nodes = list({k for k, v in dict(degree_dict.items() if v == 2})
            for visiting_node in visiting_nodes:
                neighbours = [n for n in neighbour_dict(visiting_node)]
                if degree_dict[visiting_node] == 2:
                    left_node = neighbours[0]
                    right_node = neighbours[1]
                    vector_left_node = np.array(
                        [int(cell.vertices[left_node][ix]) - int(cell.vertices[visiting_node][ix])
                         for
                         ix in range(3)]) 
                    vector_right_node = np.array([int(cell.vertices[right_node][ix]) -
                                                  int(cell.vertices[visiting_node][ix]) for ix in
                                                  range(3)]) 

                    dot_prod = np.dot(vector_left_node / np.linalg.norm(vector_left_node),
                                      vector_right_node / np.linalg.norm(vector_right_node))
                    dist = np.linalg.norm([int(cell.vertices[right_node][ix]) - int(
                        cell.vertices[left_node][ix]) for ix in range(3)])

                    if (abs(dot_prod) > dot_prod_thresh and dist < max_dist_thresh) or dist <= min_dist_thresh:
                        skel_nx.remove_node(visiting_node)
                        skel_nx.add_edge(left_node, right_node)
                        change += 1
        log_reps.debug(f'sparsening took {time.time() - start}. Reduced {n_nodes_start} to '
                       f'{skel_nx.number_of_nodes()} nodes')
        """

    return skels

# load files and merge dictionaries

def kimimaro_mergeskels(path_list, cell_id):
    skel_list = []
    for f in path_list:
        part_dict = load_pkl2obj(f)
        skel_list.append(part_dict[int(cell_id)])
    # merge skeletons to one connected component
    # a set of skeletons produced from the same label id
    skel = PrecomputedSkeleton.simple_merge(skel_list).consolidate()
    skel = kimimaro.postprocess(
        skel,
        dust_threshold=1000,  # physical units
        tick_threshold=3500  # physical units
    )

    # Split input skeletons into connected components and
    # then join the two nearest vertices within `radius` distance
    # of each other until there is only a single connected component
    # or no pairs of points nearer than `radius` exist.
    # Fuse all remaining components into a single skeleton.
    skel = kimimaro.join_close_components(skel_list, radius=None)  # no threshold
    #cloud_volume docu: " reduce size of skeleton by factor of 2, preserves branch and end points" link:https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeleton
    skel = skel.downsample(2) #better suited in function above with part of skels. Doesn't work there.
    degree_dict = {i: 0 for i, iv in enumerate(skel.vertices)}
    neighbour_dict = {i: [] for i in list(degree_dict.keys())}
    edges = np.hstack(skel.edges)
    #rewrite to netkx_graph as well
    nx_skel = nx.Graph()
    for i, iv in enumerate(skel.vertices):
        amount = np.where(edges == i)[0]
        degree_dict[i] = amount.size
        nx_skel.add_node(i, coord = iv)
        for e in skel.edges:
            if i not in e:
                continue
            if i == e[0]:
                neighbour_dict[i].append(e[1])
            if i == e[1]:
                neighbour_dict[i].append(e[0])
    nx_skel.add_edges_from(skel.edges)

    return skel, nx_skel,degree_dict, neighbour_dict




def kimimaro_skels_tokzip(cell_skel, cell_id, zipname):
    #write to zip file
    skel = Skeleton()
    anno = SkeletonAnnotation()
    #anno.scaling = global_params.config['scaling']
    node_mapping = {}
    cv = cell_skel.vertices
    pbar = tqdm.tqdm(total=len(cv) + len(cell_skel.edges))
    for i,v in enumerate(cv):
        c = cv[i]
        n = SkeletonNode().from_scratch(anno, int(c[0]/10)+5400, int(c[1]/10)+ 5900, int(c[2]/20)+3000)
        node_mapping[i] = n
        anno.addNode(n)
        pbar.update(1)
    for e in cell_skel.edges:
        anno.addEdge(node_mapping[e[0]], node_mapping[e[1]])
        pbar.update(1)
    skel.add_annotation(anno)
    skel.to_kzip('%s/kzip_%.i.k.zip' % (zipname, cell_id))