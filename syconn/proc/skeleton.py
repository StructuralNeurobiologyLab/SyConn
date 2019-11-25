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

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

#code from https://pypi.org/project/kimimaro/
def kimimaro_skelgen(cube_size = np.array([200, 200, 200]), cube_offset=[1000, 1000, 1000]):
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    kd = kd_factory(global_params.config.kd_seg_path)
    seg = kd.from_overlaycubes_to_matrix(np.array(cube_size, cube_offset, mag=2))
    seg_cell = np.zeros_like(seg)
    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            for z in range(seg.shape[2]):
                try:
                    seg_cell[x, y, z] = ssd.mapping_dict_reversed[seg[x, y, z]]
                except KeyError:
                    seg_cell[x, y, z] = 0
    #kimamoro code

    skels = kimimaro.skeletonize(
        seg_cell,
        teasar_params={
            'scale': 4,
            'const': 500,  # physical units
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

    return skels

# load files and merge dictionaries

def kimimaro_mergeskels(path_list, cell_id):
    skel_list = []
    for f in path_list:
        part_dict = pkl.load(f)
        skel_list.append(part_dict[cell_id])
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

    return skel

def kimimaro_skels_tokzip(skels, cell_id, zipname):
    #write to zip file
    if not os.path.exists(zipname):
        os.mkdir(zipname)
    for ii in skels:
        cell = skels[ii]
        skel = Skeleton()
        anno = SkeletonAnnotation()
        #anno.scaling = global_params.config['scaling']
        node_mapping = {}
        pbar = tqdm.tqdm(total=len(cell.vertices) + len(cell.edges))
        for i,v in enumerate(cell.vertices):
            c = cell.vertices[i] + 1
            n = SkeletonNode().from_scratch(anno, (c/[10, 10, 20]))
            node_mapping[i] = n
            anno.addNode(n)
            pbar.update(1)
        for e in cell.edges:
            anno.addEdge(node_mapping[e[0]], node_mapping[e[1]])
            pbar.update(1)
        skel.add_annotation(anno)
        skel.to_kzip('%.50s/kzip_%.i.k.zip' % (zipname, cell_id))