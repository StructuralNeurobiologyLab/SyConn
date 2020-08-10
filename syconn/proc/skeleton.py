import pickle as pkl
from typing import Optional

import numpy as np
import tqdm
from scipy import ndimage
import kimimaro
from cloudvolume import PrecomputedSkeleton
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.basics import load_pkl2obj, kd_factory
from syconn.proc.image import multi_mop_backgroundonly
from syconn import global_params


def kimimaro_skelgen(cube_size, cube_offset, overlap, cube_of_interest_bb) -> dict:
    """
    code from https://pypi.org/project/kimimaro/

    Args:
        cube_size: size of processed cube in mag 2 voxels.
        cube_offset: starting point of cubes (in mag 2 voxel coordinates)
        overlap: In mag 2 voxels.
        cube_of_interest_bb: Partial volume of the data set. Bounding box in mag 1 voxels: (lower
            coord, upper coord)

    Returns:
        Skeleton with nodes, edges in physical parameters

    """
    # volume to be processed in mag!
    # TODO: the factor 2 must be adapted when using anisotropic downsampling of the
    #  KnossosDataset
    dataset_size = (cube_of_interest_bb[1] - cube_of_interest_bb[0]) // 2

    overlap = np.array(overlap, dtype=np.int)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    kd = kd_factory(global_params.config.kd_seg_path)
    if np.all(cube_size < dataset_size):
        # this is in mag2!
        cube_size_ov = cube_size + 2 * overlap
        # offset is converted to mag 2
        # TODO: the factor 2 must be adapted when using anisotropic downsampling of the
        #  KnossosDataset
        cube_offset_ov = cube_offset - overlap
        seg = kd.load_seg(size=cube_size_ov*2, offset=np.array(cube_offset_ov)*2,
                          mag=2).swapaxes(0, 2)
    else:
        # TODO: the factor 2 must be adapted when using anisotropic downsampling of the
        #  KnossosDataset
        # converting mag 2 units to mag 1 (required by load_seg)
        seg = kd.load_seg(size=cube_size*2, offset=np.array(cube_offset)*2, mag=2).swapaxes(0, 2)

    seg_cell = np.zeros_like(seg)
    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            for z in range(seg.shape[2]):
                try:
                    seg_cell[x, y, z] = ssd.mapping_dict_reversed[seg[x, y, z]]
                except KeyError:
                    seg_cell[x, y, z] = 0

    seg_cell = multi_mop_backgroundonly(ndimage.binary_fill_holes, seg_cell, iterations=None)

    if np.all(cube_size < dataset_size) is True:
        seg_cell = seg_cell[overlap[0]:-overlap[0], overlap[1]:-overlap[1], overlap[2]:-overlap[2]]

    # kimimaro code
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
        anisotropy=kd.scales[1],  # index 1 is mag 2
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        progress=False,  # show progress bar
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )

    for ii in skels:
        cell = skels[ii]
        for i, v in enumerate(cell.vertices):
            c = cell.vertices[i]  # already in physical coordinates (nm)
            # now add the offset in physical coordinates, both are originally in mag 2
            # TODO: the factor 1/2 must be adapted when using anisotropic downsampling of the
            #  KnossosDataset
            c = np.array(c + (cube_offset - cube_of_interest_bb[0] // 2) * kd.scales[1],
                         dtype=np.int)
            cell.vertices[i] = c
        # cloud_volume docu: " reduce size of skeleton by factor of 2, preserves branch and end
        # points" link:https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeleton
        # cell = cell.downsample(2)
        # code from sparsify_skeleton_fast in syconn.procs.super_segmentation_helper
        # modify for kimimaro_skeletons

    return skels


def kimimaro_mergeskels(path_list, cell_id):
    """
    For debugging. Load files and merge dictionaries.

    Args:
        path_list: list of paths to locations for partial skeletons generated by kimimaro
        cell_id: ssv.ids

    Returns: merged skeletons with nodes in physical parameters

    """
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
    # better suited in function above with part of skels. Doesn't work there.
    skel = skel.downsample(4)
    # Split input skeletons into connected components and
    # then join the two nearest vertices within `radius` distance
    # of each other until there is only a single connected component
    # or no pairs of points nearer than `radius` exist.
    # Fuse all remaining components into a single skeleton.
    skel = kimimaro.join_close_components(skel, radius=None)  # no threshold
    # cloud_volume docu: " reduce size of skeleton by factor of 2, preserves branch and end points"
    # link:https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeleton
    # skel = skel.downsample(4) #better suited in function above with part of skels. Doesn't work
    # there.
    degree_dict = {i: 0 for i, iv in enumerate(skel.vertices)}
    neighbour_dict = {i: [] for i in list(degree_dict.keys())}

    return skel, degree_dict, neighbour_dict


def kimimaro_skels_tokzip(cell_skel, cell_id, zipname):
    # write to zip file
    skel = Skeleton()
    anno = SkeletonAnnotation()
    # anno.scaling = global_params.config['scaling']
    node_mapping = {}
    cv = cell_skel.vertices
    pbar = tqdm.tqdm(total=len(cv) + len(cell_skel.edges))
    for i, v in enumerate(cv):
        n = SkeletonNode().from_scratch(anno, int((v[0])+54000), int((v[1]) + 59000),
                                        int((v[2]) + 3000*20))
        # above only for example_cube with certain offset
        # n = SkeletonNode().from_scratch(anno, int(c[0] / 10), int(c[1] / 10), int(c[2] / 20) )
        # pdb.set_trace()
        node_mapping[i] = n
        anno.addNode(n)
        pbar.update(1)
    for e in cell_skel.edges:
        anno.addEdge(node_mapping[e[0]], node_mapping[e[1]])
        pbar.update(1)
    skel.add_annotation(anno)
    skel.to_kzip('%s/kzip_%.i.k.zip' % (zipname, cell_id), force_overwrite=True)
