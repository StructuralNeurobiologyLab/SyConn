from syconn import global_params
global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v6/'
cs_version = 'agg_0'
# global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/'
# cs_version = '0'
# global_params.wd = '/home/kloping/wholebrain/songbird/j0126/areaxfs_v6/'  # local test

import numpy as np
import timeit
from scipy.spatial import cKDTree
from typing import Union, Tuple, List, Optional, Dict, Generator, Any

from syconn.reps.super_segmentation import *
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.meshes import merge_meshes

# helper function for testing
def write_dict_to_txt(dict, fname):
    f = open(fname, "w")
    f.write(str(dict))
    f.close()

def read_dict_from_txt(fname: str):
    f = open(fname, 'r')
    return eval(f.read())

def cs_partner(id) -> Optional[List[int]]:
    """
    Contact site specific attribute.
    Returns:
        return the IDs of two
        supervoxels which are part of the contact site.
    """
    partner = [id >> 32]
    partner.append(id - (partner[0] << 32))
    return partner

def merge_superseg_objects(cell_obj1, cell_obj2):

    # TODO: test why working_dir='/tmp/' raise some errors
    # merge meshes
    merged_cell = SuperSegmentationObject(ssv_id=-1, working_dir=None, version='tmp')
    for mesh_type in ['sv', 'sj', 'syn_ssv', 'vc', 'mi']:
        mesh1 = cell_obj1.load_mesh(mesh_type)
        mesh2 = cell_obj2.load_mesh(mesh_type)
        ind_lst = [mesh1[0], mesh2[0]]
        vert_lst = [mesh1[1], mesh2[1]]
        merged_cell._meshes[mesh_type] = merge_meshes(ind_lst, vert_lst)
        merged_cell._meshes[mesh_type] += ([None, None], ) # add normals
    
    # merge skeletons
    merged_cell.skeleton = {}
    cell_obj1.load_skeleton()
    cell_obj2.load_skeleton()
    merged_cell.skeleton['edges'] = np.concatenate([cell_obj1.skeleton['edges'],
                                                    cell_obj2.skeleton['edges'] +
                                                    len(cell_obj1.skeleton['nodes'])]) # additional offset
    merged_cell.skeleton['nodes'] = np.concatenate([cell_obj1.skeleton['nodes'],
                                                    cell_obj2.skeleton['nodes']])
    merged_cell.skeleton['diameters'] = np.concatenate([cell_obj1.skeleton['diameters'],
                                                        cell_obj2.skeleton['diameters']])
    return merged_cell


if __name__ == "__main__":

    ssd = SuperSegmentationDataset() # class holding all cell representations

    sd_synssv = SegmentationDataset(obj_type='syn_ssv')  # class holding all synapse candidates between cells
    contact_sites = SegmentationDataset(obj_type='cs', version=cs_version)  # class holding all contact-site between SVs
    dict_sv2ssv = ssd.mapping_dict_reversed  # dict: {supervoxel : super-supervoxel}

    cell_id2cs_ids = {}
    cs_id2cell_ids = {}
    # Loop through all contact_sites ids and if two corresponding cells are found
    # store the cell_id and corresponding cs_id into dictionary
    print("Creating cell_id2cs_ids lookup table")
    tic = timeit.default_timer()
    for cs_id in contact_sites.ids:
        sv_partner = cs_partner(int(cs_id))
        if sv_partner[0] in dict_sv2ssv and sv_partner[1] in dict_sv2ssv:
            c1 = dict_sv2ssv[sv_partner[0]]
            c2 = dict_sv2ssv[sv_partner[1]]

            # store cell_id and cs_id into dictionaries
            if c1 in cell_id2cs_ids:
                cell_id2cs_ids[c1].add(cs_id)
            else:
                cell_id2cs_ids[c1] = {cs_id}
            if c2 in cell_id2cs_ids:
                cell_id2cs_ids[c2].add(cs_id)
            else:
                cell_id2cs_ids[c2] = {cs_id}
            # TODO: (c1, c2) and (c2, c1) should be the same
            if cs_id in cs_id2cell_ids:
                cs_id2cell_ids[cs_id].add((c1, c2))
            else:
                cs_id2cell_ids[cs_id] = {(c1, c2)}
    toc = timeit.default_timer()
    print("Time elapsed: {}".format(toc - tic))
    
    print("Generating merged cells")
    # 200 ~ 5000 cells
    count = 0
    progress = 0
    sampled_ids = sd_synssv.ids[:100]
    for syn_id in sampled_ids:
        syn_obj = sd_synssv.get_segmentation_object(syn_id)
        syn_obj.load_attr_dict()
        c1, c2 = syn_obj.attr_dict['neuron_partners']

        cell_obj1, cell_obj2 = ssd.get_super_segmentation_object([c1, c2])
        merged_cell = merge_superseg_objects(cell_obj1, cell_obj2)
        merged_cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling  # coordinates of all nodes

        # labels:
        # 2 for false-merger, 1 for true merger, 0 for ignore
        node_labels = np.zeros((len(merged_cell_nodes), ))

        syn_coord = syn_obj.rep_coord * merged_cell.scaling
        # Find all the common contact-sites of two cells, and get their coordinates
        cs_coord_list = []
        common_cs = cell_id2cs_ids[c1].intersection(cell_id2cs_ids[c2])
        for cs_id in common_cs:
            cs_obj = contact_sites.get_segmentation_object(cs_id)
            # TODO: determine if the contact is big enough
            # cs.size, cs.mesh_area --> threshhold 0.1
            cs_coord = cs_obj.rep_coord * merged_cell.scaling
            cs_coord_list.append(cs_coord)

        # find medium cube around artificial merger and set it to 1 (true merger)
        kdtree = cKDTree(merged_cell_nodes)
        # find all skeleton nodes which are close to the synapse and all contact-sites
        ixs = kdtree.query_ball_point(syn_coord, r=30e3)  # nanometer
        node_labels[ixs] = int(1)
        for cs_coord in cs_coord_list:
            ixs = kdtree.query_ball_point(cs_coord, r=30e3)
            node_labels[ixs] = int(1)

        # find small cube around artificial merger and set it to 2 (false merger)
        ixs = kdtree.query_ball_point(syn_coord, r=1.5e3)  # nanometer
        node_labels[ixs] = int(2)
        for cs_coord in cs_coord_list:
            ixs = kdtree.query_ball_point(cs_coord, r=1.5e3)
            node_labels[ixs] = int(2)

        # write out annotated skeletons to ['merger_gt']
        merged_cell.skeleton['merger_gt'] = node_labels

        # write all data to kzip (skeleton and mesh)
        fname = f'syn{syn_obj.id}_cells{cell_obj1.id}_{cell_obj2.id}.k.zip'
        target_fnames = []
        tmp_dest_p = []
        tmp_dest_p.append('{}_{}.pkl'.format(fname, 'meta'))
        target_fnames.append('{}.pkl'.format('meta'))
        write_obj2pkl(tmp_dest_p[-1], {'version_dict': merged_cell.version_dict,
                                       'scaling': merged_cell.scaling,
                                       'working_dir': merged_cell.working_dir,
                                       'sso_id': merged_cell.id})
        data2kzip(fname, tmp_dest_p, target_fnames)

        merged_cell.save_skeleton_to_kzip(fname, additional_keys=['merger_gt'])
        merged_cell.meshes2kzip(fname)
        # TODO: export2kzip should run through

        count += 1
        if count == sampled_ids // 10:
            count = 0
            progress += 1
            print("{}/{} cells generated".format(progress * 10, len(sampled_ids)))


