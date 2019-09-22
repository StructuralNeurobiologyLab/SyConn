import numpy as np
import timeit
from scipy.spatial import cKDTree
import tqdm
from typing import Union, Tuple, List, Optional, Dict, Generator, Any

from syconn import global_params
from syconn.reps.super_segmentation import *
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.meshes import merge_meshes
from syconn.handler.basics import write_obj2pkl, load_pkl2obj

# Parameters

### Path to load dataset
# global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v6/'
# cs_version = 'agg_0'
global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/'
cs_version = 0
# global_params.wd = '/home/kloping/wholebrain/songbird/j0126/areaxfs_v6/'  # local test

### Path to store output kzip files
# dest_folder = os.path.expanduser("~") + "/merged_cells_kzip_v6/"
dest_folder = os.path.expanduser("~") + "/merged_cells_kzip_v10_v4b_base_20180214_full_agglo_cbsplit/"

### Path to pickle file which stores the dictionary: cell_ids2cs_ids
# path_pkl_file = os.getcwd() + '/cell_ids2cs_ids_v6.pkl'
path_pkl_file = os.getcwd() + '/cell_ids2cs_ids_v10.pkl'

# number of cs_ids to retrieve from the dataset
# if None, retrieve all cs_ids
num_cs_id = None
create_new_cs_ids = False

# number of generated cells
if num_cs_id == None:
    num_generated_cells = 50
else:
    # if num_cs_id is not None, then generate all the cells in the cell_ids list
    num_generated_cells = None


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

def create_lookup_table(num_cs_id):
    """Loop through all contact_sites ids and if two corresponding cells are found,
    store the cell_id and corresponding cs_id into dictionary

    Returns
    -------
    cell_ids2cs_ids : dict
    cell_ids : list
    """
    print("Creating cell_id2cs_ids lookup table")
    cell_id2cs_ids = dict()
    cell_ids = list()

    tic = timeit.default_timer()
    for cs_id in contact_sites.ids[:num_cs_id]:
        sv_partner = cs_partner(int(cs_id))
        if sv_partner[0] in dict_sv2ssv and sv_partner[1] in dict_sv2ssv:
            c1 = dict_sv2ssv[sv_partner[0]]
            c2 = dict_sv2ssv[sv_partner[1]]

            # store cell_id and cs_id into dictionaries
            if c1 in cell_id2cs_ids:
                cell_id2cs_ids[c1].add(cs_id)
            else:
                cell_id2cs_ids[c1] = {cs_id}
                cell_ids.append(c1)
            if c2 in cell_id2cs_ids:
                cell_id2cs_ids[c2].add(cs_id)
            else:
                cell_id2cs_ids[c2] = {cs_id}
                cell_ids.append(c2)
            # # TODO: (c1, c2) and (c2, c1) should be the same
            # if cs_id in cs_id2cell_ids:
            #     cs_id2cell_ids[cs_id].add((c1, c2))
            # else:
            #     cs_id2cell_ids[cs_id] = {(c1, c2)}
    toc = timeit.default_timer()
    print("Time elapsed: {}".format(toc - tic))
    return cell_id2cs_ids, cell_ids


if __name__ == "__main__":
    ssd = SuperSegmentationDataset() # class holding all cell representations
    contact_sites = SegmentationDataset(obj_type='cs', version=cs_version)  # class holding all contact-site between SVs
    dict_sv2ssv = ssd.mapping_dict_reversed  # dict: {supervoxel : super-supervoxel}


    if create_new_cs_ids:
        cell_id2cs_ids, cell_ids = create_lookup_table(num_cs_id)
        write_obj2pkl(path_pkl_file, cell_id2cs_ids)
    else:
        cell_id2cs_ids = dict()
        cell_ids = list()
        try:
            cell_id2cs_ids = load_pkl2obj(path_pkl_file)
            for k in cell_id2cs_ids.keys():
                cell_ids.append(k)
            print("Loading cell_ids2cs_ids from pkl file successful.")
        except:
            print("cell_ids2cs_ids.pkl not found in {}".format(path_pkl_file))
            cell_id2cs_ids, cell_ids = create_lookup_table(num_cs_id)
            write_obj2pkl(path_pkl_file, cell_id2cs_ids)

    assert len(cell_id2cs_ids) == len(cell_ids), "inconsistent length"
    print("total number of cells: {}".format(len(cell_ids)))

    print("Generating merged cells")
    count = 0
    cell_ids = cell_ids[:num_generated_cells]
    for i in tqdm.tqdm(range(len(cell_ids) - 1)):
        for j in range(i+1, len(cell_ids)):
            c1, c2 = cell_ids[i], cell_ids[j]
            assert c1 != c2, "same cells cannot be merged."

            cell_obj1, cell_obj2 = ssd.get_super_segmentation_object([c1, c2])
            merged_cell = merge_superseg_objects(cell_obj1, cell_obj2)
            merged_cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling  # coordinates of all nodes

            # labels:
            # 1 for false-merger, 0 for true merger, -1 for ignore
            node_labels = np.zeros((len(merged_cell_nodes),)) - 1

            # Find all common contact-sites of two cells
            common_cs = cell_id2cs_ids[c1].intersection(cell_id2cs_ids[c2])
            if len(common_cs) == 0:
                continue
            cs_coord_list = []
            for cs_id in common_cs:
                cs_obj = contact_sites.get_segmentation_object(cs_id)
                # TODO: determine if the contact is big enough
                # cs.size, cs.mesh_area --> threshhold 0.1
                cs_coord = cs_obj.rep_coord * merged_cell.scaling
                cs_coord_list.append(cs_coord)

            # find medium cube around artificial merger and set it to 0 (true merger)
            kdtree = cKDTree(merged_cell_nodes)
            # find all skeleton nodes which are close to all contact-sites
            for cs_coord in cs_coord_list:
                ixs = kdtree.query_ball_point(cs_coord, r=30e3)
                node_labels[ixs] = int(0)

            # find small cube around artificial merger and set it to 1 (false merger)
            for cs_coord in cs_coord_list:
                ixs = kdtree.query_ball_point(cs_coord, r=1.5e3)
                node_labels[ixs] = int(1)

            # write out annotated skeletons to ['merger_gt']
            merged_cell.skeleton['merger_gt'] = node_labels

            # write all d*ata to kzip (skeleton and mesh)
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)

            fname = dest_folder + f'/merged{count}_cells{cell_obj1.id}_{cell_obj2.id}.k.zip'
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
    print("Total merged_cells generated: {}".format(count))
