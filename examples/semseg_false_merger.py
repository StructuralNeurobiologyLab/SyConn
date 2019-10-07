from syconn.reps.super_segmentation import *
from syconn.reps.super_segmentation_helper import semseg_of_sso_nocache
from syconn.proc.ssd_assembly import init_sso_from_kzip
from syconn.handler.prediction import get_semseg_axon_model
import numpy as np
import os, glob, re
import argparse
import random
import zipfile
import tqdm
import timeit
import networkx as nx
from scipy.spatial import cKDTree

def check_kzip_completeness(data_path: str, fnames: list):
    """
    Check the completeness of the kzip file, each kzip file must contain:
    meta.pkl, mi.pkl, sj.pkl, sv.pkl, sv.pkl, vc.pkl and annotation.xml
    """
    filtered_fnames = []
    for file_name in fnames:
        kzip_path = data_path + '/' + file_name
        zip = zipfile.ZipFile(kzip_path)
        files = zip.namelist()

        if zip is not None and set(files) == {'skeleton.pkl', 'meta.pkl', 'annotation.xml',
                                              'sj.ply', 'vc.ply', 'mi.ply', 'sv.ply'}:
            filtered_fnames.append(file_name)

    return filtered_fnames

def get_all_fname(data_path):
    """
    Get file names in the given path
    """
    fnames = []
    cell_combinations = set()
    os.chdir(data_path)

    # count = 0
    for file in glob.glob("*.k.zip"):
        # filter out duplicate merged_cells
        cell_ids = re.findall(r"(\d+)", data_path + file)[-2:]
        if tuple(cell_ids) in cell_combinations or tuple(cell_ids[::-1]) in cell_combinations:
            continue
        cell_combinations.add(tuple(cell_ids))

        fnames.append('/' + file)
        # count += 1
    return fnames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SyConn example run')
    # parser.add_argument('--working_dir', type=str,
    #                     default=os.path.expanduser("/wholebrain/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/"),
    #                     help='Working directory of SyConn')
    parser.add_argument('--kzip', type=str,
                        default=os.path.expanduser("~/merger_testset_kzip_v10_v4b_base_20180214_full_agglo_cbsplit/"),
                        help='path to kzip file which contains a cell reconstruction (see '
                             'SuperSegmentationObject().export2kzip())')
    parser.add_argument('--modelpath', type=str,
                        default=os.path.expanduser("~/e3training/merger_FCN_v10_2/"),
                        help='path to the model.pt file of the trained model.')
    args = parser.parse_args()

    # path to working directory of example cube - required to load pretrained models
    # path_to_workingdir = os.path.expanduser(args.working_dir)

    # path to cell reconstruction k.zip
    file_names = get_all_fname(os.path.abspath(os.path.expanduser(args.kzip)))
    # file_names = get_all_fname(os.path.expanduser(args.kzip))
    file_names = check_kzip_completeness(os.path.expanduser(args.kzip), file_names)

    # =====================================
    # TEST file
    # =====================================
    file_names = ['/merged60_cells789872_8671124.k.zip']

    print("{} kzip files detected in: {}".format(len(file_names), args.kzip))

    # set working directory to obtain models
    # global_params.wd = path_to_workingdir

    model_p = args.modelpath
    if model_p is None:
        m = get_semseg_axon_model()
    else:
        try:
            from elektronn3.models.base import InferenceModel
        except ImportError as e:
            msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
                  "com/ELEKTRONN/elektronn3' for more information.".format(e)
            raise ImportError(msg)
        m = InferenceModel(model_p)
        m._path = model_p

    # get model for false-merger detection
    view_props = global_params.config['merger']['view_properties_merger']
    view_props["verbose"] = True

    dest_predicted_merger = os.path.expanduser("~") + '/predicted_merger_test/'
    if not os.path.isdir(dest_predicted_merger):
        os.makedirs(dest_predicted_merger)

    tic = timeit.default_timer()
    for fname in tqdm.tqdm(file_names):
        # load SSO instance from k.zip file
        cell_kzip_fn = os.path.abspath(os.path.expanduser(args.kzip)) + fname
        sso = init_sso_from_kzip(cell_kzip_fn, sso_id=1)
        # get cell ids
        cell_ids = re.findall(r"(\d+)", fname)[-2:]
        output_fname = "mesh_" + "_".join(cell_ids) + ".k.zip"

        # run prediction and store result in new kzip
        semseg_of_sso_nocache(sso, dest_path=dest_predicted_merger + output_fname, model=m,
                              **view_props)
        node_preds = sso.semseg_for_coords(
            sso.skeleton['nodes'], view_props['semseg_key'],
            **global_params.config['merger']['map_properties_merger'])
        sso.skeleton[view_props['semseg_key']] = node_preds

        # TODO: define some criterion: size, confidence
        merger_idx2coord = dict()
        all_skeleton_nodes = sso.skeleton['nodes'] * sso.scaling
        for i in range(len(node_preds)):
            if node_preds[i] == 1:
                merger_idx2coord[i] = sso.skeleton['nodes'][i] * sso.scaling
        # Plan 1: randomly pick one coord to represent merger_location
        # if len(merger_idx2coord) > 0:
        #     merger_location = random.choice(merger_coord_list)

        # TODO: implement the binary closing to close the gap
        # TODO: but also tweak the k_neighbor in config.yml to get best result

        # create a graph of the whole skeleton
        skeleton_graph = sso.weighted_graph([view_props['semseg_key']])
        assert len(skeleton_graph.nodes()) == len(sso.skeleton['nodes'])
        # keep only the merger nodes in the graph
        for node in list(skeleton_graph.nodes()):
            if node_preds[node] == 0:
                skeleton_graph.remove_node(node)
        assert len(skeleton_graph.nodes()) == len(merger_idx2coord)
        # determine how many mergers are detected by calculating how many connected_components exist
        cc_list = sorted(nx.connected_components(skeleton_graph), key=len, reverse=True)
        # determine which two pairs of connected_components belongs to the same merger:
        cc_pairs = list()
        node_kdtree = cKDTree(all_skeleton_nodes)
        for i in range(len(cc_list) - 1):
            cc = cc_list[i]
            if len(cc) < 3:
                # discard the connected_component which contains less than 3 nodes
                for node_id in cc:
                    node_preds[node_id] = 0
                continue
            # get the middle element of cc
            mid = sorted(list(cc))[len(cc) // 2]
            ixs = sorted(node_kdtree.query_ball_point(merger_idx2coord[mid], r=1.5e3))
            for j in range(i+1, len(cc_list)):
                cc2 = cc_list[j]
                if cc2 != cc:
                    if len(set(ixs).intersection(cc2)) > 2:
                        cc_pairs.append((cc, cc2))
        # refresh the node prediction after the filtering out the isolated labeled_nodes
        sso.skeleton[view_props['semseg_key']] = node_preds
        skeleton_fname = "skeleton_" + "_".join(cell_ids) + ".k.zip"
        sso.save_skeleton_to_kzip(dest_path=dest_predicted_merger + skeleton_fname,
                                  additional_keys=view_props['semseg_key'])

        # ================================
        # Information for confidence
        # ================================
        num_merger_nodes = len(merger_idx2coord)
        # if num_merger_nodes

        labeled_views = sso.load_views(view_props['semseg_key'])
        unique, counts = np.unique(labeled_views, return_counts=True)
        dict_label2count = dict(zip(unique, counts))

        # get the coordinates of all vertices
        vertices_flat = sso.mesh[1]
        vertices = vertices_flat.reshape((-1, 3))
        # get the label of all vertices
        ld = sso.label_dict('vertex')
        labeled_vertices = ld[view_props['semseg_key']]
        assert len(vertices) == len(labeled_vertices)

        vertices_kdtree = cKDTree(vertices)
        confidence_list = list()
        for merger in cc_pairs:
            mid_node1 = sorted(list(merger[0]))[len(merger[0]) // 2]
            mid_node2 = sorted(list(merger[1]))[len(merger[1]) // 2]
            central_merger_location = (merger_idx2coord[mid_node1] + merger_idx2coord[mid_node2]) / 2
            vert_ixs = vertices_kdtree.query_ball_point(central_merger_location, r=1.5e3)
            unique, counts = np.unique(labeled_vertices[vert_ixs], return_counts=True)
            dict_label2count = dict(zip(unique, counts))
            confidence = dict_label2count[1] / (dict_label2count[0] + dict_label2count[1])
            confidence_list.append(confidence)

        import pdb
        pdb.set_trace()

        # ================================
        # Information for confidence
        # ================================


    toc = timeit.default_timer()
    print("Time elapsed: {}".format(toc - tic))