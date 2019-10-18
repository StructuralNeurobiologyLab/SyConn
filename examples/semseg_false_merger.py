from syconn.reps.super_segmentation import *
from syconn.reps.super_segmentation_helper import semseg_of_sso_nocache
from syconn.proc.ssd_assembly import init_sso_from_kzip
from syconn.handler.prediction import get_semseg_axon_model
import numpy as np
import os, glob, re
import argparse
import random
import zipfile, yaml
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


def get_confidence(sso, cc_pairs) -> (list, list):
    """
    For each merger location

    Parameters
    ----------
    sso : SuperSegmentationObject
        Either get from SuperSegmentationDataset or get from init_sso_from_kzip()
    cc_pairs: list
        A list containing two sets of skeleton nodes belonging to one merger.
        'cc' stands for 'connected_component'
    :return:

    Returns
    -------
    """
    # labeled_views = sso.load_views(view_props['semseg_key'])
    # unique, counts = np.unique(labeled_views, return_counts=True)
    # dict_label2count = dict(zip(unique, counts))

    # get the coordinates of all vertices
    vertices_flat = sso.mesh[1]
    vertices = vertices_flat.reshape((-1, 3))
    # get the label of all vertices
    ld = sso.label_dict('vertex')
    labeled_vertices = ld[view_props['semseg_key']]
    assert len(vertices) == len(labeled_vertices)

    vertices_kdtree = cKDTree(vertices)
    confidence_list = list()
    merger_location_list = list()
    for merger in cc_pairs:
        mid_node1 = sorted(list(merger[0]))[len(merger[0]) // 2]
        mid_node2 = sorted(list(merger[1]))[len(merger[1]) // 2]
        central_merger_location = (merger_idx2coord[mid_node1] + merger_idx2coord[mid_node2]) / 2
        vert_ixs = vertices_kdtree.query_ball_point(central_merger_location, r=1.5e3)
        unique, counts = np.unique(labeled_vertices[vert_ixs], return_counts=True)
        dict_label2count = dict(zip(unique, counts))
        try:
            _ = dict_label2count[1]
            _ = dict_label2count[0]
        except:
            return merger_location_list, "error in getting confidence"
        confidence = dict_label2count[1] / (dict_label2count[0] + dict_label2count[1])
        confidence_list.append(confidence)
        merger_location_list.append(list(central_merger_location))

    return merger_location_list, confidence_list

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
    parser.add_argument('--dest_path', type=str,
                        default=os.path.expanduser("~/predicted_merger_TEST_BC/"),
                        help='path in where the output kzip file should be stored.')
    args = parser.parse_args()

    # path to cell reconstruction k.zip
    file_names = get_all_fname(os.path.expanduser(args.kzip))
    file_names = check_kzip_completeness(os.path.expanduser(args.kzip), file_names)

    # # TEST file
    file_names = ['/merged60_cells789872_8671124.k.zip']
    # hard to detect merger:
    # file_names = ['/merged49_cells117696_23408910.k.zip', '/merged95_cells12626016_12986527.k.zip',
    #               '/merged29_cells14432502_15006702.k.zip', '/merged38_cells17356261_26788561.k.zip']
    # file_names = ['/merged76_cells9799549_10889214.k.zip']
    # file_names = file_names[:20]
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
    map_props = global_params.config['merger']['map_properties_merger']
    view_props["verbose"] = True
    # ======= TEST ==========
    # view_props["comp_window"] = 10240 * 3
    # view_props["nb_views"] = 3
    # map_props["use_ratio"] = False
    # =======================
    print("comp_window: {}".format(view_props["comp_window"]))
    # dest_predicted_merger = os.path.expanduser("~") + '/predicted_merger_test/'
    dest_predicted_merger = args.dest_path
    if not os.path.isdir(dest_predicted_merger):
        os.makedirs(dest_predicted_merger)
    report = dict()

    count = 0
    tic = timeit.default_timer()
    for fname in tqdm.tqdm(file_names):
        count += 1
        # load SSO instance from k.zip file
        cell_kzip_fn = os.path.abspath(os.path.expanduser(args.kzip)) + fname
        sso = init_sso_from_kzip(cell_kzip_fn, sso_id=1)
        # get cell ids
        cell_ids = re.findall(r"(\d+)", fname)[-2:]
        mesh_fname = "mesh" + str(count) + "_" + "_".join(cell_ids) + ".k.zip"

        # run prediction and store result in new kzip
        semseg_of_sso_nocache(sso, dest_path=dest_predicted_merger + mesh_fname, model=m,
                              **view_props)
        node_preds = sso.semseg_for_coords(
            sso.skeleton['nodes'], view_props['semseg_key'],
            **global_params.config['merger']['map_properties_merger'])
        sso.skeleton[view_props['semseg_key']] = node_preds

        merger_idx2coord = dict()
        all_skeleton_nodes = sso.skeleton['nodes'] * sso.scaling
        for i in range(len(node_preds)):
            if node_preds[i] == 1:
                merger_idx2coord[i] = sso.skeleton['nodes'][i] * sso.scaling

        # create a graph of the whole skeleton
        skeleton_graph = sso.weighted_graph([view_props['semseg_key']])
        assert len(skeleton_graph.nodes()) == len(sso.skeleton['nodes'])
        # keep only the merger nodes in the graph
        for node in list(skeleton_graph.nodes()):
            if node_preds[node] == 0:
                skeleton_graph.remove_node(node)
        assert len(skeleton_graph.nodes()) == len(merger_idx2coord)
        # determine how many mergers are detected by calculating how many connected_components exist
        # TODO: this logic of cc_pair should be changed when dealing with "real-false-merger-data", there you won't need to find pairs
        cc_list = sorted(nx.connected_components(skeleton_graph), key=len, reverse=True)
        # determine which two pairs of connected_components belongs to the same merger:
        cc_pairs = list()
        node_kdtree = cKDTree(all_skeleton_nodes)
        for i in range(len(cc_list)):
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

        # Combine mesh and skeleton kzip into on kzip file
        z_mesh = zipfile.ZipFile(dest_predicted_merger + mesh_fname, 'a')
        z_skeleton = zipfile.ZipFile(dest_predicted_merger + skeleton_fname, 'r')
        [z_mesh.writestr(t[0], t[1].read()) for t in ((n, z_skeleton.open(n)) for n in z_skeleton.namelist())]
        z_mesh.close()
        # delete the skeleton kzip
        os.remove(dest_predicted_merger + skeleton_fname)

        # get confidence of the prediction for each merger location
        merger_location_list, confidence_list = get_confidence(sso, cc_pairs)
        report[str(count) + "_" + "_".join(cell_ids)] = [str(merger_location_list), str(confidence_list)]

    toc = timeit.default_timer()
    print("Time elapsed: {}".format(toc - tic))

    with open(os.path.expanduser('~') + r'/merger_pred_report100_ratio.yml', 'w') as outfile:
        yaml.dump(report, outfile, default_flow_style=False)
        print("Result stored in merger_pred_report.yml")