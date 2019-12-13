import sys
from syconn.handler.basics import load_pkl2obj
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc.skeleton import kimimaro_mergeskels, kimimaro_skels_tokzip
from syconn import global_params
from syconn.reps.super_segmentation_object import SuperSegmentationObject

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

path2results_dc, ssv_ids, zipname = args
results_dc = load_pkl2obj(path2results_dc)
for ssv_id in ssv_ids:
    ssv_id = int(ssv_id)
    combined_skel, nx_skels, degree_dict, neighbour_dict = kimimaro_mergeskels(results_dc[ssv_id], ssv_id)
    kimimaro_skels_tokzip(combined_skel,ssv_id, zipname)
    sso = SuperSegmentationObject(ssv_id, working_dir=global_params.config.working_dir)
    knx_dict = dict()
    sso.k_skeleton = combined_skel
    sso.knx_skeleton = nx_skels
    sso.knx_skeleton_dict = knx_dict
    sso.knx_skeleton_dict["neighbours"] = neighbour_dict
    sso.knx_skeleton_dict["nodes"] = combined_skel.vertices
    sso.knx_skeleton_dict["edges"] = nx_skels.edges
    sso.knx_skeleton_dict["degree"] = degree_dict
    sso.save_skeleton_kimimaro(skel = "k_skeleton")
    sso.save_skeleton_kimimaro(skel = "knx_skeleton")
    sso.save_skeleton_kimimaro(skel = "knx_skeleton_dict")
    sso.cnn_axoness2skel_kimimaro()

    # savelist = [combined_skel, nx_skels, degree_dict, neighbour_dict, ssv_id]
with open(path_out_file, "wb") as f:
    pkl.dump("0", f)

