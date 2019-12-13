import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import majority_vote_compartments_kimimaro, \
    majorityvote_skeleton_property_kimimaro
from syconn.reps.super_segmentation_object import SuperSegmentationObject
from syconn import global_params

#code from QSUB_map_semsegaxoness2skel

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

ssv_ids = args[0]
pred_key_appendix = args[1]
map_properties = global_params.config['compartments']['map_properties_semsegax']
pred_key = global_params.config['compartments']['view_properties_semsegax']['semseg_key']
max_dist = global_params.config['compartments']['dist_axoness_averaging']
for ssv_id in ssv_ids:
    sso = SuperSegmentationObject(ssv_id, working_dir=global_params.config.working_dir)
    sso.load_skeleton_kimimaro(skel = "knx_skeleton_dict")
    if not sso.knx_skeleton_dict is None or len(sso.knx_skeleton_dict["nodes"]) >= 2:
        # vertex predictions
        node_preds = sso.knx_skeleton_dict["axoness"]
        # perform average only on axon dendrite and soma predictions
        nodes_ax_den_so = np.array(node_preds, dtype=np.int)
        # set en-passant and terminal boutons to axon class.
        nodes_ax_den_so[nodes_ax_den_so == 3] = 1
        nodes_ax_den_so[nodes_ax_den_so == 4] = 1
        sso.knx_skeleton_dict[pred_key] = node_preds

        # average along skeleton, stored as: "{}_avg{}".format(pred_key, max_dist)
        majorityvote_skeleton_property_kimimaro(sso, prop_key=pred_key,
                                       max_dist=max_dist)
        # suffix '_avg{}' is added by `_average_node_axoness_views`
        nodes_ax_den_so = sso.knx_skeleton_dict["{}_avg{}".format(pred_key, max_dist)]
        # recover bouton predictions within axons and store smoothed result
        nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
        nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
        sso.knx_skeleton_dict["{}_avg{}".format(pred_key, max_dist)] = nodes_ax_den_so

        # will create a compartment majority voting after removing all soma nodes
        # the restul will be written to: ``ax_pred_key + "_comp_maj"``
        majority_vote_compartments_kimimaro(sso, "{}_avg{}".format(pred_key, max_dist))
        nodes_ax_den_so = sso.knx_skeleton_dict["{}_avg{}_comp_maj".format(pred_key, max_dist)]
        # recover bouton predictions within axons and store majority result
        nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
        nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
        sso.knx_skeleton_dict["{}_avg{}_comp_maj".format(pred_key, max_dist)] = nodes_ax_den_so
        sso.save_skeleton_kimimaro(skel= "knx_skeleton_dict")

    else:
        print("Skeleton of SSV %d has no or less than two nodes." % ssv_id)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)