import os
import json
import argparse
import numpy as np

from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.prediction import int2str_converter, str2int_converter
from knossos_utils import KnossosDataset


info = {
    "@type": "neuroglancer_segment_properties",
    "inline": {
        "ids": None,
        "properties": [
            {
                "id": "Cell-types",
                "type": "tags",
                "tags": None,
                "values": None
            },
            {
                "id": "Volume",
                "type": "number",
                "description": "Cell volume (µm³)",
                "data_type": "float32",
                "values": None
            },
            # {
            #     "id": "TotalPathLength",
            #     "type": "number",
            #     "description": "Total path length of the cell (µm)",
            #     "data_type": "int32",
            #     "values": None
            # },
            #{
            #    "id": "MeanSynapseArea",
            #    "type": "number",
            #    "description": "Mean synapse area per cell (µm²)",
            #    "data_type": "float32",
            #    "values": None
            #},
            {
                "id": "NSynapses",
                "type": "number",
                "description": "Synapse count per cell",
                "data_type": "uint32",
                "values": None
            }
            
        ]
    }
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

parser = argparse.ArgumentParser(description='Generate info file for Neuroglancer precomputed segmentation properties')
parser.add_argument('--wd', type=str, default=None, help='SyConn working directory of the dataset')
parser.add_argument('--use_tpl_mask', action='store_true', help='Use total path length filter if there are a lot of cells in the dataset')

args = parser.parse_args()
if args.wd is not None:
    global_params.wd = args.wd
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    if "j0251" in global_params.config.working_dir:
        gt_type = "ctgt_j0251_v2"
    else:
        gt_type = "ctgt_v2"

    # load cell properties
    cts = ssd.load_numpy_data("celltype_cnn_e3")
    unique_cts = np.unique(cts)
    labels = []

    for ct in sorted(unique_cts):
        labels.append(int2str_converter(ct, gt_type=gt_type))

    if "GPi" and "GPe" in labels:
        # remap GPe and GPi to GP
        cts[cts == 7] = 6
        cts[cts == 8] = 7
        cts[cts == 9] = 8
        cts[cts == 10] = 9
        labels.remove("GPi")
        labels[labels == "GPe"] = "GP"

    if "STN" in labels:
        labels[labels == "STN"] = "exc"

    print(labels)

    sizes = ssd.load_numpy_data("size")
    vol = (sizes * np.prod(ssd.scaling)) / 1e9  # µm³

    syn_ssvs = ssd.load_numpy_data("syn_ssv")
    func = np.vectorize(len)
    syn_counts = func(syn_ssvs)
    
    total_path_length = ssd.load_numpy_data("total_edge_length")
    ssvs = ssd.ssv_ids

    if args.use_tpl_mask:
        mask = total_path_length > 150
        cts = cts[mask]
        vol = vol[mask]
        syn_counts = syn_counts[mask]
        total_path_length = total_path_length[mask]
        ssvs = ssd.ssv_ids[mask]

    info["inline"]["ids"] = list(map(str, ssvs))
    info["inline"]["properties"][0]["tags"] = labels
    info["inline"]["properties"][0]["values"] = list(map(lambda e: [e], cts))
    info["inline"]["properties"][1]["values"] = list(vol)
    # info["inline"]["properties"][2]["values"] = list(total_path_length)
    info["inline"]["properties"][2]["values"] = list(syn_counts) 
    
    info = json.dumps(info, cls=NumpyEncoder)

    with open("/home/hashir/j0126/j0126_assembled_core_relabeled.json", "w") as f:
        f.write(info)

    

    

    

    