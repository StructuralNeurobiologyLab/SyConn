from functools import total_ordering
import os
import json
import argparse
import numpy as np

from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.prediction import int2str_converter
from syconn.handler.logger import log_main as logger


info = {
    "@type": "neuroglancer_segment_properties",
    "inline": {
        "ids": None,
        "properties": []
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate info file for Neuroglancer precomputed segmentation properties')
    parser.add_argument('--wd', type=str, default=None, help='SyConn working directory of the dataset')

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

        logger.info("Retrieving cell types")
        for ct in sorted(unique_cts):
            labels.append(int2str_converter(ct, gt_type=gt_type))

        labels = np.array(labels, dtype=str)
        if "GPi" and "GPe" in labels:
            # remap GPe and GPi to GP
            cts[cts == 7] = 6
            cts[cts == 8] = 7
            cts[cts == 9] = 8
            cts[cts == 10] = 9
            labels = labels[labels != 'GPi']
            labels[labels == "GPe"] = "GP"

        if "j0126" in global_params.config.working_dir and "STN" in labels:
            labels[labels == "STN"] = "exc"

        logger.info(f"Found cell types: {labels}")
        sizes = ssd.load_numpy_data("size")

        # correct sizes overflow if needed
        neg_value_indices = np.where(sizes < 0)[0]
        if len(neg_value_indices) > 0:
            logger.info("Correcting sizes overflow")
            sd = SegmentationDataset('sv', working_dir=global_params.config.working_dir)
            for i in neg_value_indices:
                sv = sd.get_segmentation_object(ssd.ssv_ids[i])
                sv.load_attr_dict()
                sizes[i] = sv.attr_dict["size"]

        logger.info("Converting voxel sizes to volume (um)")
        vol = (sizes * np.prod(ssd.scaling)) / 1e9  # µm³

        ct_certainty = ssd.load_numpy_data("celltype_cnn_e3_certainty")
        ct_certainty = np.around(ct_certainty, decimals=4).astype(str)

        total_edge_length = ssd.load_numpy_data("total_edge_length")
        if not isinstance(total_edge_length, np.ndarray):
            logger.warning("Loading cached value from /home/shared")
            if "rag_flat_Jan2019_v3" in global_params.config.working_dir:
                total_edge_length = np.load('/home/shared/j0251/j0251_rag_flat_Jan2019_v3/total_edge_lengths.npy')
            elif "agglo2" in global_params.config.working_dir:
                total_edge_length = np.load('/home/shared/j0251/j0251_72_seg_20210127_agglo2/total_edge_lengths.npy')
            elif "areaxfs" in global_params.config.working_dir:
                total_edge_length = np.load('/home/shared/j0126/j0126_areaxfs_v10/total_edge_lengths.npy')
            elif "assembled_core_relabeled" in global_params.config.working_dir:
                total_edge_length = np.load('/home/shared/j0126/j0126_assembled_core_relabeled/total_edge_lengths.npy')
        
        total_edge_length /= 1000  # convert to um
        total_edge_length = total_edge_length.astype(np.int32)

        logger.info("Counting total number of synapse per cell")
        syn_ssvs = ssd.load_numpy_data("syn_ssv")
        func = np.vectorize(len)
        syn_counts = func(syn_ssvs)
        
        ssvs = ssd.ssv_ids

        if "j0251" in global_params.config.working_dir:
            mask = total_edge_length > 150
            cts = cts[mask]
            vol = vol[mask]
            syn_counts = syn_counts[mask]
            total_edge_length = total_edge_length[mask]
            ssvs = ssd.ssv_ids[mask]
            ct_certainty = ct_certainty[mask]

        info["inline"]["ids"] = list(map(str, ssvs))

        # Cell types
        info["inline"]["properties"].append(
            {
                "id": "Cell-types",
                "type": "tags",
                "tags": list(labels),
                "values": list(map(lambda e: [e], cts))
            }
        )

        # Cell volume
        info["inline"]["properties"].append(
            {
                "id": "Volume",
                "type": "number",
                "description": "Cell volume (µm³)",
                "data_type": "float32",
                "values": list(vol)
            }
        )

        # Total edge length
        info["inline"]["properties"].append(
            {
                "id": "TotalPathLength",
                "type": "number",
                "description": "Total path length of the cell (µm)",
                "data_type": "int32",
                "values": list(total_edge_length)
            }
        )

        # if "rag_flat_Jan2019_v3" in global_params.config.working_dir:
        #     mean_mesh_areas = np.load("/home/shared/j0251/j0251_rag_flat_Jan2019_v3/mean_mesh_areas.npy")
        #     mean_mesh_areas = np.around(mean_mesh_areas, decimals=4)
        #     info["inline"]["properties"].append(
        #         {
        #             "id": "MeanSynapseArea",
        #             "type": "number",
        #             "description": "Average synaptic area per cell (µm²)",
        #             "data_type": "float32",
        #             "values": list(mean_mesh_areas)
        #         }
        #     )

        # Number of synapses per cell
        info["inline"]["properties"].append(
            {
                "id": "NSynapses",
                "type": "number",
                "description": "Synapse count per cell",
                "data_type": "uint32",
                "values": list(syn_counts)
            }
        )

        # Cell type certainty
        # info["inline"]["properties"].append(
        #     {
        #         "id": "CellTypeCertainty",
        #         "type": "label",
        #         "description": "Certainty of cell type prediction",
        #         "values": list(ct_certainty)
        #     }
        # )
        
        info = json.dumps(info, cls=NumpyEncoder)

        with open("/home/hashir/j0251/rag_flat_Jan2019_v3.json", "w") as f:
            f.write(info)

    

    

    

    