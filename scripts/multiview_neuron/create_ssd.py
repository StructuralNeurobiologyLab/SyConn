# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import networkx as nx
import numpy as np
import tqdm

from syconn.config import global_params
from syconn.handler.logger import log_main, initialize_logging
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.basics import parse_cc_dict_from_kzip
from syconn.proc import ssd_proc


if __name__ == '__main__':
    log = initialize_logging('create_neuron_ssd', global_params.wd + '/logs/',
                             overwrite=False)
    suffix = global_params.rag_suffix
    # TODO: the following paths currently require prior glia-splitting
    kml_p = "{}/glia/neuron_rag_ml{}.k.zip".format(global_params.wd, suffix)
    g_p = "{}/glia/neuron_rag{}.bz2".format(global_params.wd, suffix)
    cc_dict = parse_cc_dict_from_kzip(kml_p)
    cc_dict_inv = {}
    for ssv_id, cc in cc_dict.items():
        for sv_id in cc:
            cc_dict_inv[sv_id] = ssv_id
    rag_g = nx.read_edgelist(g_p, nodetype=np.uint)
    log.info('Parsed RAG from {} with {} SSVs and {} SVs.'.format(
        kml_p, len(cc_dict), len(cc_dict_inv)))
    ssd = SuperSegmentationDataset(working_dir=global_params.wd, version='new',
                                   ssd_type="ssv", sv_mapping=cc_dict_inv)
    # create cache-arrays for frequently used attributes
    ssd.save_dataset_shallow()
    ssd.save_dataset_deep(qsub_pe="openmp", n_max_co_processes=200)
    log.info('Finished SSD initialization. Starting cellular '
                  'organelle mapping.')

    # # map cellular organelles to SSVs
    # TODO: increase number of jobs in the next two QSUB submissions and sort by SSV size (descending)
    ssd_proc.aggregate_segmentation_object_mappings(
        ssd, global_params.existing_cell_organelles, qsub_pe="openmp")
    ssd_proc.apply_mapping_decisions(
        ssd, global_params.existing_cell_organelles, qsub_pe="openmp")
    log.info('Finished mapping of cellular organelles to SSVs. '
                  'Writing individual SSV graphs.')

    # Write SSV RAGs
    pbar = tqdm.tqdm(total=len(ssd.ssv_ids), mininterval=0.5)
    for ssv in ssd.ssvs:
        # get all nodes in CC of this SSV
        if len(cc_dict[ssv.id]) > 1:  # CCs with 1 node do not exist in the global RAG
            n_list = nx.node_connected_component(rag_g, ssv.id)
            # get SSV RAG as subgraph
            ssv_rag = nx.subgraph(rag_g, n_list)
        else:
            ssv_rag = nx.Graph()
            # ssv.id is the minimal SV ID, and therefore the only SV in this case
            ssv_rag.add_edge(ssv.id, ssv.id)
        nx.write_edgelist(ssv_rag, ssv.edgelist_path)
        pbar.update(1)
    pbar.close()
    log.info('Finished saving individual SSV RAGs.')
