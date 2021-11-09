# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import networkx as nx
from knossos_utils import knossosdataset
import numpy as np
from syconn import global_params
from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.compression import save_to_h5py

if __name__ == '__main__':
    assert 'areaxfs_v6' in global_params.config.working_dir, 'Required dataset not available!'
    curr_dir = os.path.dirname(os.path.realpath(__file__)) + '/'

    raw_kd_path = global_params.config.kd_seg_path
    mi_kd_path = global_params.config.kd_mi_path
    vc_kd_path = global_params.config.kd_vc_path
    sj_kd_path = global_params.config.kd_sj_path
    sym_kd_path = global_params.config.kd_sym_path
    asym_kd_path = global_params.config.kd_asym_path

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(raw_kd_path)

    kd_mi = knossosdataset.KnossosDataset()
    kd_mi.initialize_from_knossos_path(mi_kd_path)

    kd_vc = knossosdataset.KnossosDataset()
    kd_vc.initialize_from_knossos_path(vc_kd_path)

    kd_sj = knossosdataset.KnossosDataset()
    kd_sj.initialize_from_knossos_path(sj_kd_path)

    kd_sym = knossosdataset.KnossosDataset()
    kd_sym.initialize_from_knossos_path(sym_kd_path)

    kd_asym = knossosdataset.KnossosDataset()
    kd_asym.initialize_from_knossos_path(asym_kd_path)

    # get data
    for example_cube_id in range(1, 4):
        kzip_p = '{}/example_cube{}.k.zip'.format(curr_dir, example_cube_id)
        data_dir = "{}/data{}/".format(curr_dir, example_cube_id)
        os.makedirs(data_dir, exist_ok=True)
        bb = parse_movement_area_from_zip(kzip_p)
        raw = kd.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        seg = kd.from_overlaycubes_to_matrix(size=bb[1] - bb[0], offset=bb[0], mag=1)
        seg_mi = kd_mi.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        seg_vc = kd_vc.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        seg_sj = kd_sj.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        sym = kd_sym.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        asym = kd_asym.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)

        # save data
        save_to_h5py([raw], data_dir + 'raw.h5', hdf5_names=['raw'])
        save_to_h5py([seg], data_dir + 'seg.h5', hdf5_names=['seg'])
        save_to_h5py([seg_mi], data_dir + 'mi.h5', hdf5_names=['mi'])
        save_to_h5py([seg_vc], data_dir + 'vc.h5', hdf5_names=['vc'])
        save_to_h5py([seg_sj], data_dir + 'sj.h5', hdf5_names=['sj'])
        save_to_h5py([sym], data_dir + 'sym.h5', hdf5_names=['sym'])
        save_to_h5py([asym], data_dir + 'asym.h5', hdf5_names=['asym'])

        # store subgraph of SV-agglomeration
        rag_g = nx.read_edgelist(global_params.config.neuron_svgraph_path, nodetype=np.uint64)
        sv_ids = np.unique(seg)
        rag_sub_g = rag_g.subgraph(sv_ids)
        print('Writing subgraph within {} and {} SVs.'.format(
            bb, rag_sub_g.number_of_nodes()))
        g_p = "{}/glia/astrocyte_svgraph.bz2".format(global_params.config.working_dir)
        rag_g = nx.read_edgelist(g_p, nodetype=np.uint64)
        rag_sub_g_glia = rag_g.subgraph(sv_ids)
        rag_sub_g = nx.union(rag_sub_g, rag_sub_g_glia)
        os.makedirs(data_dir, exist_ok=True)
        print('Writing subgraph within {} and {} SVs.'.format(
            bb, rag_sub_g.number_of_nodes()))
        nx.write_edgelist(rag_sub_g, data_dir + "/rag.bz2")




