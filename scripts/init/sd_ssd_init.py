# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.extraction import cs_processing_steps as cps
from syconn.exec import exec_init

if __name__ == '__main__':
    exec_init.run_create_sds()

    # Create SSD and incorporate RAG
    #ssd = ss.SuperSegmentationDataset(version="new", ssd_type="ssv",
    #                                  sv_mapping='/mnt/j0126/areaxfs_v10/RAGs/v4b_20180214_nocb_merges_reconnected_knossos_mergelist.txt')
    #ssd.save_dataset_shallow()

    # About 2.5h, mostly due to a few very large ssvs, since there workers iterate sequentially over the individual svs per ssv
    #ssd.save_dataset_deep(qsub_pe="default", qsub_queue='all.q', n_max_co_processes=5000, stride=100)


    # Map objects to SSVs # TODO: This moved to scripts/multiview_neuron/create_ssd.py
    #from syconn.proc import ssd_proc

    #ssd = ss.SuperSegmentationDataset(working_dir=wd)

    # First step: Took 3.5h, but only on two workers # TODO: make dependent on global_params.existing_cell_organelles
    #ssd_proc.aggregate_segmentation_object_mappings(ssd, ['sj', 'vc', 'mi'], qsub_pe='default', qsub_queue='all.q')

    # Second step: 1h # TODO: make dependent on global_params.existing_cell_organelles
    #ssd_proc.apply_mapping_decisions(ssd, ['sj', 'vc', 'mi'], qsub_pe='default', qsub_queue='all.q')

    # Extract contact sites
    # About 2h
    # from syconn.extraction import cs_extraction_steps as ces
    # POPULATES CS CD
    #ces.find_contact_sites(cd, kd_seg_path, n_max_co_processes=5000,
    #                       qsub_pe='default', qsub_queue='all.q')

    # ces.extract_agg_contact_sites(cd, wd,
    #                               n_folders_fs=10000, suffix="",
    #                               n_max_co_processes=5000, qsub_pe='default', qsub_queue='all.q')


    #from syconn.extraction import cs_processing_steps as cps
    # cps.combine_and_split_cs_agg(wd, cs_gap_nm=300,
    #                              stride=100, qsub_pe='default', qsub_queue='all.q',
    #                              n_max_co_processes=200)

    ############################################################################################
    # ##### Cell object extraction #####
    # from_probabilities_to_objects(cd, filename, hdf5names)
    #
    #
    # ##### Object Processing #####
    # sj_sd = SegmentationDataset("sj", working_dir="path/to/wd")
    # sd_proc.dataset_analysis(sj_sd)
    #
    # # ??The segmentation needs to be written to a KnossosDataset before running this
    # sd_proc.map_objects_to_sv(sj_sd, obj_type, kd_path,
    #                           qsub_pe=my_qsub_pe, nb_cpus=1,
    #                           n_max_co_processes=100)
    #
    # ##### SSD Assembly #####
    # # ??create SSD and mergelist (knossos)
    # # mergelist can be supplied to ssd as parameter during initialization and will be applied automatically
    # ssd_proc.aggregate_segmentation_object_mappings(ssd, obj_types, qsub_pe=my_qsub_pe)
    # ssd_proc.apply_mapping_decisions(ssd, obj_types, qsub_pe=my_qsub_pe)
    #
    #
    # ##### CS Extraction #####
    # ces.find_contact_sites(cset, knossos_path, filename, n_max_co_processes=200,
    #                        qsub_pe=my_qsub_pe)
    # ces.extract_agg_contact_sites(cset, filename, hdf5name, working_dir,
    #                               n_folders_fs=10000, suffix="",
    #                               n_max_co_processes=200, qsub_pe=my_qsub_pe)
    #
    # ##### CS Processing #####
    # cps.combine_and_split_cs_agg(working_dir, cs_gap_nm=300,
    #                              stride=100, qsub_pemyqsub_pe,
    #                              n_max_co_processes=200)
    #
    #
    # ## CS Classification ##
    # sd_proc.export_sd_to_knossosdataset(cs_sd, cs_kd, block_edge_length=512,
    #                                     qsub_pe=my_qsub_pe, n_max_co_processes=100)
    # cps.overlap_mapping_sj_to_cs_via_kd(cs_sd, sj_sd, cs_kd, qsub_pe=my_qsub_pe, n_max_co_processes=100, n_folders_fs=10000)
    #
    #
    # ## Synaptic Classification ##
    #
