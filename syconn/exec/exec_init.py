# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)
from knossos_utils import chunky
from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.proc import sd_proc
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.logger import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.handler.basics import chunkify


# TODO: make it work with new SyConn
def run_create_sds(chunk_size=None, n_folders_fs=10000, generate_sv_meshs=False):
    """

    Parameters
    ----------
    chunk_size :
    n_folders_fs :
    generate_sv_meshs :

    Returns
    -------

    """
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    log = initialize_logging('create_sds', global_params.config.working_dir + '/logs/',
                             overwrite=False)

    # Sets initial values of object
    kd = knossosdataset.KnossosDataset()
    # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    kd.initialize_from_knossos_path(global_params.config.kd_seg_path)

    # TODO: get rid of explicit voxel extraction, all info necessary should be extracted at the beginning, e.g. size, bounding box etc and then refactor to only use those cached attributes!
    # resulting ChunkDataset, required for SV extraction --
    # Object extraction - 2h, the same has to be done for all cell organelles
    cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    log.info('Generating SegmentationDatasets for cell and cell organelle supervoxels.')
    oew.from_ids_to_objects(cd, "sv", overlaydataset_path=global_params.config.kd_seg_path,
                            n_chunk_jobs=5000, hdf5names=["sv"], n_max_co_processes=None,
                            qsub_pe='default', qsub_queue='all.q', qsub_slots=1,
                            n_folders_fs=n_folders_fs)

    # Object Processing -- Perform after mapping to also cache mapping ratios
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q',
                             compute_meshprops=True)

    # TODO: Add preprocessing of SV meshes only if config flag is set
    # preprocess sample locations (and meshes if they did not exist yet)
    log.debug("Caching sample locations (and meshes if not provided during init.).")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, 800)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type='sv')
    multi_params = [[par, so_kwargs] for par in multi_params]

    if generate_sv_meshs:
        _ = qu.QSUB_script(multi_params, "mesh_caching",
                           n_max_co_processes=global_params.NCORE_TOTAL,
                           pe="openmp", queue=None, script_folder=None, suffix="")
        sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q',
                                 compute_meshprops=True)
    _ = qu.QSUB_script(multi_params, "sample_location_caching",
                       n_max_co_processes=global_params.NCORE_TOTAL,
                       pe="openmp", queue=None, script_folder=None, suffix="")
    # recompute=False: only collect new sample_location property
    sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q',
                             compute_meshprops=True, recompute=False)
    log.info('Finished object extraction for cell SVs.')
    # create SegmentationDataset for each cell organelle
    for co in global_params.existing_cell_organelles:
        cd_dir = global_params.config.working_dir + "chunkdatasets/{}/".format(co)
        # Class that contains a dict of chunks (with coordinates) after initializing it
        cd = chunky.ChunkDataset()
        cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                      box_coords=[0, 0, 0], fit_box_size=True)
        prob_kd_path_dict = {co: getattr(global_params.config, 'kd_{}_path'.format(co))}
        # This creates a SegmentationDataset of type 'co'
        prob_thresh = global_params.config.entries["Probathresholds"][co]  # get probability threshold
        oew.from_probabilities_to_objects(cd, co, membrane_kd_path=global_params.config.kd_seg_path,
                                          prob_kd_path_dict=prob_kd_path_dict, thresholds=[prob_thresh],
                                          workfolder=global_params.config.working_dir,
                                          hdf5names=[co], n_max_co_processes=None, qsub_pe='default',
                                          qsub_queue='all.q', n_folders_fs=n_folders_fs, debug=False)
        sd_co = SegmentationDataset(obj_type=co, working_dir=global_params.config.working_dir)
        sd_proc.dataset_analysis(sd_co, qsub_pe="default", qsub_queue='all.q',
                                 compute_meshprops=True)
        # About 0.2 h per object class  # TODO: optimization required
        log.debug('Mapping objects {} to SVs.'.format(co))
        sd_proc.map_objects_to_sv(sd, co, global_params.config.kd_seg_path, qsub_pe='default',
                                  qsub_queue='all.q')
        log.info('Finished object extraction for {} SVs.'.format(co))

