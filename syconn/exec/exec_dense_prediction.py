# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from typing import Optional, Tuple

import numpy as np

from syconn import global_params
from syconn.handler.prediction import predict_dense_to_kd


def predict_myelin(kd_raw_path: str = None,
                   cube_of_interest: Optional[Tuple[np.ndarray]] = None):
    """
    Generates a probability map for myelinated neuron voxels at
    ``global_params.config.working_dir + '/knossosdatasets/myelin/'`` stored
    in the raw channel, i.e. as uint8 (0..255).

    Examples:
        The entire myelin prediction for a single cell reconstruction including a smoothing
        is implemented as follows::

            from syconn import global_params
            from syconn.reps.super_segmentation import *
            from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

            # init. example data set
            global_params.wd = '~/SyConn/example_cube1/'

            # initialize example cell reconstruction
            ssd = SuperSegmentationDataset()
            ssv = list(ssd.ssvs)[0]
            ssv.load_skeleton()

            # get myelin predictions
            myelinated = map_myelin2coords(ssv.skeleton["nodes"], mag=4)
            ssv.skeleton["myelin"] = myelinated
            # this will generate a smoothed version at ``ssv.skeleton["myelin_avg10000"]``
            majorityvote_skeleton_property(ssv, "myelin")
            # store results as a KNOSSOS readable k.zip file
            ssv.save_skeleton_to_kzip(dest_path='~/{}_myelin.k.zip'.format(ssv.id),
                additional_keys=['myelin', 'myelin_avg10000'])

    Args:
        kd_raw_path: Path to the KnossosDataset of the raw data.
        cube_of_interest: Bounding box of the volume of interest (minimum and maximum
            coordinate in voxels in the respective magnification (see kwarg `mag`).

    """
    if kd_raw_path is None:
        kd_raw_path = global_params.config.kd_seg_path
    predict_dense_to_kd(kd_raw_path, global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_myelin, n_channel=2, mag=4, target_channels=[(1,)],
                        target_names=['myelin'], cube_of_interest=cube_of_interest)


def predict_synapsetype(cube_of_interest: Optional[Tuple[np.ndarray]] = None):
    """
    Generates synapse type predictions at every dataset voxel stored in
    ``global_params.config.working_dir + '/knossosdatasets/syntype_v2/'`` as
    overlay.

    Notes:
        Label 1: asymmetric, label 2: symmetric.

    Args:
        cube_of_interest: Bounding box of the volume of interest (minimum and maximum
            coordinate in voxels in the respective magnification (see kwarg `mag`).

    """
    predict_dense_to_kd(global_params.config.kd_seg_path,
                        global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_syntype,
                        mag=1, n_channel=4, target_names=['syntype_v2'],
                        target_channels=[(1, 2)],
                        cube_of_interest=cube_of_interest)


def predict_cellorganelles(cube_of_interest: Optional[Tuple[np.ndarray]] = None):
    """
    Generates synapse type predictions at every dataset voxel stored in
    ``global_params.config.working_dir + '/knossosdatasets/synapsetype/'`` as
    overlay.

    Notes:
        Labels:
            * 0: Background.
            * 1: Mitochondria.
            * 2: Vesicle clouds.
            * 3: Synaptic junction.

    Args:
        cube_of_interest: Bounding box of the volume of interest (minimum and maximum
            coordinate in voxels in the respective magnification (see kwarg `mag`).

    """
    predict_dense_to_kd(global_params.config.kd_seg_path,
                        global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_mivcsj,
                        mag=1, n_channel=4, target_names=['mivcsj'],
                        target_channels=[(1, 2, 3)],
                        cube_of_interest=cube_of_interest)


def predict_er(cube_of_interest: Optional[Tuple[np.ndarray]] = None):
    """
    Generates ER predictions at every dataset voxel stored in
    ``global_params.config.working_dir + '/knossosdatasets/er/'`` as
    overlay.

    Notes:
        Labels:
            * 0: Background.
            * 1: ER

    Args:
        cube_of_interest: Bounding box of the volume of interest (minimum and maximum
            coordinate in voxels in the respective magnification (see kwarg `mag`).

    """
    predict_dense_to_kd(global_params.config.kd_seg_path,
                        global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_er,
                        mag=1, n_channel=2, target_names=['er'],
                        target_channels=[(1,)],
                        cube_of_interest=cube_of_interest)


def predict_golgi(cube_of_interest: Optional[Tuple[np.ndarray]] = None):
    """
    Generates golgi predictions at every dataset voxel stored in
    ``global_params.config.working_dir + '/knossosdatasets/golgi/'`` as
    overlay.

    Notes:
        Labels:
            * 0: Background.
            * 1: Golgi Apparatus

    Args:
        cube_of_interest: Bounding box of the volume of interest (minimum and maximum
            coordinate in voxels in the respective magnification (see kwarg `mag`).

    """
    predict_dense_to_kd(global_params.config.kd_seg_path,
                        global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_golgi,
                        mag=1, n_channel=2, target_names=['golgi'],
                        target_channels=[(1,)],
                        cube_of_interest=cube_of_interest)