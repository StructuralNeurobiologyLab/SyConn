# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from syconn import global_params
from syconn.handler.prediction import predict_dense_to_kd


def predict_myelin():
    """
    Generates a probability map for myelinated neuron voxels at
    ``global_params.config.working_dir + '/knossosdatasets/myelin'``.

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

    """
    predict_dense_to_kd(global_params.config.kd_seg_path,
                        global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_myelin, n_channel=2, mag=4,
                        target_channels=[(1, )], target_names=['myelin'])
