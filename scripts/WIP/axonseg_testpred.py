# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from syconn.handler.prediction import get_semseg_axon_model
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import semseg_of_sso_nocache
from syconn.handler import log_main
import os


if __name__ == '__main__':
    model_ident = 'axonseg_UNet-Dice-resizeconv_80nm'
    # from elektronn3.models.base import InferenceModel, THIS import currently does not work due to tbb availability issues
    m = get_semseg_axon_model()

    dest_dir = os.path.expanduser('~/testpred_axonsemseg_{}/'.format(model_ident))
    ssd = SuperSegmentationDataset(working_dir='/wholebrain/scratch/areaxfs3/', version='ctgt',
                                   sso_caching=True)
    os.makedirs(dest_dir, exist_ok=True)

    for sso in list(ssd.ssvs)[::-1]:
        if len(sso.mesh[1]) == 0:
            log_main.critical('{} has empty vertex array.'.format(sso))
        semseg_of_sso_nocache(sso, m, model_ident, (512, 256), 2, 51.2e3, verbose=True,
                              dest_path='{}/{}.k.zip'.format(dest_dir, sso.id))
