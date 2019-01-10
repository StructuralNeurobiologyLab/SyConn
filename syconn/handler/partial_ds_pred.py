# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import time

from syconn.handler.prediction import _pred_dataset


if __name__ == '__main__':
    while True:
        # import error might occur when saving project...
        try:
            import sys
            import numpy as np
            break
        except ImportError:
            time.sleep(0.2)


    args = sys.argv
    kd_p = np.array(args[1])
    kd_pred_p = np.array(args[2])
    cd_p = np.array(args[3])
    model_p = np.array(args[4])
    ps = np.array(args[5])
    gpu_id = np.array(args[6], dtype=np.int)
    i = np.array(args[7], dtype=np.int)
    n = np.array(args[8], dtype=np.int)

    ps = tuple(np.array(str(ps).split('/'), dtype=np.int))

    _pred_dataset(str(kd_p), str(kd_pred_p), str(cd_p), str(model_p),
                  imposed_patch_size=ps, gpu_id=gpu_id, i=i, n=n)
