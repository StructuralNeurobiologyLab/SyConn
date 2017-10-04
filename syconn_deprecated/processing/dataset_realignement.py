import cPickle as pkl
import glob
import itertools
from knossos_utils import knossosdataset
import numpy as np
import os

import dataset_realignement_helper as drh

from syconnmp import qsub_utils as qu
from syconnmp import shared_mem as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../multi_proc/")


def invert_coord_map(path_coord_map, stride=100, qsub_pe=None, qsub_queue=None,
                     nb_cpus=1):
        source_to_target_map = np.load(path_coord_map)[..., :-1]
        target_shape = np.ceil([2, np.max(source_to_target_map[0]/16),
                                np.max(source_to_target_map[1]/16),
                                source_to_target_map.shape[-1]]).astype(np.int)
        print target_shape
        multi_params = []
        zs = range(source_to_target_map.shape[-1])
        for z_block in [zs[i:i + stride] for i in xrange(0, len(zs), stride)]:
            multi_params.append([z_block, path_coord_map, target_shape])

        if qsub_pe is None and qsub_queue is None:
            results = sm.start_multiprocess(drh.invert_coord_map_thread,
                                            multi_params, nb_cpus=nb_cpus)

        elif qu.__QSUB__:
            path_to_out = qu.QSUB_script(multi_params,
                                         "invert_coord_map",
                                         n_cores=nb_cpus,
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=script_folder)
            # path_to_out = "/u/sdorkenw/QSUB/invert_coord_map/out/"
            out_files = glob.glob(path_to_out + "/*")
            results = []
            for out_file in out_files:
                with open(out_file) as f:
                    results.append(pkl.load(f))
        else:
            raise Exception("QSUB not available")

        target_to_source_map = np.zeros(target_shape)
        for result in results:
            print result[0]
            target_to_source_map[..., result[0]] = result[1]

        np.save(path_coord_map[:-4] + "_inv.npy", target_to_source_map)


def realign_from_target_w_kd(inv_coord_map_path, kd_from_path, kd_to_path,
                             size=(512, 512, 512), stride=20, qsub_pe=None,
                             qsub_queue=None, nb_cpus=8):
        kd_from = knossosdataset.KnossosDataset()
        kd_from.initialize_from_knossos_path(kd_from_path)

        if not os.path.exists(kd_to_path):
            kd_to = knossosdataset.KnossosDataset()
            kd_to.initialize_without_conf(kd_to_path,
                                          boundary=kd_from.boundary,
                                          scale=kd_from.scale,
                                          experiment_name=kd_from.experiment_name,
                                          mags=kd_from.mag)

        offsets = np.array(list(itertools.product(
            *[range(0, kd_from.boundary[i], size[i]) for i in range(3)])))

        multi_params = []
        for offset_block in [offsets[i:i + stride] for i in xrange(0, len(offsets), stride)]:
            multi_params.append([offset_block, size, inv_coord_map_path,
                                 kd_from_path, kd_to_path])

        if qsub_pe is None and qsub_queue is None:
            results = sm.start_multiprocess(drh.realign_from_target_w_kd_thread,
                                            multi_params, nb_cpus=nb_cpus)

        elif qu.__QSUB__:
            path_to_out = qu.QSUB_script(multi_params,
                                         "realign_from_target_w_kd",
                                         n_cores=nb_cpus,
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=script_folder)
