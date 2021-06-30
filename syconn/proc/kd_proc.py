# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from typing import Optional

import numpy as np

from knossos_utils import KnossosDataset
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap


def convert_cube_size_kd(source_kd: str, target_kd_path: str, cube_size: np.ndarray,
                         do_raw: bool = False, nb_threads: int = 1, compresslevel: Optional[int] = None):
    """

    Args:
        source_kd:
        target_kd_path:
        cube_size:
        do_raw:
        nb_threads:
        compresslevel: Compression level used for storing segmentation data (not applied if `do_raw` is true).

    Returns:

    """
    kd = basics.kd_factory(source_kd)
    # init new KnossosDataset
    target_kd = KnossosDataset()
    target_kd._cube_shape = cube_size
    scale = np.array(kd.scale)
    target_kd.scales = [scale * mag for mag in sorted(kd.available_mags)]
    target_kd.initialize_without_conf(target_kd_path, kd.boundary, scale, kd.experiment_name,
                                      mags=list(kd.available_mags), create_pyk_conf=True,
                                      create_knossos_conf=False, server_format='knossos')
    _ = basics.kd_factory(target_kd_path)  # test if init is possible

    for mag in kd.available_mags:
        cs = np.array(cube_size) * mag
        grid = np.mgrid[0:kd.boundary[0]:cs[0], 0:kd.boundary[1]:cs[1], 0:kd.boundary[2]:cs[2]]
        chunk_coords = grid.reshape(3, -1).swapaxes(1, 0)
        njobs = max(nb_threads, int(np.ceil(len(chunk_coords) / 4)))
        multi_params = [(source_kd, target_kd_path, coords, do_raw, mag, cs, compresslevel) for coords in
                        basics.chunkify(chunk_coords, njobs)]
        start_multiprocess_imap(_convert_cube_size_kd_thread, multi_params, nb_cpus=nb_threads)


def _convert_cube_size_kd_thread(args):
    kd_source, kd_target, coords, do_raw, mag, cube_size, compresslevel = args
    kd_source = basics.kd_factory(kd_source)
    kd_target = basics.kd_factory(kd_target)

    for coord in coords:
        if do_raw:
            data = kd_source.load_raw(size=cube_size, offset=coord, mag=mag)
            kd_target.save_raw(offset=coord, mags=[mag], data=data, data_mag=mag)
        else:
            data = kd_source.load_seg(size=cube_size, offset=coord, mag=mag)
            kd_target.save_seg(offset=coord, mags=[mag], data=data, data_mag=mag, compresslevel=compresslevel)
