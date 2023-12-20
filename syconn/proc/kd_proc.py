# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from typing import Optional

import numpy as np
import tqdm

from knossos_utils import KnossosDataset
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap
from . import log_proc


def convert_cube_size_kd(source_kd: str, target_kd_path: str, cube_size: np.ndarray,
                         do_raw: bool = False, nb_threads: int = 1, compresslevel: Optional[int] = None):
    """
    Converts the cube size of a KnossosDataset (KD) and saves it to a new location. This function
    can handle both raw and segmentation data. The conversion is performed in parallel using
    multiprocessing.
    
    Args:
        source_kd (str): Path to the source KD.
        target_kd_path (str): Path where the converted KD will be saved.
        cube_size (np.ndarray): Desired cube size for the new KD.
        do_raw (bool, optional): If True, the function will handle raw data. If False, it will
            handle segmentation data. Defaults to False.
        nb_threads (int, optional): Number of threads to use for multiprocessing. Defaults to 1.
        compresslevel (Optional[int], optional): Compression level used for storing segmentation
            data. Not applied if `do_raw` is true. Defaults to None.
    
    Returns:
        None
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
        start_multiprocess_imap(_convert_cube_size_kd_thread, multi_params, nb_cpus=nb_threads, desc=f'mag={mag}')


def _convert_cube_size_kd_thread(args):
    """
    Helper function for convert_cube_size_kd. This function is designed to be used with
    multiprocessing. It loads a chunk of data from the source KD, converts it, and saves it to the
    target KD.
    
    Args:
        args (tuple): A tuple containing the following parameters:
            - kd_source (str): Path to the source KD.
            - kd_target (str): Path to the target KD.
            - coords (np.ndarray): Coordinates of the chunk to be processed.
            - do_raw (bool): If True, the function will handle raw data. If False, it will handle
                segmentation data.
            - mag (int): Magnification level of the data.
            - cube_size (np.ndarray): Desired cube size for the new KD.
            - compresslevel (Optional[int]): Compression level used for storing segmentation data.
                Not applied if `do_raw` is true.
    
    Returns:
        None
    """
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


def check_complete(kd1_p, kd2_p, mags, do_raw=False):
    """
    Checks if two KnossosDatasets (KDs) are identical. This function can handle both raw and
    segmentation data.
    
    Args:
        kd1_p (str): Path to the first KD.
        kd2_p (str): Path to the second KD.
        mags (list): List of magnification levels to check.
        do_raw (bool, optional): If True, the function will handle raw data. If False, it will
            handle segmentation data. Defaults to False.
    
    Returns:
        None
    
    Raises:
        ValueError: If the data in the two KDs is not identical.
    """
    kd1 = basics.kd_factory(kd1_p)
    kd2 = basics.kd_factory(kd2_p)

    for mag in mags:
        cs = np.array(kd2.cube_shape) * mag
        grid = np.mgrid[0:kd1.boundary[0]:cs[0], 0:kd1.boundary[1]:cs[1], 0:kd1.boundary[2]:cs[2]]
        chunk_coords = grid.reshape(3, -1).swapaxes(1, 0)
        for coord in tqdm.tqdm(chunk_coords, total=len(chunk_coords)):
            if do_raw:
                data1 = kd1.load_raw(size=kd2.cube_shape, offset=coord, mag=mag)
                data2 = kd1.load_raw(size=kd2.cube_shape, offset=coord, mag=mag)
            else:
                data1 = kd1.load_seg(size=kd2.cube_shape, offset=coord, mag=mag)
                data2 = kd1.load_seg(size=kd2.cube_shape, offset=coord, mag=mag)
            if not np.all(data1 == data2):
                raise ValueError(f'Data is not identical.')

