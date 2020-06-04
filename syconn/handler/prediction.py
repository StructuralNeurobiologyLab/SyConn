# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

# import here, otherwise it might fail if it is imported after importing torch
# see https://github.com/pytorch/pytorch/issues/19739

try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build
from ..handler.config import initialize_logging
from ..reps import log_reps
from ..proc.image import apply_morphological_operations
from ..mp import batchjob_utils as qu
from ..handler.basics import chunkify
from ..handler import log_handler, log_main, basics
from .compression import load_from_h5py, save_to_h5py
from .basics import read_txt_from_zip, get_filepaths_from_dir,\
    parse_cc_dict_from_kzip
from .. import global_params

import re
import os
import sys
import time
import tqdm
from logging import Logger
import shutil
import numpy as np
from typing import Iterable, Union, Optional, Any, Tuple, Callable, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from collections import Counter
from knossos_utils.knossosdataset import KnossosDataset
from scipy.stats import entropy
from scipy.special import softmax
from knossos_utils.chunky import ChunkDataset, save_dataset
from knossos_utils import knossosdataset
# for readthedocs build
try:
    import torch
except ImportError:
    pass


def load_gt_from_kzip(zip_fname, kd_p, raw_data_offset=75, verbose=False,
                      mag=1):
    """
    Loads ground truth from zip file, generated with Knossos. Corresponding
    dataset config file is located at kd_p.

    Args:
        zip_fname: str
        kd_p: str or List[str]
        raw_data_offset: int or np.array
            number of voxels used for additional raw offset, i.e. the offset for the
            raw data will be label_offset - raw_data_offset, while the raw data
            volume will be label_volume + 2*raw_data_offset. It will
            use 'kd.scaling' to account for dataset anisotropy if scalar or a
            list of length 3 hast to be provided for a custom x, y, z offset.
        verbose: bool
        mag: int
            Data mag. level.

    Returns: np.array, np.array
        raw data (float32) (multiplied with 1/255.), label data (uint16)

    """
    if type(kd_p) is str or type(kd_p) is bytes:
        kd_p = [kd_p]
    raw_data = []
    label_data = []
    for curr_p in kd_p:
        kd = basics.kd_factory(curr_p)
        bb = kd.get_movement_area(zip_fname)
        offset, size = bb[0], bb[1] - bb[0]
        scaling = np.array(kd.scale, dtype=np.int)
        if np.isscalar(raw_data_offset):
            raw_data_offset = np.array(scaling[0] * raw_data_offset / scaling,
                                       dtype=np.int)
            if verbose:
                print('Using scale adapted raw offset:', raw_data_offset)
        elif len(raw_data_offset) != 3:
            raise ValueError("Offset for raw cubes has to have length 3.")
        else:
            raw_data_offset = np.array(raw_data_offset)
        raw = kd.load_raw(size=(size // mag + 2 * raw_data_offset) * mag,
                          offset=(offset // mag - raw_data_offset) * mag,
                          mag=mag).swapaxes(0, 2)
        raw_data.append(raw[None, ])
        label = kd.load_kzip_seg(zip_fname, mag=mag).swapaxes(0, 2)
        label = label
        label_data.append(label[None, ])
    raw = np.concatenate(raw_data, axis=0).astype(np.float32)
    label = np.concatenate(label_data, axis=0)
    try:
        _ = parse_cc_dict_from_kzip(zip_fname)
    except:  # mergelist.txt does not exist
        label = np.zeros(size)
        return raw.astype(np.float32) / 255., label
    return raw.astype(np.float32) / 255., label


def predict_kzip(kzip_p, m_path, kd_path, clf_thresh=0.5, mfp_active=False,
                 dest_path=None, overwrite=False, gpu_ix=0,
                 imposed_patch_size=None):
    """
    Predicts data contained in k.zip file (defined by bounding box in knossos)

    Args:
        kzip_p: str
            path to kzip containing the raw data cube information
        m_path: str
            path to predictive model
        kd_path: str
            path to knossos dataset
        clf_thresh: float
            classification threshold
        mfp_active: False
        dest_path: str
            path to destination folder, if None folder of k.zip is used.
        overwrite: bool
        gpu_ix: int
        imposed_patch_size: tuple

    Returns:

    """
    cube_name = os.path.splitext(os.path.basename(kzip_p))[0]
    if dest_path is None:
        dest_path = os.path.dirname(kzip_p)
    from elektronn2.utils.gpu import initgpu
    if not os.path.isfile(dest_path + "/%s_data.h5" % cube_name) or overwrite:
        raw, labels = load_gt_from_kzip(kzip_p, kd_p=kd_path,
                                        raw_data_offset=0)
        raw = xyz2zxy(raw)
        initgpu(gpu_ix)
        from elektronn2.neuromancer.model import modelload
        m = modelload(m_path, imposed_patch_size=list(imposed_patch_size)
        if isinstance(imposed_patch_size, tuple) else imposed_patch_size,
                      override_mfp_to_active=mfp_active, imposed_batch_size=1)
        original_do_rates = m.dropout_rates
        m.dropout_rates = ([0.0, ] * len(original_do_rates))
        pred = m.predict_dense(raw[None, ], pad_raw=True)[1]
        # remove area without sufficient FOV
        pred = zxy2xyz(pred)
        raw = zxy2xyz(raw)
        save_to_h5py([pred, raw], dest_path + "/%s_data.h5" % cube_name,
                     ["pred", "raw"])
    else:
        pred, raw = load_from_h5py(dest_path + "/%s_data.h5" % cube_name,
                                   hdf5_names=["pred", "raw"])
    offset = parse_movement_area_from_zip(kzip_p)[0]
    overlaycubes2kzip(dest_path + "/%s_pred.k.zip" % cube_name,
                      (pred >= clf_thresh).astype(np.uint32),
                      offset, kd_path)


def predict_h5(h5_path, m_path, clf_thresh=None, mfp_active=False,
               gpu_ix=0, imposed_patch_size=None, hdf5_data_key=None,
               data_is_zxy=True, dest_p=None, dest_hdf5_data_key="pred",
               as_uint8=True):
    """
    Predicts data from h5 file. Assumes raw data is already float32.

    Args:
        h5_path: str
            path to h5 containing the raw data
        m_path: str
            path to predictive model
        clf_thresh: float
            classification threshold, if None, no thresholding
        mfp_active: False
        gpu_ix: int
        imposed_patch_size: tuple
        hdf5_data_key: str
            if None, it uses the first entry in the list returned by
            'load_from_h5py'
        data_is_zxy: bool
            if False, it will assumes data is [X, Y, Z]
        dest_p: str
        dest_hdf5_data_key: str
        as_uint8: bool

    Returns:

    """
    if hdf5_data_key:
        raw = load_from_h5py(h5_path, hdf5_names=[hdf5_data_key])[0]
    else:
        raw = load_from_h5py(h5_path, hdf5_names=None)
        assert len(raw) == 1, "'hdf5_data_key' not given but multiple hdf5 " \
                              "elements found. Please define raw data key."
        raw = raw[0]
    if not data_is_zxy:
        raw = xyz2zxy(raw)
    from elektronn2.utils.gpu import initgpu
    initgpu(gpu_ix)
    if raw.dtype.kind in ('u', 'i'):
        raw = raw.astype(np.float32) / 255.
    from elektronn2.neuromancer.model import modelload
    m = modelload(m_path, imposed_patch_size=list(imposed_patch_size)
    if isinstance(imposed_patch_size, tuple) else imposed_patch_size,
                  override_mfp_to_active=mfp_active, imposed_batch_size=1)
    original_do_rates = m.dropout_rates
    m.dropout_rates = ([0.0, ] * len(original_do_rates))
    pred = m.predict_dense(raw[None, ], pad_raw=True)[1]
    pred = zxy2xyz(pred)
    raw = zxy2xyz(raw)
    if as_uint8:
        pred = (pred * 255).astype(np.uint8)
        raw = (raw * 255).astype(np.uint8)
    if clf_thresh:
        pred = (pred >= clf_thresh).astype(np.float32)
    if dest_p is None:
        dest_p = h5_path[:-3] + "_pred.h5"
    if hdf5_data_key is None:
        hdf5_data_key = "raw"
    save_to_h5py([raw, pred], dest_p, [hdf5_data_key, dest_hdf5_data_key])


def overlaycubes2kzip(dest_p: str, vol: np.ndarray, offset: np.ndarray,
                      kd_path: str):
    """
    Writes segmentation volume to kzip.

    Args:
        dest_p: str
            path to k.zip
        vol: np.array
            Segmentation or prediction (unsigned integer, XYZ).
        offset: np.array
        kd_path: str

    Returns: np.array [Z, X, Y]

    """
    kd = basics.kd_factory(kd_path)
    kd.from_matrix_to_cubes(offset=offset, kzip_path=dest_p,
                            mags=[1], data=vol)


def xyz2zxy(vol: np.ndarray) -> np.ndarray:
    """
    Swaps axes to ELEKTRONN convention ([M, .., X, Y, Z] -> [M, .., Z, X, Y]).

    Args:
        vol: np.array [M, .., X, Y, Z]

    Returns: np.array [M, .., Z, X, Y]

    """
    # assert vol.ndim == 3  # removed for multi-channel support
    # adapt data to ELEKTRONN conventions (speed-up)
    vol = vol.swapaxes(-2, -3)  # y x z
    vol = vol.swapaxes(-3, -1)  # z x y
    return vol


def zxy2xyz(vol: np.ndarray) -> np.ndarray:
    """
    Swaps axes to ELEKTRONN convention ([M, .., Z, X, Y] -> [M, .., X, Y, Z]).

    Args:
        vol: np.array [M, .., Z, X, Y]

    Returns: np.array [M, .., X, Y, Z]

    """
    # assert vol.ndim == 3  # removed for multi-channel support
    vol = vol.swapaxes(-2, -3)  # x z y
    vol = vol.swapaxes(-2, -1)  # x y z
    return vol


def xyz2zyx(vol: np.ndarray) -> np.ndarray:
    """
    Swaps axes to ELEKTRONN convention ([M, .., X, Y, Z] -> [M, .., Z, X, Y]).

    Args:
        vol: np.array [M, .., X, Y, Z]

    Returns: np.array [M, .., Z, X, Y]

    """
    # assert vol.ndim == 3  # removed for multi-channel support
    # adapt data to ELEKTRONN conventions (speed-up)
    vol = vol.swapaxes(-1, -3)  # [..., z, y, x]
    return vol


def zyx2xyz(vol: np.ndarray) -> np.ndarray:
    """
    Swaps axes to ELEKTRONN convention ([M, .., Z, X, Y] -> [M, .., X, Y, Z]).

    Args:
        vol: np.array [M, .., Z, X, Y]

    Returns: np.array [M, .., X, Y, Z]

    """
    # assert vol.ndim == 3  # removed for multi-channel support
    vol = vol.swapaxes(-1, -3)  # [..., x, y, z]
    return vol


def create_h5_from_kzip(zip_fname: str, kd_p: str,
                        foreground_ids: Optional[Iterable[int]] = None,
                        overwrite: bool = True, raw_data_offset: int = 75,
                        debug: bool = False, mag: int = 1,
                        squeeze_data: int = True,
                        target_labels: Optional[Iterable[int]] = None,
                        apply_mops_seg: Optional[List[str]] = None):
    """
    Create .h5 files for ELEKTRONN (zyx) input. Only supports binary labels
    (0=background, 1=foreground).

    Examples:
        Suppose your k.zip file contains the segmentation GT with two segmentation
        IDs 1, 2 and is stored at ``kzip_fname``. The corresponding
        ``KnossosDataset`` is located at ``kd_path`` .
        The following code snippet will create an ``.h5`` file in the folder of
        ``kzip_fname`` with the raw data (additional offset controlled by
        ``raw_data_offset``) and the label data (either binary or defined by
        ``target_labels``) with the keys ``raw`` and ``label`` respectively::

            create_h5_from_kzip(d_p=kd_path, raw_data_offset=75,
            zip_fname=kzip_fname, mag=1, foreground_ids=[1, 2],
            target_labels=[1, 2])

    Args:
        zip_fname: Path to the annotated kzip file.
        kd_p: Path to the underlying raw data stored as KnossosDataset.
        foreground_ids: IDs which have to be converted to foreground, i.e. 1. Everything
            else is considered background (0). If None, everything except 0 is
            treated as foreground.
        overwrite: If True, will overwrite existing .h5 files
        raw_data_offset: Number of voxels used for additional raw offset, i.e. the offset for the
            raw data will be label_offset - raw_data_offset, while the raw data
            volume will be label_volume + 2*raw_data_offset. It will
            use 'kd.scaling' to account for dataset anisotropy if scalar or a
            list of length 3 hast to be provided for a custom x, y, z offset.
        debug: If True, file will have an additional 'debug' suffix and
            raw_data_offset is set to 0. Also their bit depths are adatped to be the
            same.
        mag: Data mag. level.
        squeeze_data: If True, label and raw data will be squeezed.
        target_labels: If set, `foreground_ids` must also be set. Each ID in
            `foreground_ids` will be mapped to the corresponding label in
            `target_labels`.
        apply_mops_seg: List of string identifiers for ndimage morphological operations.
    """
    if not squeeze_data and apply_mops_seg is not None:
        raise ValueError('Data might have axis with length one if squeeze_data=False.')
    if target_labels is not None and foreground_ids is None:
        raise ValueError('`target_labels` is set, but `foreground_ids` is None.')
    fname, ext = os.path.splitext(zip_fname)
    if fname[-2:] == ".k":
        fname = fname[:-2]
    if debug:
        file_appendix = '_debug'
        raw_data_offset = 0
    else:
        file_appendix = ''
    fname_dest = fname + file_appendix + ".h5"
    if os.path.isfile(fname_dest) and not overwrite:
        print("File at {} already exists. Skipping.".format(fname_dest))
        return
    raw, label = load_gt_from_kzip(zip_fname, kd_p, mag=mag,
                                   raw_data_offset=raw_data_offset)
    if squeeze_data:
        raw = raw.squeeze()
        label = label.squeeze()
    if foreground_ids is None:
        try:
            cc_dc = parse_cc_dict_from_kzip(zip_fname)
            foreground_ids = np.concatenate(list(cc_dc.values()))
        except:  # mergelist.txt does not exist
            foreground_ids = []
        print("Foreground IDs not assigned. Inferring from "
              "'mergelist.txt' in k.zip.:", foreground_ids)
    create_h5_gt_file(fname_dest, raw, label, foreground_ids, debug=debug,
                      target_labels=target_labels, apply_mops_seg=apply_mops_seg)


def create_h5_gt_file(fname: str, raw: np.ndarray, label: np.ndarray,
                      foreground_ids: Optional[Iterable[int]] = None,
                      target_labels: Optional[Iterable[int]] = None,
                      debug: bool = False,
                      apply_mops_seg: Optional[List[str]] = None):
    """
    Create .h5 files for ELEKTRONN input from two arrays.
    Only supports binary labels (0=background, 1=foreground). E.g. for creating
    true negative cubes set foreground_ids=[] to be an empty list. If set to
    None, everything except 0 is treated as foreground.

    Args:
        fname: str
            Path where h5 file should be saved
        raw: np.array
        label: np.array
        foreground_ids: iterable
            ids which have to be converted to foreground, i.e. 1. Everything
            else is considered background (0). If None, everything except 0 is
            treated as foreground.
        target_labels: Iterable
            If set, `foreground_ids` must also be set. Each ID in `foreground_ids` will
            be mapped to the corresponding label in `target_labels`.
        debug: bool
            will store labels and raw as uint8 ranging from 0 to 255
        apply_mops_seg: List of string identifiers for ndimage morphological operations.
    """
    if target_labels is not None and foreground_ids is None:
        raise ValueError('`target_labels` is set, but `foreground_ids` is None.')
    print(os.path.split(fname)[1])
    print("Label (before):", label.shape, label.dtype, label.min(), label.max())
    label = binarize_labels(label, foreground_ids, target_labels=target_labels)
    label = xyz2zxy(label)
    raw = xyz2zxy(raw)
    if apply_mops_seg is not None:
        label = apply_morphological_operations(label, morph_ops=apply_mops_seg)
    label = label.astype(np.uint16)
    print("Raw:", raw.shape, raw.dtype, raw.min(), raw.max())
    print("Label (after mapping):", label.shape, label.dtype, label.min(), label.max())
    print("-----------------\nGT Summary:\n%s\n" % str(Counter(label.flatten()).items()))
    if not fname[-2:] == "h5":
        fname = fname + ".h5"
    if debug:
        raw = (raw * 255).astype(np.uint8, copy=False)
        label = label.astype(np.uint8) * 255
    save_to_h5py([raw, label], fname, hdf5_names=["raw", "label"])


def binarize_labels(labels: np.ndarray, foreground_ids: Iterable[int],
                    target_labels: Optional[Iterable[int]] = None):
    """
    Transforms label array to binary label array (0=background, 1=foreground) or
    to the labels provided in `target_labels` by mapping the foreground IDs
    accordingly.

    Args:
        labels: np.array
        foreground_ids: iterable
        target_labels: Iterable
            labels used for mapping foreground IDs.

    Returns: np.array

    """
    new_labels = np.zeros_like(labels)
    if foreground_ids is None:
        target_labels = [1]
        if len(np.unique(labels)) > 2:
            print("------------ WARNING -------------\n"
                  "Found more than two different labels during label "
                  "conversion\n"
                  "----------------------------------")
        new_labels[labels != 0] = 1
    else:
        try:
            _ = iter(foreground_ids)
        except TypeError:
            foreground_ids = [foreground_ids]
        if target_labels is None:
            target_labels = [1 for _ in foreground_ids]
        for ii, ix in enumerate(foreground_ids):
            new_labels[labels == ix] = target_labels[ii]
    labels = new_labels
    assert len(np.unique(labels)) <= len(target_labels) + 1
    assert 0 <= np.max(labels) <= np.max(target_labels)
    assert 0 <= np.min(labels) <= np.max(target_labels)
    return labels.astype(np.uint16)


def parse_movement_area_from_zip(zip_fname: str) -> np.ndarray:
    """
    Parse MovementArea (e.g. bounding box of labeled volume) from annotation.xml
    in (k.)zip file.

    Args:
        zip_fname: str

    Returns: np.array
        Movement Area [2, 3]

    """
    anno_str = read_txt_from_zip(zip_fname, "annotation.xml").decode()
    line = re.findall("MovementArea (.*)/>", anno_str)
    assert len(line) == 1
    line = line[0]
    bb_min = np.array([re.findall(r'min.\w="(\d+)"', line)], dtype=np.uint)
    bb_max = np.array([re.findall(r'max.\w="(\d+)"', line)], dtype=np.uint)
    # Movement area is stored with 0-indexing! No adjustment needed
    return np.concatenate([bb_min, bb_max])


def pred_dataset(*args, **kwargs):
    log_handler.warning("'pred_dataset' will be replaced by 'predict_dense_to_kd' in"
                        " the near future.")
    return _pred_dataset(*args, **kwargs)


def _pred_dataset(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size=None,
                  mfp_active=False, gpu_id=0, overwrite=False, i=None, n=None):
    """
    Helper function for dataset prediction. Runs prediction on whole or partial
    knossos dataset. Imposed patch size has to be given in Z, X, Y!

    Args:
        kd_p: str
            path to knossos dataset .conf file
        kd_pred_p: str
            path to the knossos dataset head folder which will contain the prediction
        cd_p: str
            destination folder for chunk dataset containing prediction
        model_p:  str
            path tho ELEKTRONN2 model
        imposed_patch_size: tuple or None
            patch size (Z, X, Y) of the model
        mfp_active: bool
            activate max-fragment pooling (might be necessary to change patch_size)
        gpu_id: int
            the GPU used
        overwrite: bool
            True: fresh predictions ; False: earlier prediction continues
        i:
        n:

    Returns:

    """
    from elektronn2.utils.gpu import initgpu
    initgpu(gpu_id)
    from elektronn2.neuromancer.model import modelload
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_p, fixed_mag=1)

    m = modelload(model_p, imposed_patch_size=list(imposed_patch_size)
    if isinstance(imposed_patch_size, tuple) else imposed_patch_size,
                  override_mfp_to_active=mfp_active, imposed_batch_size=1)
    original_do_rates = m.dropout_rates
    m.dropout_rates = ([0.0, ] * len(original_do_rates))
    offset = m.target_node.shape.offsets
    offset = np.array([offset[1], offset[2], offset[0]], dtype=np.int)
    cd = ChunkDataset()
    cd.initialize(kd, kd.boundary, [512, 512, 256], cd_p, overlap=offset,
                  box_coords=np.zeros(3), fit_box_size=True)

    ch_dc = cd.chunk_dict
    print('Total number of chunks for GPU/GPUs:', len(ch_dc.keys()))

    if i is not None and n is not None:
        chunks = ch_dc.values()[i::n]
    else:
        chunks = ch_dc.values()
    print("Starting prediction of %d chunks in gpu %d\n" % (len(chunks), gpu_id))

    if not overwrite:
        for chunk in chunks:
            try:
                _ = chunk.load_chunk("pred")[0]
            except Exception as e:
                chunk_pred(chunk, m)
    else:
        for chunk in chunks:
            try:
                chunk_pred(chunk, m)
            except KeyboardInterrupt as e:
                print("Exiting out from chunk prediction: ", str(e))
                return
    save_dataset(cd)

    # single gpu processing also exports the cset to kd
    if n is None:
        kd_pred = KnossosDataset()
        kd_pred.initialize_without_conf(kd_pred_p, kd.boundary, kd.scale,
                                        kd.experiment_name, mags=[1, 2, 4, 8])
        cd.export_cset_to_kd(kd_pred, "pred", ["pred"], [4, 4], as_raw=True,
                             stride=[256, 256, 256])


def predict_dense_to_kd(kd_path: str, target_path: str, model_path: str,
                        n_channel: int, target_names: Optional[Iterable[str]] = None,
                        target_channels: Optional[Iterable[Iterable[int]]] = None,
                        channel_thresholds: Optional[Iterable[Union[float, Any]]] = None,
                        log: Optional[Logger] = None, mag: int = 1,
                        overlap_shape_tiles: Tuple[int, int, int] = (40, 40, 20),
                        cube_of_interest: Optional[Tuple[np.ndarray]] = None,
                        overwrite: bool = False):
    """
    Helper function for dense dataset prediction. Runs predictions on the whole
    knossos dataset located at `kd_path`.
    Prediction results will be written to KnossosDatasets called `target_names`
    at `target_path`. If no threshold and only one channel per `target_names`
    is given, the resulting KnossosDataset will contain a probability map in the
    raw channel as uint8 (0..255).
    Otherwise the classification results will be written to the overlay channel.

    Notes:
        *  TODO: Currently has a high GPU memory requirement (minimum 12GB).

    Args:
        kd_path: Path to KnossosDataset .conf file of the raw data.
        target_path: Destination directory for the output KnossosDataset(s)
            which contain the prediction(s).
        model_path: Path to elektronn3 model for predictions. Loaded via the
            :class:`~elektronn3.inference.inference.Predictor`.
        n_channel: Number of channels predicted by the model.
        target_names: Names of target knossos datasets, e.g.
            ``target_names=['synapse_fb', 'synapse_type']``. Defaults to
            ``['pred']``. Length must match with `target_channels`.
        target_channels: Channel_ids in prediction for each target knossos data set
            e.g. ``target_channels=[(1, 2)]`` if prediction has two foreground labels.
            Defaults to ``[[ix for ix in range(n_channel)]]``.
            Length must match with `target_names`.
        channel_thresholds: Thresholds for channels: If None and number of channels
            for target kd is 1: probabilities are stored. Else: 0.5 as default
            e.g. ``channel_thresholds=[None,0.5,0.5]``.
        log: Logger.
        mag: Data magnification level.
        overlap_shape_tiles: [WIP] Overlap in voxels [XYZ] used for each
            tile predicted during inference.
            Currently the following chunk/tile properties are used additionally
            (`overlap_shape` is the per-chunk overlap)::

                chunk_size = np.array([1024, 1024, 256], dtype=np.int)  # XYZ
                n_tiles = np.array([4, 4, 16])
                tile_shape = (chunk_size / n_tiles).astype(np.int)
                # the final input shape must be a multiple of tile_shape
                overlap_shape = tile_shape // 2

        cube_of_interest: Bounding box of the volume of interest (minimum and maximum
            coordinate in voxels in the respective magnification (see kwarg `mag`).
        overwrite: Overwrite existing KDs.

    """
    # TODO: switch to pyk confs
    if log is None:
        log_name = 'dense_prediction'
        if target_names is not None:
            log_name += '_' + "".join(target_names)
        log = initialize_logging(log_name, global_params.config.working_dir + '/logs/',
                                 overwrite=False)
    if target_names is None:
        target_names = ['pred']
    if target_channels is None:
        target_channels = [[ix for ix in range(n_channel)]]
    if not len(target_names) == len(target_channels):
        msg = 'For every target name the target channels have to be specified.'
        log_reps.error(msg)
        raise ValueError(msg)
    if channel_thresholds is None:
        channel_thresholds = [None for _ in range(n_channel)]

    kd = basics.kd_factory(kd_path)
    if cube_of_interest is None:
        cube_of_interest = (np.zeros(3, ), kd.boundary // mag)

    # TODO: these should be config parameters
    overlap_shape_tiles = np.array([30, 31, 20])
    overlap_shape = overlap_shape_tiles
    if qu.batchjob_enabled():
        chunk_size = np.array([1024, 1024, 512])
    else:  # assume small dataset volume
        chunk_size = np.array([482, 481, 236])
    tile_shape = [271, 181, 138]

    cd = ChunkDataset()
    cd.initialize(kd, cube_of_interest[1], chunk_size, target_path + '/cd_tmp/',
                  box_coords=cube_of_interest[0], list_of_coords=[],
                  fit_box_size=True, overlap=overlap_shape)
    chunk_ids = list(cd.chunk_dict.keys())
    # init target KnossosDatasets
    target_kd_path_list = [target_path+'/{}/'.format(tn) for tn in target_names]
    for path in target_kd_path_list:
        if os.path.isdir(path):
            if not overwrite:
                msg = f'Found existing KD at "{path}" but overwrite is set to False.'
                log.error(msg)
                raise ValueError(msg)
            log.debug('Found existing KD at {}. Removing it now.'.format(path))
            shutil.rmtree(path)
    for path in target_kd_path_list:
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_without_conf(path, kd.boundary, kd.scale,
                                          kd.experiment_name, [2**x for x in range(6)],)
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_from_knossos_path(path)
    # init batchjob parameters
    multi_params = chunk_ids
    multi_params = chunkify(multi_params, global_params.config.ngpu_total)
    multi_params = [(ch_ids, kd_path, target_path, model_path, overlap_shape,
                     overlap_shape_tiles, tile_shape, chunk_size, n_channel, target_channels,
                     target_kd_path_list, channel_thresholds, mag, cube_of_interest)
                    for ch_ids in multi_params]
    log.info('Started dense prediction of {} in {:d} chunk(s).'.format(
        ", ".join(target_names), len(chunk_ids)))
    n_cores_per_job = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'] if\
        qu.batchjob_enabled() else global_params.config['ncores_per_node']

    qu.batchjob_script(multi_params, "predict_dense",
                       n_cores=n_cores_per_job, remove_jobfolder=True, log=log,
                       additional_flags="--gres=gpu:1")
    log.info('Finished dense prediction of {}'.format(", ".join(target_names)))


def dense_predictor(args):
    """
    Volumes are transformed by XYZ <-> ZYX before they are passed to the
    model.

    Args:
        args: Tuple(
            chunk_ids: list
                list of chunks in chunk dataset
            kd_p : str
                path to knossos dataset .conf file
            cd_p : str
                destination folder for chunk dataset containing prediction
            model_p : str
                path to model
            offset :
            chunk_size:
            ...
            )

    Returns:

    """
    # TODO: remove chunk necessity
    # TODO: clean up (e.g. redundant chunk sizes, ...)
    #
    chunk_ids, kd_p, target_p, model_p, overlap_shape, overlap_shape_tiles,\
    tile_shape, chunk_size, n_channel, target_channels, target_kd_path_list, \
    channel_thresholds, mag, cube_of_interest = args

    # init KnossosDataset:
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_p)

    # init ChunkDataset:
    cd = ChunkDataset()
    cd.initialize(kd, cube_of_interest[1], chunk_size, target_p + '/cd_tmp/',
                  box_coords=cube_of_interest[0], list_of_coords=[],
                  fit_box_size=True, overlap=overlap_shape)

    # init Target KnossosDataset
    target_kd_dict = {}
    for path in target_kd_path_list:
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_from_knossos_path(path)
        target_kd_dict[path] = target_kd

    # init Predictor
    from elektronn3.inference import Predictor
    ix = 0
    tile_shape = np.array(tile_shape)
    while True:
        try:
            out_shape = (chunk_size + 2 * np.array(overlap_shape)).astype(np.int)[::-1]  # ZYX
            out_shape = np.insert(out_shape, 0, n_channel)  # output must equal chunk size
            predictor = Predictor(model_p, strict_shapes=True, tile_shape=tile_shape[::-1],
                                  out_shape=out_shape, overlap_shape=overlap_shape_tiles[::-1],
                                  apply_softmax=True)
            predictor.model.ae = False
            _ = predictor.predict(np.zeros(out_shape[1:])[None, None])
            break
        except RuntimeError:  # cuda MemoryError
            if np.all(tile_shape % 2):
                raise ValueError('Cannot reduce tile shape anymore. Please adapt '
                                 'the tile/overlap/chunk shape in the function '
                                 'that is calling `dense_predictor`.')
            while tile_shape[ix] % 2:
                ix += 1
            tile_sh_orig = np.array(tile_shape)
            tile_shape[ix] = tile_shape[ix] // 2
            log_main.warn(f'Changed tile shape from {tile_sh_orig} to '
                          f'{tile_shape} to reduce memory requirements.')
            ix = (ix + 1) % 3  # permute spatial dimension which is reduced

    # predict Chunks
    for ch_id in chunk_ids:
        ch = cd.chunk_dict[ch_id]
        ol = ch.overlap

        size = np.array(np.array(ch.size) + 2 * np.array(ol),
                        dtype=np.int)

        coords = np.array(np.array(ch.coordinates) - np.array(ol),
                          dtype=np.int)
        raw = kd.load_raw(size=size*mag, offset=coords*mag, mag=mag)

        pred = dense_predicton_helper(raw.astype(np.float32) / 255., predictor,
                                      is_zyx=True, return_zyx=True)

        # slice out the original input volume along ZYX, i.e. the last three axes
        pred = pred[..., ol[2]:-ol[2], ol[1]:-ol[1], ol[0]:-ol[0]]
        for j in range(len(target_channels)):
            ids = target_channels[j]
            path = target_kd_path_list[j]
            data = np.zeros_like(pred[0]).astype(np.uint64)
            save_as_raw = not (len(ids) > 1)
            for label in ids:
                t = channel_thresholds[label]
                # if threshold is given or multiple target labels per dataset
                # store classification results
                # TODO: argmax might be more reasonable
                if not save_as_raw:
                    if t is None:
                        t = 255 / 2
                    if t < 1.:
                        t = 255 * t
                    pred_mask = pred[label] > t
                    data[pred_mask] = label
                else:
                    # no thresholding and only one label in the target KnossosDataset
                    # -> store probability map.
                    data = pred[label]
            if save_as_raw:
                target_kd_dict[path].save_raw(
                    offset=ch.coordinates*mag, data=data.astype(np.uint8),
                    data_mag=mag, mags=[mag, mag*2, mag*4],
                    fast_resampling=True, upsample=False)
            else:
                target_kd_dict[path].save_seg(
                    offset=ch.coordinates * mag, data=data, data_mag=mag,
                    mags=[mag, mag * 2, mag * 4],
                    fast_resampling=True, upsample=False)


def dense_predicton_helper(raw: np.ndarray, predictor: 'Predictor', is_zyx=False,
                           return_zyx=False) -> np.ndarray:
    """

    Args:
        raw: The input data array in CXYZ.
        predictor: The model which performs the inference. Requires ``predictor.predict``.
        is_zyx:
        return_zyx:

    Returns:
        The inference result in CXYZ as uint8 between 0..255.
    """
    # transform raw data
    if not is_zyx:
        raw = xyz2zyx(raw)
    # predict: pred of the form (N, C, [D,], H, W)
    pred = predictor.predict(raw[None, None]).numpy()
    pred = np.array(pred[0]) * 255  # remove N-axis
    pred = pred.astype(np.uint8)
    if not return_zyx:
        pred = zyx2xyz(pred)
    return pred


def to_knossos_dataset(kd_p, kd_pred_p, cd_p, model_p,
                       imposed_patch_size, mfp_active=False):
    """

    Args:
        kd_p:
        kd_pred_p:
        cd_p:
        model_p:
        imposed_patch_size:
        mfp_active:

    Returns:

    """
    from elektronn2.neuromancer.model import modelload
    log_reps.warning('Depracation Warning; "to_knossos_dataset" is deprecated and will be '
                     'replaced by "predict_dense_to_kd" which immediately .')
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_p, fixed_mag=1)
    kd_pred = KnossosDataset()
    m = modelload(model_p, imposed_patch_size=list(imposed_patch_size)
    if isinstance(imposed_patch_size, tuple) else imposed_patch_size,
                  override_mfp_to_active=mfp_active, imposed_batch_size=1)
    original_do_rates = m.dropout_rates
    m.dropout_rates = ([0.0, ] * len(original_do_rates))
    offset = m.target_node.shape.offsets
    offset = np.array([offset[1], offset[2], offset[0]], dtype=np.int)
    cd = ChunkDataset()
    cd.initialize(kd, kd.boundary, [512, 512, 256], cd_p, overlap=offset,
                  box_coords=np.zeros(3), fit_box_size=True)
    kd_pred.initialize_without_conf(kd_pred_p, kd.boundary, kd.scale,
                                    kd.experiment_name, mags=[1,2,4,8])
    cd.export_cset_to_kd(kd_pred, "pred", ["pred"], [4, 4], as_raw=True,
                         stride=[256, 256, 256])


def prediction_helper(raw, model, override_mfp=True,
                      imposed_patch_size=None):
    """
    Helper function for predicting raw volumes (range: 0 to 255; uint8).
    Will change X, Y, Z to ELEKTRONN format (Z, X, Y) and returns prediction
    in standard format [X, Y, Z]. Imposed patch size has to be given in Z, X, Y!

    Args:
        raw: np.array
            volume [X, Y, Z]
        model: str or model object
            path to model (.mdl)
        override_mfp: bool
        imposed_patch_size: tuple
            in Z, X, Y FORMAT!

    Returns: np.array
        prediction data [X, Y, Z]

    """
    if type(model) == str:
        from elektronn2.neuromancer.model import modelload
        m = modelload(model, imposed_patch_size=list(imposed_patch_size)
        if isinstance(imposed_patch_size, tuple) else imposed_patch_size,
                      override_mfp_to_active=override_mfp, imposed_batch_size=1)
        original_do_rates = m.dropout_rates
        m.dropout_rates = ([0.0, ] * len(original_do_rates))
    else:
        m = model
    raw = xyz2zxy(raw)
    if raw.dtype.kind in ('u', 'i'):
        # convert to float 32 and scale it
        raw = raw.astype(np.float32) / 255.
    if not raw.dtype == np.float32:
        # assume already normalized between 0 and 1
        raw = raw.astype(np.float32)
    assert 0 <= np.max(raw) <= 1.0 and 0 <= np.min(raw) <= 1.0
    pred = m.predict_dense(raw[None,], pad_raw=True)[1]
    return zxy2xyz(pred)


def chunk_pred(ch: 'chunky.Chunk', model: 'torch.nn.Module', debug: bool = False):
    """
    Helper function to write chunks.

    Args:
        ch: Chunk
        model: str or model object
        debug: bool

    Returns:

    """
    raw = ch.raw_data()
    pred = prediction_helper(raw, model) * 255
    pred = pred.astype(np.uint8)
    ch.save_chunk(pred, "pred", "pred", overwrite=True)
    if debug:
        ch.save_chunk(raw, "pred", "raw", overwrite=False)


def get_glia_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new` """
    from elektronn3.models.base import InferenceModel
    # m = torch.jit.load(global_params.config.mpath_glia_e3)
    # m = InferenceModel(m, normalize_func=naive_view_normalization_new)
    m = InferenceModel(global_params.config.mpath_glia_e3, normalize_func=naive_view_normalization_new)
    return m


def get_celltype_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new`
     Unlike the other e3 InferenceModel instances, here the view normalization
     is applied in the downstream inference method (`predict_sso_celltype`)
      because the celltype model also gets scalar values as input which should
      not be normalized."""
    try:
        from elektronn3.models.base import InferenceModel
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    m = torch.jit.load(global_params.config.mpath_celltype_e3)
    m = InferenceModel(m, bs=40, multi_gpu=True)
    # m = InferenceModel(global_params.config.mpath_celltype_e3, bs=40)
    return m


def get_semseg_spiness_model_pts():
    from elektronn3.models.base import InferenceModel
    m = torch.jit.load(global_params.config.mpath_glia_e3)
    m = InferenceModel(m)
    return m


def get_semseg_spiness_model():
    try:
        from elektronn3.models.base import InferenceModel
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    path = global_params.config.mpath_spiness
    m = torch.jit.load(path)
    m = InferenceModel(m)
    # m = InferenceModel(path)
    m._path = path
    return m


def get_semseg_axon_model():
    try:
        from elektronn3.models.base import InferenceModel
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    path = global_params.config.mpath_axonsem
    m = torch.jit.load(path)
    m = InferenceModel(m)
    # m = InferenceModel(path)
    m._path = path
    return m


def get_tripletnet_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new` """
    try:
        from elektronn3.models.base import InferenceModel
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    m = torch.jit.load(global_params.config.mpath_tnet)
    m = InferenceModel(m)
    # m = InferenceModel(global_params.config.mpath_tnet)
    return m


def get_myelin_cnn():
    """
    elektronn3 model trained to predict binary myelin-in class.

    Returns:
        The trained Inference model.
    """
    try:
        from elektronn3.inference.inference import Predictor
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    m = torch.jit.load(global_params.config.mpath_myelin)
    m = Predictor(m)
    # m = Predictor(global_params.config.mpath_myelin)
    return m


def get_knn_tnet_embedding_e3():
    tnet_eval_dir = "{}/pred/".format(global_params.config.mpath_tnet)
    return knn_clf_tnet_embedding(tnet_eval_dir)


def get_pca_tnet_embedding_e3():
    tnet_eval_dir = "{}/pred/".format(global_params.config.mpath_tnet)
    return pca_tnet_embedding(tnet_eval_dir)


def naive_view_normalization(d):
    # TODO: Remove with new dataset, only necessary for backwards compat.
    d = d.astype(np.float32)
    # perform pseudo-normalization
    # (proper normalization: how to store mean and std for inference?)
    if not (np.min(d) >= 0 and np.max(d) <= 1.0):
        for ii in range(len(d)):
            curr_view = d[ii]
            if 0 <= np.max(curr_view) <= 1.0:
                curr_view = curr_view - 0.5
            else:
                curr_view = curr_view / 255. - 0.5
            d[ii] = curr_view
    else:
        d = d - 0.5
    return d


def naive_view_normalization_new(d):
    return d.astype(np.float32) / 255. - 0.5


def knn_clf_tnet_embedding(fold, fit_all=False):
    """
    Currently it assumes embedding for GT views has been created already in 'fold'
    and put into l_train_%d.npy / l_valid_%d.npy files.

    Args:
        fold: str
        fit_all: bool

    Returns:

    """
    train_fnames = get_filepaths_from_dir(
        fold, fname_includes=["l_axoness_train"], ending=".npy")
    valid_fnames = get_filepaths_from_dir(
        fold, fname_includes=["l_axoness_valid"], ending=".npy")

    train_d = []
    train_l = []
    valid_d = []
    valid_l = []
    for tf in train_fnames:
        train_l.append(np.load(tf))
        tf = tf.replace("l_axoness_train", "ls_axoness_train")
        train_d.append(np.load(tf))
    for tf in valid_fnames:
        valid_l.append(np.load(tf))
        tf = tf.replace("l_axoness_valid", "ls_axoness_valid")
        valid_d.append(np.load(tf))

    train_d = np.concatenate(train_d).astype(dtype=np.float32)
    train_l = np.concatenate(train_l).astype(dtype=np.uint16)
    valid_d = np.concatenate(valid_d).astype(dtype=np.float32)
    valid_l = np.concatenate(valid_l).astype(dtype=np.uint16)

    nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='auto',
                                n_jobs=16, weights='uniform')
    if fit_all:
        nbrs.fit(np.concatenate([train_d, valid_d]),
                 np.concatenate([train_l, valid_l]).ravel())
    else:
        nbrs.fit(train_d, train_l.ravel())
    return nbrs


def pca_tnet_embedding(fold, n_components=3, fit_all=False):
    """
    Currently it assumes embedding for GT views has been created already in 'fold'
    and put into l_train_%d.npy / l_valid_%d.npy files.

    Args:
        fold: str
        n_components: int
        fit_all: bool

    Returns:

    """
    train_fnames = get_filepaths_from_dir(
        fold, fname_includes=["l_axoness_train"], ending=".npy")
    valid_fnames = get_filepaths_from_dir(
        fold, fname_includes=["l_axoness_valid"], ending=".npy")

    train_d = []
    train_l = []
    valid_d = []
    valid_l = []
    for tf in train_fnames:
        train_l.append(np.load(tf))
        tf = tf.replace("l_axoness_train", "ls_axoness_train")
        train_d.append(np.load(tf))
    for tf in valid_fnames:
        valid_l.append(np.load(tf))
        tf = tf.replace("l_axoness_valid", "ls_axoness_valid")
        valid_d.append(np.load(tf))

    train_d = np.concatenate(train_d).astype(dtype=np.float32)
    train_l = np.concatenate(train_l).astype(dtype=np.uint16)
    valid_d = np.concatenate(valid_d).astype(dtype=np.float32)
    valid_l = np.concatenate(valid_l).astype(dtype=np.uint16)

    pca = PCA(n_components, whiten=True, random_state=0)
    if fit_all:
        pca.fit(np.concatenate([train_d, valid_d]))
    else:
        pca.fit(train_d)
    return pca


def views2tripletinput(views):
    views = views[:, :, :1]  # use first view only
    out_d = np.concatenate([views,
                            np.ones_like(views),
                            np.ones_like(views)], axis=2)
    return out_d.astype(np.float32)


def certainty_estimate(inp: np.ndarray, is_logit: bool = False) -> float:
    """
    Estimates the certainty of (independent) predictions of the same sample:
        1. If `is_logit` is True, Generate pseudo-probabilities from the
           input using softmax.
        2. Sum the evidence per class and (re-)normalize.
        3. Compute the entropy, scale it with the maximum entropy (equal
           probabilities) and subtract it from 1.

    Args:
        inp: 2D array of prediction results (N: number of samples,
            C: Number of classes)
        is_logit: If True, applies ``softmax(inp, axis=1)``.

    Returns:
        Certainty measure based on the entropy of a set of (independent)
        predictions.
    """
    if not inp.ndim == 2:
        raise ValueError('Input is not two dimensional.')
    if is_logit:
        proba = softmax(inp, axis=1)
    else:
        proba = inp
    # sum probabilities across samples
    proba = np.sum(proba, axis=0)
    # normalize
    proba = proba / np.sum(proba)
    # maximum entropy at equal probabilities: -sum(1/N*ln(1/N) = ln(N)
    entr_max = np.log(len(proba))
    entr_norm = entropy(proba) / entr_max
    # convert to certainty estimate
    return 1 - entr_norm


def str2int_converter(comment: str, gt_type: str) -> int:
    if gt_type == "axgt":
        if comment == "gt_axon":
            return 1
        elif comment == "gt_dendrite":
            return 0
        elif comment == "gt_soma":
            return 2
        elif comment == "gt_bouton":
            return 3
        elif comment == "gt_terminal":
            return 4
        else:
            return -1
    elif gt_type == "spgt":
        if "head" in comment:
            return 1
        elif "neck" in comment:
            return 0
        elif "shaft" in comment:
            return 2
        elif "other" in comment:
            return 3
        else:
            return -1
    elif gt_type == 'ctgt_j0251':
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
                             INT=8, FS=9, LTS=10)
        return str2int_label[comment]
    else:
        raise ValueError("Given groundtruth type is not valid.")


# create function that converts information in string type to the
# information in integer type

def int2str_converter(label: int, gt_type: str) -> str:
    """
    TODO: remove redundant definitions.
    Converts integer label into semantic string.

    Args:
        label: int
        gt_type: str
            e.g. spgt for spines, axgt for cell compartments or ctgt for cell type

    Returns: str

    """
    if type(label) == str:
        label = int(label)
    if gt_type == "axgt":
        if label == 1:
            return "gt_axon"
        elif label == 0:
            return "gt_dendrite"
        elif label == 2:
            return "gt_soma"
        elif label == 3:
            return "gt_bouton"
        elif label == 4:
            return "gt_terminal"
        else:
            return -1  # TODO: Check if somewhere -1 is handled, otherwise return "N/A"
    elif gt_type == "spgt":
        if label == 1:
            return "head"
        elif label == 0:
            return "neck"
        elif label == 2:
            return "shaft"
        elif label == 3:
            return "other"
        else:
            return -1  # TODO: Check if somewhere -1 is already used, otherwise return "N/A"
    elif gt_type == 'ctgt':
        if label == 1:
            return "MSN"
        elif label == 0:
            return "EA"
        elif label == 2:
            return "GP"
        elif label == 3:
            return "INT"
        else:
            return -1  # TODO: Check if somewhere -1 is already used, otherwise return "N/A"
    elif gt_type == 'ctgt_v2':
        # DA and TAN are type modulatory, if this is changes, also change `certainty_celltype`
        l_dc_inv = dict(STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6)
        l_dc = {v: k for k, v in l_dc_inv.items()}
        try:
            return l_dc[label]
        except KeyError:
            print('Unknown label "{}"'.format(label))
            return -1
    elif gt_type == 'ctgt_j0251':
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
                             INT=8, FS=9, LTS=10)
        int2str_label = {v: k for k, v in str2int_label.items()}
        return int2str_label[label]
    else:
        raise ValueError("Given ground truth type is not valid.")