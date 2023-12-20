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
import os
import re
import shutil
from collections import Counter
from logging import Logger
from typing import Iterable, Union, Optional, Any, Tuple, List

import numpy as np
from knossos_utils import knossosdataset
from knossos_utils.chunky import ChunkDataset, save_dataset
from knossos_utils.knossosdataset import KnossosDataset
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from .basics import read_txt_from_zip, get_filepaths_from_dir, \
    parse_cc_dict_from_kzip
from .compression import load_from_h5py, save_to_h5py
from .. import global_params
from ..handler import log_handler, log_main, basics
from ..handler.basics import chunkify
from ..handler.config import initialize_logging
from ..mp import batchjob_utils as qu
from ..proc.image import apply_morphological_operations
from ..reps import log_reps

# for readthedocs build
try:
    import torch
except ImportError:
    pass


def load_gt_from_kzip(zip_fname, kd_p, raw_data_offset=75, verbose=False,
                      mag=1):
    """
    Loads ground truth from zip file, generated with Knossos. Corresponding
    dataset config file is located at `kd_p`.
    
    Args:
        zip_fname (str): Path to the zip file containing ground truth data.
        kd_p (str or List[str]): Path or list of paths to the Knossos dataset
            configuration file(s).
        raw_data_offset (int or np.array): Number of voxels for additional raw
            offset, i.e., the offset for the raw data will be label_offset -
            raw_data_offset, while the raw data volume will be label_volume +
            2*raw_data_offset. It will use 'kd.scaling' to account for dataset
            anisotropy if scalar or a list of length 3 has to be provided for
            a custom x, y, z offset. Defaults to 75 if not provided.
        verbose (bool, optional): Enables verbose output. Defaults to False.
        mag (int): Magnification level of the data. Defaults to 1 if not
            provided.
    
    Returns:
        tuple: A tuple containing two numpy arrays:
            - raw data (float32) normalized by 1/255.
            - label data (uint16)
    """
    if type(kd_p) is str or type(kd_p) is bytes:
        kd_p = [kd_p]
    raw_data = []
    label_data = []
    for curr_p in kd_p:
        kd = basics.kd_factory(curr_p)
        bb = kd.get_movement_area(zip_fname)
        offset, size = bb[0], bb[1] - bb[0]
        scaling = np.array(kd.scale, dtype=np.int32)
        if np.isscalar(raw_data_offset):
            raw_data_offset = np.array(scaling[0] * raw_data_offset / scaling,
                                       dtype=np.int32)
            if verbose:
                log_handler.debug(f'Using scale adapted raw offset: {raw_data_offset}')
        elif len(raw_data_offset) != 3:
            raise ValueError("Offset for raw cubes has to have length 3.")
        else:
            raw_data_offset = np.array(raw_data_offset)
        raw = kd.load_raw(size=(size // mag + 2 * raw_data_offset) * mag,
                          offset=(offset // mag - raw_data_offset) * mag,
                          mag=mag).swapaxes(0, 2)
        raw_data.append(raw[None,])
        label = kd.load_kzip_seg(zip_fname, mag=mag).swapaxes(0, 2)
        label = label
        label_data.append(label[None,])
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
    Predicts data contained in a k.zip file using a specified predictive model.
    
    Args:
        kzip_p (str): Path to the k.zip file containing raw data cube information.
        m_path (str): Path to the predictive model.
        kd_path (str): Path to the Knossos dataset configuration file.
        clf_thresh (float, optional): Classification threshold. If not specified,
            the function uses a default value.
        mfp_active (bool, optional): Flag to activate max-fragment pooling. Defaults
            to False. When set to False, the prediction uses standard processing.
        dest_path (str, optional): Destination folder path. If None, the folder of
            k.zip is used. Defaults to None.
        overwrite (bool, optional): Flag to enable overwriting of existing files.
            Defaults to False.
        gpu_ix (int, optional): GPU index for model prediction. Defaults to 0.
        imposed_patch_size (tuple, optional): Imposed patch size for the model. If
            not specified, the model determines the patch size.
    
    Returns:
        The function returns predictions for the data in the specified k.zip file. If
        a destination path is provided, it will save the predictions there, otherwise,
        it saves in the folder of the k.zip file. The function supports GPU prediction,
        overwriting existing files, and custom patch sizes when specified.
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
        pred = m.predict_dense(raw[None,], pad_raw=True)[1]
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
    Predicts data from an h5 file using a specified predictive model.
    
    Args:
        h5_path (str): Path to the h5 file containing raw data.
        m_path (str): Path to the predictive model.
        clf_thresh (float, optional): Classification threshold. If None, no thresholding
            is applied.
        mfp_active (bool): Flag to activate max-fragment pooling. Defaults to False.
        gpu_ix (int, optional): GPU index for model prediction. Defaults to 0.
        imposed_patch_size (tuple, optional): Imposed patch size for the model.
        hdf5_data_key (str, optional): Key for raw data in the h5 file. If None, the first
            entry is used.
        data_is_zxy (bool): Flag to indicate data order. If False, data is assumed to be in
            [X, Y, Z] format.
        dest_p (str): Destination path for the prediction output.
        dest_hdf5_data_key (str, optional): Key for predicted data in the output h5 file.
            Defaults to 'pred'.
        as_uint8 (bool, optional): Flag to store prediction as uint8. Defaults to True.
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
    pred = m.predict_dense(raw[None,], pad_raw=True)[1]
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
        dest_p (str): Path to the destination k.zip file.
        vol (np.ndarray): Segmentation or prediction volume as an unsigned integer
                           array in XYZ order.
        offset (np.ndarray): Offset of the volume in the dataset.
        kd_path (str): Path to the Knossos dataset configuration file.
    
    Returns: 
        np.ndarray: A numpy array with shape [Z, X, Y] following the segmentation
                    or prediction volume.
    """
    kd = basics.kd_factory(kd_path)
    kd.from_matrix_to_cubes(offset=offset, kzip_path=dest_p,
                            mags=[1], data=vol)


def xyz2zxy(vol: np.ndarray) -> np.ndarray:
    """
    Swaps axes to ELEKTRONN convention ([M, .., X, Y, Z] -> [M, .., Z, X, Y]).
    
    Args:
        vol (np.ndarray): Input volume in the shape [M, .., X, Y, Z].
    
    Returns:
        np.ndarray: Reordered volume in the shape [M, .., Z, X, Y].
    """
    # assert vol.ndim == 3  # removed for multi-channel support
    # adapt data to ELEKTRONN conventions (speed-up)
    vol = vol.swapaxes(-2, -3)  # y x z
    vol = vol.swapaxes(-3, -1)  # z x y
    return vol


def zxy2xyz(vol: np.ndarray) -> np.ndarray:
    """
    Swaps axes to ELEKTRONN convention, converting a volume from ZXY to XYZ order.
    
    Args:
        vol (np.ndarray): Input volume with shape [M, .., Z, X, Y].
    
    Returns:
        np.ndarray: The volume rearranged to shape [M, .., X, Y, Z].
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
    Create .h5 files for elektronn3 (zyx) input. Only supports binary labels
    (0=background, 1=foreground).
    
    Examples:
        Suppose your k.zip file contains the segmentation GT with two
        segmentation IDs 1, 2 and is stored at ``zip_fname``. The
        corresponding ``KnossosDataset`` is located at ``kd_p``.
        The following code snippet will create an ``.h5`` file in the
        folder of ``zip_fname`` with the raw data (additional offset
        controlled by ``raw_data_offset``) and the label data (either
        binary or defined by ``target_labels``) with the keys ``raw`` and
        ``label`` respectively:
    
            create_h5_from_kzip(d_p=kd_p, raw_data_offset=75,
            zip_fname=zip_fname, mag=1, foreground_ids=[1, 2],
            target_labels=[1, 2])
    
    Args:
        zip_fname (str): Path to the annotated kzip file.
        kd_p (str): Path to the underlying raw data stored as KnossosDataset.
        foreground_ids (Optional[Iterable[int]], optional): IDs to be
            converted to foreground (1). Defaults to None, meaning all
            non-zero are foreground.
        overwrite (bool, optional): If True, overwrites existing h5 files.
            Defaults to False.
        raw_data_offset (int, optional): Number of voxels for additional
            raw offset. Defaults to 75. Set to 0 if `debug` is True.
        debug (bool, optional): If True, adds a 'debug' suffix and adapts
            bit depths. Defaults to False.
        mag (int, optional): Magnification level of the data. Defaults to 1.
        squeeze_data (bool, optional): If True, squeezes label and raw data.
            Defaults to False.
        target_labels (Optional[Iterable[int]], optional): Target labels
            for mapping foreground IDs. Must be set if `foreground_ids`
            is specified. Defaults to None.
        apply_mops_seg (Optional[List[str]], optional): Morphological
            operations to apply to segmentation. Defaults to None.
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
    Creates an h5 file for ELEKTRONN input from raw and label arrays, supporting only
    binary labels (0=background, 1=foreground). For creating true negative cubes, set
    `foreground_ids=[]` to indicate no foreground. If `foreground_ids` is None, all
    non-zero values are treated as foreground.
    
    Args:
        fname (str): Path where the h5 file will be saved.
        raw (np.ndarray): Raw data array.
        label (np.ndarray): Label data array for binary classification.
        foreground_ids (Optional[Iterable[int]], optional): IDs to be treated as
            foreground. If None, all non-zero values are considered foreground. Defaults to None.
        target_labels (Optional[Iterable[int]], optional): If set, `foreground_ids` must also
            be set. Each ID in `foreground_ids` will be mapped to the corresponding label in
            `target_labels`. Defaults to None.
        debug (bool, optional): If True, stores labels and raw as uint8 ranging from 0 to 255
            for debugging purposes. Defaults to False.
        apply_mops_seg (Optional[List[str]], optional): List of string identifiers for
            morphological operations to be applied to segmentation. Defaults to None.
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
        labels (np.ndarray): Input label array.
        foreground_ids (Iterable[int]): IDs to be considered as foreground.
        target_labels (Optional[Iterable[int]], optional): Target labels for
            mapping foreground IDs. Defaults to None if not provided.
    
    Returns:
        np.ndarray: Transformed label array.
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
    Parses the MovementArea (bounding box of labeled volume) from an annotation.xml
    file within a (k.)zip file.
    
    Args:
        zip_fname (str): Path to the zip file containing the annotation.xml file.
    
    Returns:
        np.ndarray: Array representing the Movement Area with shape [2, 3].
    """
    anno_str = read_txt_from_zip(zip_fname, "annotation.xml").decode()
    line = re.findall("MovementArea (.*)/>", anno_str)
    assert len(line) == 1
    line = line[0]
    bb_min = np.array([re.findall(r'min.\w="(\d+)"', line)], dtype=np.uint64)
    bb_max = np.array([re.findall(r'max.\w="(\d+)"', line)], dtype=np.uint64)
    # Movement area is stored with 0-indexing! No adjustment needed
    return np.concatenate([bb_min, bb_max])


def pred_dataset(*args, **kwargs):
    log_handler.warning("'pred_dataset' will be replaced by 'predict_dense_to_kd' in"
                        " the near future.")
    return _pred_dataset(*args, **kwargs)


def _pred_dataset(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size=None,
                  mfp_active=False, gpu_id=0, overwrite=False, i=None, n=None):
    """
    Predicts a dataset or a subset of it using a specified model and saves the results.
    
    Args:
        kd_p (str): Path to the Knossos dataset configuration file.
        kd_pred_p (str): Path to the Knossos dataset head folder for the prediction.
        cd_p (str): Destination folder for the chunk dataset containing the prediction.
        model_p (str): Path to the ELEKTRONN2 model.
        imposed_patch_size (tuple, optional): Model patch size in Z, X, Y order. Defaults to None.
        mfp_active (bool, optional): Flag to activate max-fragment pooling. Defaults to False.
        gpu_id (int, optional): GPU index used for prediction. Defaults to 0.
        overwrite (bool, optional): If True, overwrites existing predictions. Defaults to False.
        i (optional): Index of the current processing unit. Defaults to None.
        n (optional): Total number of processing units. Defaults to None.
    
    Returns:
        This function does not return any values; it saves the prediction results to the specified 
        destination folder.
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
    offset = np.array([offset[1], offset[2], offset[0]], dtype=np.int32)
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
    """TODO: Use pyknossos conf like here to support bigger cube size:
        target_kd = knossosdataset.KnossosDataset()
        target_kd._cube_shape = cube_shape
        scale = np.array(global_params.config['scaling'])
        target_kd.scales = [scale, ]
        target_kd.initialize_without_conf(path, kd.boundary, scale, kd.experiment_name,
                                          mags=[1, ], create_pyk_conf=True, create_knossos_conf=False)
        target_kd = basics.kd_factory(path)  # test if init is possible
    """
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
                        overwrite: bool = False,
                        cube_shape_kd: Optional[Tuple[int]] = None):
    """
    Performs dense prediction on a Knossos dataset and writes the results to new KnossosDatasets.
    
    This function runs predictions on the entire Knossos dataset specified by `kd_path`. The
    predictions are written to new KnossosDatasets with names specified by `target_names` in the
    directory `target_path`. If no threshold is provided and there is only one channel per target
    name, the KnossosDataset will contain a probability map in the raw channel as uint8 (0..255).
    Otherwise, classification results will be written to the overlay channel.
    
    Notes:
        * Requires a significant amount of GPU memory (at least 12GB).
        * The resulting KnossosDatasets do not use pyknossos configurations.
        * The GPU memory usage should be adjustable from the configuration or determined
          automatically.
    
    Args:
        kd_path: Path to the KnossosDataset configuration file of the raw data.
        target_path: Destination directory for the output KnossosDatasets containing predictions.
        model_path: Path to the elektronn3 model used for predictions. Loaded via the
            :class:`~elektronn3.inference.inference.Predictor`.
        n_channel: Number of channels predicted by the model.
        target_names: Names of the target KnossosDatasets, e.g. `target_names=['synapse_fb', 
            'synapse_type']`. Defaults to `['pred']`. Length must match with `target_channels`.
        target_channels: Channel IDs in the prediction for each target KnossosDataset, e.g.
            `target_channels=[(1, 2)]` if the prediction has two foreground labels. Defaults to 
            `[[ix for ix in range(n_channel)]]`. Length must match with `target_names`.
        channel_thresholds: Thresholds for channels. If None and the number of channels for the
            target KnossosDataset is 1, probabilities are stored. Otherwise, defaults to 0.5.
        log: Logger for output messages.
        mag: Magnification level of the data.
        overlap_shape_tiles: Overlap in voxels [XYZ] used for each tile predicted during inference.
            Properties such as chunk size and tile count may additionally influence the overlap::
            
                chunk_size = np.array([1024, 1024, 256], dtype=np.int32)  # XYZ
                n_tiles = np.array([4, 4, 16])
                tile_shape = (chunk_size / n_tiles).astype(np.int32)
                # final input shape must be multiple of tile_shape
                overlap_shape = tile_shape // 2
    
        cube_of_interest: Bounding box of the volume of interest (min and max coordinate in voxels)
            in the respective magnification level (see kwarg `mag`).
        overwrite: Whether to overwrite existing KnossosDatasets.
        cube_shape_kd: Cube shape for storing sub-volumes in KnossosDataset on the file system.
    """
    if log is None:
        log = initialize_logging('dense_predictions', global_params.config.working_dir + '/logs/', overwrite=False)
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
    if cube_shape_kd is None:
        cube_shape_kd = (256, 256, 256)
    # TODO: these should be config parameters
    overlap_shape_tiles = np.array([30, 31, 20])
    overlap_shape = overlap_shape_tiles
    chunk_size = np.array([482, 481, 236])
    # if qu.batchjob_enabled():
    #     chunk_size *= 2
    tile_shape = [271, 181, 138]

    cd = ChunkDataset()
    cd.initialize(kd, cube_of_interest[1], chunk_size, target_path + '/cd_tmp/',
                  box_coords=cube_of_interest[0], list_of_coords=[],
                  fit_box_size=True, overlap=overlap_shape)
    chunk_ids = list(cd.chunk_dict.keys())
    # init target KnossosDatasets
    target_kd_path_list = [target_path + '/{}/'.format(tn) for tn in target_names]
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
        target_kd._cube_shape = cube_shape_kd
        scale = np.array(global_params.config['scaling'])
        target_kd.scales = [scale, ]
        # TODO: use pyk conf!
        target_kd.initialize_without_conf(path, kd.boundary, kd.scale,
                                          kd.experiment_name, [2 ** x for x in range(6)],
                                          create_pyk_conf=False, create_knossos_conf=True)
        try:  # make sure init works
            basics.kd_factory(path)
        except ValueError as e:
            log.error(f'Could not initialize KnossosDataset at "{path}". {e}')
    # init batchjob parameters
    multi_params = chunk_ids
    multi_params = chunkify(multi_params, global_params.config.ngpu_total)
    multi_params = [(ch_ids, kd_path, target_path, model_path, overlap_shape,
                     overlap_shape_tiles, tile_shape, chunk_size, n_channel, target_channels,
                     target_kd_path_list, channel_thresholds, mag, cube_of_interest)
                    for ch_ids in multi_params]
    log.info('Started dense prediction of {} in {:d} chunk(s).'.format(", ".join(target_names), len(chunk_ids)))
    n_cores_per_job = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'] if \
        qu.batchjob_enabled() else global_params.config['ncores_per_node']

    qu.batchjob_script(multi_params, "predict_dense", n_cores=n_cores_per_job, suffix='_' + '_'.join(target_names),
                       remove_jobfolder=True, log=log, additional_flags="--gres=gpu:1")
    log.info('Finished dense prediction of {}'.format(", ".join(target_names)))


def dense_predictor(args):
    """
    Transforms volumes and performs model predictions.
    
    This function is responsible for transforming volumes by switching XYZ and ZYX order before
    passing them to the model for prediction. The function may also transform the predicted
    volumes back if necessary.
    
    Args:
        args: Tuple(
            chunk_ids: list
                List of chunk IDs in the chunk dataset.
            kd_p: str
                Path to the Knossos dataset configuration file.
            cd_p: str
                Destination folder for the chunk dataset containing prediction.
            model_p: str
                Path to the model used for predictions.
            offset: Variable type and description from the old docstring.
            chunk_size: Variable type and description from the old docstring.
            ... Additional parameters related to the prediction process.
    
    Returns:
        None. The function is used for its side effects of writing predictions to disk.
    """
    # TODO: remove chunk necessity
    # TODO: clean up (e.g. redundant chunk sizes, ...)
    #
    chunk_ids, kd_p, target_p, model_p, overlap_shape, overlap_shape_tiles, tile_shape, chunk_size, n_channel, \
    target_channels, target_kd_path_list, channel_thresholds, mag, cube_of_interest = args

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
        target_kd = basics.kd_factory(path)
        target_kd_dict[path] = target_kd

    # init Predictor
    from elektronn3.inference import Predictor
    ix = 0
    tile_shape = np.array(tile_shape)
    while True:
        try:
            out_shape = (chunk_size + 2 * np.array(overlap_shape)).astype(np.int32)[::-1]  # ZYX
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
                        dtype=np.int32)

        coords = np.array(np.array(ch.coordinates) - np.array(ol),
                          dtype=np.int32)
        raw = kd.load_raw(size=size * mag, offset=coords * mag, mag=mag)

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
                    offset=ch.coordinates * mag, data=data.astype(np.uint8),
                    data_mag=mag, mags=[mag, mag * 2, mag * 4],
                    fast_resampling=True, upsample=False)
            else:
                target_kd_dict[path].save_seg(
                    offset=ch.coordinates * mag, data=data, data_mag=mag,
                    mags=[mag, mag * 2, mag * 4],
                    fast_resampling=True, upsample=False)


def dense_predicton_helper(raw: np.ndarray, predictor: 'Predictor', is_zyx=False,
                           return_zyx=False) -> np.ndarray:
    """
    Assists with the prediction of dense volumes.
    
    This function helps with the prediction of dense volumes by handling the transformation
    of data formats and calling the model's prediction method.
    
    Args:
        raw: The input data array in CXYZ format.
        predictor: The model which performs the inference. Requires `predictor.predict`.
        is_zyx: A flag indicating if the input data is already in ZYX format.
        return_zyx: A flag indicating if the output data should be in ZYX format.
    
    Returns:
        The inference result in CXYZ format as uint8 ranging from 0 to 255.
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
    Converts a chunk dataset to a Knossos dataset using a model for prediction.
    
    This function is deprecated and will be replaced by `predict_dense_to_kd`. It predicts the
    entire or partial Knossos dataset and writes the results to a Knossos dataset.
    
    Args:
        kd_p: The path to the Knossos dataset configuration file.
        kd_pred_p: The path to the Knossos dataset directory which will contain the prediction.
        cd_p: The destination folder for the chunk dataset containing the prediction.
        model_p: The path to the ELEKTRONN2 model.
        imposed_patch_size: The patch size of the model.
        mfp_active: A flag indicating whether max-fragment pooling is active.
    
    Returns:
        None. The function is used for its side effects of writing predictions to disk.
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
    offset = np.array([offset[1], offset[2], offset[0]], dtype=np.int32)
    cd = ChunkDataset()
    cd.initialize(kd, kd.boundary, [512, 512, 256], cd_p, overlap=offset,
                  box_coords=np.zeros(3), fit_box_size=True)
    """TODO: Use pyknossos conf like here to support bigger cube size:
        target_kd = knossosdataset.KnossosDataset()
        target_kd._cube_shape = cube_shape
        scale = np.array(global_params.config['scaling'])
        target_kd.scales = [scale, ]
        target_kd.initialize_without_conf(path, kd.boundary, scale, kd.experiment_name,
                                          mags=[1, ], create_pyk_conf=True, create_knossos_conf=False)
        target_kd = basics.kd_factory(path)  # test if init is possible
    """
    kd_pred.initialize_without_conf(kd_pred_p, kd.boundary, kd.scale,
                                    kd.experiment_name, mags=[1, 2, 4, 8])
    cd.export_cset_to_kd(kd_pred, "pred", ["pred"], [4, 4], as_raw=True,
                         stride=[256, 256, 256])


def prediction_helper(raw, model, override_mfp=True,
                      imposed_patch_size=None):
    """
    Helper function for predicting raw volumes (range: 0 to 255; uint8).
    This function predicts raw volumes and converts the coordinate format from
    XYZ to the ELEKTRONN-specific ZXY before returning the prediction in the
    original XYZ format. Ensure the imposed patch size is specified in ZXY.
    
    Args:
        raw: np.array
            The raw volume data in XYZ format.
        model: str or model object
            The path to the model (.mdl) or the model object itself.
        override_mfp: bool
            A flag indicating whether to override max fragment pooling.
        imposed_patch_size: tuple
            The patch size imposed on the model, in ZXY format.
    
    Returns: np.array
        The prediction data in XYZ format.
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
    Helper function to write chunks to disk.
    
    This function writes chunks of predicted data to disk during the prediction process. It handles
    individual chunks, storing the output of the prediction made by the given model.
    
    Args:
        ch: The chunk of data to be predicted (Chunk).
        model: The model or the path to the model used for prediction (str or model object).
        debug: A flag indicating whether to run in debug mode (bool).
    
    Returns:
        None. The function writes data to disk as a side effect.
    """
    raw = ch.raw_data()
    pred = prediction_helper(raw, model) * 255
    pred = pred.astype(np.uint8)
    ch.save_chunk(pred, "pred", "pred", overwrite=True)
    if debug:
        ch.save_chunk(raw, "pred", "raw", overwrite=False)


def get_glia_model_e3():
    """
    Retrieves the elektronn3 model for glia prediction, typically trained with
    `naive_view_normalization_new`.
    
    Returns:
        The trained InferenceModel for glia prediction.
    """
    from elektronn3.models.base import InferenceModel
    m = InferenceModel(global_params.config.mpath_glia_e3, normalize_func=naive_view_normalization_new)
    return m


def get_celltype_model_e3():
    """
    Retrieves the elektronn3 model trained for cell type prediction. Unlike standard
    e3 InferenceModel instances which employ `naive_view_normalization_new`, this model
    applies view normalization in a distinct downstream inference method
    (`predict_sso_celltype`) due to the additional scalar inputs which should not undergo
    normalization.
    
    Returns:
        The trained InferenceModel for cell type prediction, which handles view
        normalization differently to accommodate additional scalar inputs.
    """
    try:
        from elektronn3.models.base import InferenceModel
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    m = torch.jit.load(global_params.config.mpath_celltype_e3)
    m = InferenceModel(m, bs=40, multi_gpu=True)
    return m


def get_semseg_spiness_model():
    """
    Retrieves the elektronn3 model trained for semantic segmentation of spines.
    
    Returns:
        The trained InferenceModel for semantic segmentation of spines.
    """
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
    m._path = path
    return m


def get_semseg_axon_model():
    """
    Retrieves the elektronn3 model trained for semantic segmentation of axons.
    
    Returns:
        The trained InferenceModel for semantic segmentation of axons.
    """
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
    m._path = path
    return m


def get_tripletnet_model_e3():
    """
    Retrieves the elektronn3 model trained with `naive_view_normalization_new`.
    
    Returns:
        The trained InferenceModel used for comparison tasks in a triplet
        network configuration.
    """
    try:
        from elektronn3.models.base import InferenceModel
    except ImportError as e:
        msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
              "com/ELEKTRONN/elektronn3' for more information.".format(e)
        log_main.error(msg)
        raise ImportError(msg)
    m = torch.jit.load(global_params.config.mpath_tnet)
    m = InferenceModel(m)
    return m


def get_myelin_cnn():
    """
    Retrieves the elektronn3 model trained to predict binary myelin-in class.
    
    Returns:
        The trained Inference model for myelin prediction.
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
    return m


def get_knn_tnet_embedding_e3():
    """
    Retrieves the K-Nearest Neighbors classifier trained on triplet network embeddings.
    
    Returns:
        The trained KNeighborsClassifier instance.
    """
    tnet_eval_dir = "{}/pred/".format(global_params.config.mpath_tnet)
    return knn_clf_tnet_embedding(tnet_eval_dir)


def get_pca_tnet_embedding_e3():
    """
    Retrieves the PCA transformation trained on triplet network embeddings.
    
    Returns:
        The trained PCA instance. This instance is fitted to the specific
        embeddings generated by the triplet network, allowing for dimensionality
        reduction and visualization of high-dimensional data.
    """
    tnet_eval_dir = "{}/pred/".format(global_params.config.mpath_tnet)
    return pca_tnet_embedding(tnet_eval_dir)


def naive_view_normalization(d):
    """
    Performs a simple normalization of the input data for backward compatibility.
    
    Args:
        d: The input data to be normalized.
    
    Returns:
        The normalized data.
    """
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
    """
    Performs a simple normalization on the input data.
    
    Args:
        d: The input data to be normalized.
    
    Returns:
        The normalized data.
    """
    return d.astype(np.float32) / 255. - 0.5


def knn_clf_tnet_embedding(fold, fit_all=False):
    """
    Trains a K-Nearest Neighbors classifier on triplet network embeddings.
    
    This method assumes embedding for GT views has been created already in 'fold'
    and stored in l_train_%d.npy / l_valid_%d.npy files.
    
    Args:
        fold (str): The directory containing the embeddings and labels.
        fit_all (bool): A flag indicating whether to fit the classifier on all data.
    
    Returns:
        The trained KNeighborsClassifier instance.
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
    Performs PCA on triplet network embeddings.
    
    Args:
        fold: The directory containing the embeddings and labels.
        n_components: The number of principal components to keep.
        fit_all: A flag indicating whether to fit the PCA on all data.
    
    Returns:
        The trained PCA instance. Assumes embedding for GT views has been
        created already in 'fold' and put into l_train_%d.npy / l_valid_%d.npy
        files.
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
    """
    Converts input views to the format required by a triplet network.
    
    Args:
        views: The input views to be converted.
    
    Returns:
        The converted views in the format required by the triplet network.
    """
    views = views[:, :, :1]  # use first view only
    out_d = np.concatenate([views,
                            np.ones_like(views),
                            np.ones_like(views)], axis=2)
    return out_d.astype(np.float32)


def certainty_estimate(inp: np.ndarray, is_logit: bool = False) -> float:
    """
    Estimates the certainty of (independent) predictions of the same sample:
        1. If `is_logit` is True, apply softmax on the input.
        2. Sum the evidence per class and (re-)normalize.
        3. Compute the entropy, scale it with the maximum entropy (equal
           probabilities) and subtract it from 1 to get the certainty.
    
    Args:
        inp: 2D array of prediction results (N: number of samples,
             C: Number of classes).
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
    # sum probabilities across samples and normalize
    proba = np.mean(proba, axis=0)
    # maximum entropy at equal probabilities: -sum(1/N*ln(1/N)) = ln(N)
    entr_max = np.log(len(proba))
    entr_norm = entropy(proba) / entr_max
    # convert to certainty estimate
    return 1 - entr_norm


def str2int_converter(comment: str, gt_type: str) -> int:
    """
    Converts semantic string labels into integer labels based on the ground truth type.
    
    Args:
        comment: The semantic string label to be converted.
        gt_type: The type of ground truth, which determines the conversion logic.
    
    Returns:
        The integer label corresponding to the semantic string label.
    """
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
                             FS=8, LTS=9)
        return str2int_label[comment]
    elif gt_type == 'ctgt_j0251_v2':
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
                             FS=8, LTS=9, NGF=10)
        return str2int_label[comment]
    else:
        raise ValueError("Given groundtruth type is not valid.")


# create function that converts information in string type to the
# information in integer type

def int2str_converter(label: int, gt_type: str) -> str:
    """
    Converts an integer label into a semantic string based on the specified ground truth
    type. The conversion is specific to the domain of connectomics, where labels
    represent various cellular structures or cell types.
    
    Args:
        label: An integer representing the label to be converted.
        gt_type: A string specifying the ground truth type. It determines the mapping
                 from integer labels to semantic strings. Examples include 'spgt' for
                 spines, 'axgt' for cell compartments, 'ctgt' for cell types, 'ctgt_v2',
                 and 'ctgt_j0251' for different versions of cell type ground truths.
    
    Returns:
        A string representing the semantic label corresponding to the input integer.
        If the label does not match any known category, a value indicating an
        unknown label is returned (commonly -1 or "N/A").
    
    Raises:
        ValueError: If the provided ground truth type is not recognized by the function.
    
    TODO:
        - Remove redundant definitions.
        - Check if the return value for unknown labels (-1) is handled appropriately
          elsewhere in the code, otherwise consider returning "N/A".
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
                             FS=8, LTS=9)
        int2str_label = {v: k for k, v in str2int_label.items()}
        return int2str_label[label]
    elif gt_type == 'ctgt_j0251_v2':
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
                             FS=8, LTS=9, NGF=10)
        int2str_label = {v: k for k, v in str2int_label.items()}
        return int2str_label[label]
    else:
        raise ValueError("Given ground truth type is not valid.")
