# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import re
import numpy as np
import os
import sys
import time
import tqdm
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from collections import Counter
from knossos_utils.chunky import ChunkDataset, save_dataset
from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)
from knossos_utils.knossosdataset import KnossosDataset

from syconn.handler.config import initialize_logging
from syconn.reps import log_reps
from syconn.mp import batchjob_utils as qu
from syconn.handler.basics import chunkify
from elektronn3.inference import Predictor

from ..handler import log_handler
from syconn.handler import log_main
from .compression import load_from_h5py, save_to_h5py
from .basics import read_txt_from_zip, get_filepaths_from_dir,\
    parse_cc_dict_from_kzip
from .. import global_params
#import syconn.extraction.object_extraction_wrapper

def load_gt_from_kzip(zip_fname, kd_p, raw_data_offset=75, verbose=False):
    """
    Loads ground truth from zip file, generated with Knossos. Corresponding
    dataset config file is locatet at kd_p.

    Parameters
    ----------
    zip_fname : str
    kd_p : str or List[str]
    raw_data_offset : int or np.array
        number of voxels used for additional raw offset, i.e. the offset for the
        raw data will be label_offset - raw_data_offset, while the raw data
        volume will be label_volume + 2*raw_data_offset. It will
        use 'kd.scaling' to account for dataset anisotropy if scalar or a
        list of length 3 hast to be provided for a custom x, y, z offset.
    verbose : bool

    Returns
    -------
    np.array, np.array
        raw data (float32) (multiplied with 1/255.), label data (uint16)
    """
    if type(kd_p) is str or type(kd_p) is bytes:
        kd_p = [kd_p]
    bb = parse_movement_area_from_zip(zip_fname)
    offset, size = bb[0], bb[1] - bb[0]
    raw_data = []
    label_data = []
    for curr_p in kd_p:
        kd = KnossosDataset()
        kd.initialize_from_knossos_path(curr_p)
        scaling = np.array(kd.scale, dtype=np.int)
        if np.isscalar(raw_data_offset):
            raw_data_offset = np.array(scaling[0] * raw_data_offset / scaling)
            if verbose:
                print('Using scale adapted raw offset:', raw_data_offset)
        elif len(raw_data_offset) != 3:
            raise ValueError("Offset for raw cubes has to have length 3.")
        else:
            raw_data_offset = np.array(raw_data_offset)
        raw = kd.from_raw_cubes_to_matrix(size + 2 * raw_data_offset,
                                          offset - raw_data_offset, nb_threads=2,
                                          mag=1, show_progress=False)
        raw_data.append(raw[..., None])
        label = kd.from_kzip_to_matrix(zip_fname, size, offset, mag=1,
                                       verbose=False, show_progress=False)
        label = label.astype(np.uint16)
        label_data.append(label[..., None])
    raw = np.concatenate(raw_data, axis=-1).astype(np.float32)
    label = np.concatenate(label_data, axis=-1).astype(np.uint16)
    try:
        _ = parse_cc_dict_from_kzip(zip_fname)
    except:  # mergelist.txt does not exist
        label = np.zeros(size).astype(np.uint16)
        return raw.astype(np.float32) / 255., label
    return raw.astype(np.float32) / 255., label


def predict_kzip(kzip_p, m_path, kd_path, clf_thresh=0.5, mfp_active=False,
                 dest_path=None, overwrite=False, gpu_ix=0,
                 imposed_patch_size=None):
    """
    Predicts data contained in k.zip file (defined by bounding box in knossos)

    Parameters
    ----------
    kzip_p : str
        path to kzip containing the raw data cube information
    m_path : str
        path to predictive model
    kd_path : str
        path to knossos dataset
    clf_thresh : float
        classification threshold
    overwrite : bool
    mfp_active : False
    imposed_patch_size : tuple
    dest_path : str
        path to destination folder, if None folder of k.zip is used.
    gpu_ix : int
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

    Parameters
    ----------
    h5_path : str
        path to h5 containing the raw data
    m_path : str
        path to predictive model
    clf_thresh : float
        classification threshold, if None, no thresholding
    mfp_active : False
    imposed_patch_size : tuple
    gpu_ix : int
    hdf5_data_key: str
        if None, it uses the first entry in the list returned by
        'load_from_h5py'
    data_is_zxy : bool
        if False, it will assumes data is [X, Y, Z]
    as_uint8: bool
    dest_p : str
    dest_hdf5_data_key : str
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


def overlaycubes2kzip(dest_p, vol, offset, kd_path):
    """
    Writes segmentation volume to kzip.

    Parameters
    ----------
    dest_p : str
        path to k.zip
    vol : np.array [X, Y, Z]
        Segmentation or prediction (uint)
    offset : np.array
    kd_path : str

    Returns
    -------
    np.array [Z, X, Y]
    """
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)
    kd.from_matrix_to_cubes(offset=offset, kzip_path=dest_p,
                            mags=[1], data=vol)


def xyz2zxy(vol):
    """
    Swaps axes to ELEKTRONN convention ([X, Y, Z] -> [Z, X, Y]).
    Parameters
    ----------
    vol : np.array [X, Y, Z]

    Returns
    -------
    np.array [Z, X, Y]
    """
    # assert vol.ndim == 3  # removed for multi-channel support
    # adapt data to ELEKTRONN conventions (speed-up)
    vol = vol.swapaxes(-2, -3)  # y x z
    vol = vol.swapaxes(-3, -1)  # z x y
    return vol


def zxy2xyz(vol):
    """
    Swaps axes to ELEKTRONN convention ([Z, X, Y] -> [X, Y, Z]).
    Parameters
    ----------
    vol : np.array [Z, X, Y]

    Returns
    -------
    np.array [X, Y, Z]
    """
    # assert vol.ndim == 3  # removed for multi-channel support
    vol = vol.swapaxes(-2, -3)  # x z y
    vol = vol.swapaxes(-2, -1)  # x y z
    return vol


def create_h5_from_kzip(zip_fname, kd_p, foreground_ids=None, overwrite=True,
                        raw_data_offset=75, debug=False):
    """
    Create .h5 files for ELEKTRONN input. Only supports binary labels
     (0=background, 1=foreground).

    Parameters
    ----------
    zip_fname: str
    kd_p : str
    foreground_ids : iterable
        ids which have to be converted to foreground, i.e. 1. Everything
        else is considered background (0). If None, everything except 0 is
        treated as foreground.
    overwrite : bool
        If True, will overwrite existing .h5 files
    raw_data_offset : int
        number of voxels used for additional raw offset, i.e. the offset for the
        raw data will be label_offset - raw_data_offset, while the raw data
        volume will be label_volume + 2*raw_data_offset. It will
        use 'kd.scaling' to account for dataset anisotropy if scalar or a
        list of length 3 hast to be provided for a custom x, y, z offset.
    debug : bool
        if True, file will have an additional 'debug' suffix and
        raw_data_offset is set to 0. Also their bit depths are adatped to be the
        same
    """
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
    raw, label = load_gt_from_kzip(zip_fname, kd_p,
                                   raw_data_offset=raw_data_offset)
    if foreground_ids is None:
        try:
            cc_dc = parse_cc_dict_from_kzip(zip_fname)
            foreground_ids = np.concatenate(list(cc_dc.values()))
        except:  # mergelist.txt does not exist
            foreground_ids = []
        print("Foreground IDs not assigned. Inferring from "
              "'mergelist.txt' in k.zip.:", foreground_ids)
    create_h5_gt_file(fname_dest, raw, label, foreground_ids, debug=debug)


def create_h5_gt_file(fname, raw, label, foreground_ids=None, debug=False):
    """
    Create .h5 files for ELEKTRONN input from two arrays.
    Only supports binary labels (0=background, 1=foreground). E.g. for creating
    true negative cubes set foreground_ids=[] to be an empty list. If set to
    None, everything except 0 is treated as foreground.

    Parameters
    ----------
    fname: str
        Path where h5 file should be saved
    raw : np.array
    label : np.array
    foreground_ids : iterable
        ids which have to be converted to foreground, i.e. 1. Everything
        else is considered background (0). If None, everything except 0 is
        treated as foreground.
    debug : bool
        will store labels and raw as uint8 ranging from 0 to 255
    """
    print(os.path.split(fname)[1])
    label = binarize_labels(label, foreground_ids)
    label = xyz2zxy(label)
    raw = xyz2zxy(raw)
    print("Raw:", raw.shape, raw.dtype, raw.min(), raw.max())
    print("Label:", label.shape, label.dtype, label.min(), label.max())
    print("-----------------\nGT Summary:\n%s\n" %str(Counter(label.flatten()).items()))
    if not fname[-2:] == "h5":
        fname = fname + ".h5"
    if debug:
        raw = (raw * 255).astype(np.uint8)
        label = label.astype(np.uint8) * 255
    save_to_h5py([raw, label], fname, hdf5_names=["raw", "label"])


def binarize_labels(labels, foreground_ids):
    """
    Transforms label array to binary label array (0=background, 1=foreground),
    given foreground ids.

    Parameters
    ----------
    labels : np.array
    foreground_ids : iterable

    Returns
    -------
    np.array
    """
    new_labels = np.zeros_like(labels)
    if foreground_ids is None:
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
        for ix in foreground_ids:
            new_labels[labels == ix] = 1
    labels = new_labels
    assert len(np.unique(labels)) <= 2
    assert 0 <= np.max(labels) <= 1
    assert 0 <= np.min(labels) <= 1
    return labels


def parse_movement_area_from_zip(zip_fname):
    """
    Parse MovementArea (e.g. bounding box of labeled volume) from annotation.xml
    in (k.)zip file.

    Parameters
    ----------
    zip_fname : str

    Returns
    -------
    np.array
        Movement Area [2, 3]
    """
    anno_str = read_txt_from_zip(zip_fname, "annotation.xml").decode()
    line = re.findall("MovementArea (.*)/>", anno_str)
    assert len(line) == 1
    line = line[0]
    bb_min = np.array([re.findall('min.\w="(\d+)"', line)], dtype=np.uint)
    bb_max = np.array([re.findall('max.\w="(\d+)"', line)], dtype=np.uint)
    # Movement area is stored with 0-indexing! No adjustment needed
    return np.concatenate([bb_min, bb_max])


def pred_dataset(*args, **kwargs):
    log_handler.warning("'pred_dataset' will be replaced by 'predict_dataset' in"
                        " the near future.")
    return pred_dataset(*args, **kwargs)


def predict_dataset(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size=None,
                    mfp_active=False, gpu_ids=(0, ), overwrite=True):
    """
    Runs prediction on the complete knossos dataset.
    Imposed patch size has to be given in Z, X, Y!

    Parameters
    ----------
    kd_p : str
        path to knossos dataset .conf file
    kd_pred_p : str
        path to the knossos dataset head folder which will contain the
        prediction (will be created)
    cd_p : str
        destination folder for the chunk dataset containing prediction
        (will be created)
    model_p : str
        path to the ELEKTRONN2 model
    imposed_patch_size : tuple or None
        patch size (Z, X, Y) of the model
    mfp_active : bool
        activate max-fragment pooling (might be necessary to change patch_size)
    gpu_ids : tuple of int
        the GPU/GPUs to be used
    overwrite : bool
        True: fresh predictions ; False: earlier prediction continues


    Returns
    -------

    """
    if isinstance(gpu_ids, int) or len(gpu_ids) == 1:
        _pred_dataset(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size,
                 mfp_active, gpu_ids, overwrite)
    else:
        print("Starting multi-gpu prediction with GPUs:", gpu_ids)
        # TODO: replace by QSUB script
        _multi_gpu_ds_pred(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size, gpu_ids)


def _pred_dataset(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size=None,
                  mfp_active=False, gpu_id=0, overwrite=False, i=None, n=None):
    """
    Helper function for dataset prediction. Runs prediction on whole or partial
    knossos dataset. Imposed patch size has to be given in Z, X, Y!

    Parameters
    ----------
    kd_p : str
        path to knossos dataset .conf file
    kd_pred_p : str
        path to the knossos dataset head folder which will contain the prediction
    cd_p : str
        destination folder for chunk dataset containing prediction
    model_p : str
        path tho ELEKTRONN2 model
    imposed_patch_size : tuple or None
        patch size (Z, X, Y) of the model
    mfp_active : bool
        activate max-fragment pooling (might be necessary to change patch_size)
    gpu_id : int
        the GPU used
    overwrite : bool
        True: fresh predictions ; False: earlier prediction continues
        

    Returns
    -------

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
    print("Starting prediction of %d chunks in gpu %d\n" % (len(chunks),
                                                            gpu_id))

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


def predict_dense_to_kd(kd_path, target_path, model_path, n_channel, target_names=None, target_channels=None, channel_thresholds=None, log=None, mag=1):
    
    """
    Helper function for dense dataset prediction. Runs prediction on whole
    knossos dataset

    Parameters
    ----------
    kd_path : str
        path to knossos dataset .conf file
    target_path : str
        destination folder for target knossos datasets containing prediction
    model_path : str
        path to elektronn3 model for predictions
    n_channel: int
        number of channels predicted by model
        eg.: n_channel=3
    target_names: list
        names of target knossos datasets
        eg.:  target_names=['synapse_fb','synapse_type']
    target_channels: list of tuples
        channel_ids in prediction for each target knossos data set
        eg. target_channels=[(0,),(1,2)]
    channel_thresholds: list
        thresholds for channels: 
        if None and number of channels for target kd is 1: probabilities are stored
        else: 0.5 as default
        eg. channel_thresholds=[None,0.5,0.5]
    log:
        
    Returns
    -------

    """

    if target_names is None:
        target_names = ['pred']
    if target_channels is None:
        target_channels = [[i] for i in range(n_channel)]
    if channel_thresholds is None:
        channel_thresholds = [None for i in range(n_channel)]    
    if log is None:
        log = initialize_logging('dense_prediction', global_params.config.working_dir+ '/logs/', overwrite=False)

    #init KnossosDataset:
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_path, fixed_mag=mag)

    #chunck properties:
    chunk_size = np.array([512,512,512])
    n_tiles = np.array([4,4,16])
    overlap_shape=np.array([8,8,4])
    tile_shape = (chunk_size/n_tiles)
    output_dim = n_channel

    #init ChunckDataset:    
    cd = ChunkDataset()
    cd.initialize(kd, kd.boundary//mag, chunk_size, target_path, box_coords=np.zeros(3), fit_box_size=True)
    ch_dc = cd.chunk_dict
    chunk_ids = ch_dc.keys()
    
    #init target KnossosDatasets:
    target_kd_path_list = [target_path+'/kd_{}/'.format(tn) for tn in target_names]
    for path in target_kd_path_list:
        if os.path.isdir(path):
            log.debug('Found existing KD at {}. Removing it now.'.format(path))
            shutil.rmtree(path)
    for path in target_kd_path_list:
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_without_conf(path, cd.box_size, kd.scale, kd.experiment_name, kd.mag)
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_from_knossos_path(path)

    #init QSUB parameters
    multi_params = list(chunk_ids)
    max_n_jobs_gpu = global_params.NGPU_TOTAL * 2
    max_n_jobs_gpu = np.max([max_n_jobs_gpu, len(multi_params) // 100]) 
    multi_params = chunkify(multi_params, max_n_jobs_gpu)
    multi_params = [(ch_ids, kd_path, target_path, model_path, overlap_shape, tile_shape, chunk_size, output_dim, target_channels,target_kd_path_list,channel_thresholds, mag) for ch_ids in multi_params]

    log.info('Starting dense prediction.')
    path_to_out = qu.QSUB_script(multi_params, "predict_dense",
                                 n_max_co_processes=global_params.NGPU_TOTAL,
                                 suffix="", n_cores=global_params.NCORES_PER_NODE/(global_params.NGPU_TOTAL/NNODES_TOTAL))
    
    log.info('Finished dense prediction of {} Chunks'.format(len(chunk_ids)))




def dense_predictor(args):
    """

    Parameters
    ----------
    args : tuplel
        (
        chunk_ids: list
            list of chunks in chunck dataset
        kd_p : str
            path to knossos dataset .conf file
        cd_p : str
            destination folder for chunk dataset containing prediction
        model_p : str
            path to model
        offset : 
        chunk_size:
        )

    Returns
    -------

    """
  
    chunk_ids, kd_p, target_p, model_p, overlap_shape, tile_shape, chunk_size, output_dim, target_channels, target_kd_path_list,channel_thresholds, mag  = args
    
    
    #init KnossosDataset:
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_p, fixed_mag=1)

    #init ChunckDataset:
    cd = ChunkDataset()
    cd.initialize(kd, kd.boundary//mag, chunk_size, target_p,
                    box_coords=np.zeros(3), fit_box_size=True)

    #init Target KnossosDataset
    target_kd_dict = {}
    for path in target_kd_path_list:
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_from_knossos_path(path)
        target_kd_dict[path] = target_kd
    
    #init Predictor
    out_shape = np.insert(chunk_size[::-1],0,3) #output must equal chunck size
    predictor = Predictor(model_p,strict_shapes=True,tile_shape=tile_shape[::-1],out_shape=out_shape,overlap_shape=overlap_shape[::-1],apply_softmax=False)
    predictor.model.ae = False
    
    #predict Chunks:
    pbar = tqdm.tqdm(total=len(chunk_ids))
    for ch_id in chunk_ids:
        ch = cd.chunk_dict[ch_id]
        size = np.array(np.array(ch.size), dtype=np.int)
        coords = np.array(np.array(ch.coordinates), dtype=np.int)
        raw = kd.from_raw_cubes_to_matrix(size, coords, mag=mag)
        pred = dense_predicton_helper(raw, predictor)
        
        for j in range(len(target_channels)):
            ids = target_channels[j]
            path = target_kd_path_list[j]
            data = np.zeros_like(pred[0]).astype(np.uint8)
            n=0
            for i in ids:
                t = channel_thresholds[i]
                if t is not None or len(ids)>i:
                    if t is None:
                        t = 255/2
                    if t <1.:
                        t = 255*t
                    data += ((pred[i]>t)*n).astype(np.uint8)
                    n+=1
                else:
                    data = pred[i]

            target_kd_dict[path].from_matrix_to_cubes(ch.coordinates,data=data,mags=mag,fast_downsampling=False,
                overwrite=True,nb_threads=global_params.NCORES_PER_NODE/(global_params.NGPU_TOTAL/NNODES_TOTAL),
                as_raw=True,datatype=np.uint8)

        
        pbar.update(1)
    pbar.close()
       

def dense_predicton_helper(raw, predictor):
    #transform raw data
    raw = xyz2zxy(raw)
    raw = raw.astype(np.float32) / 255.
    #predict: pred of the form (N, C, [D,], H, W) 
    pred = predictor.predict(raw[None,None,])
    pred = np.array(pred[0])* 255
    pred = pred.astype(np.uint8)
    pred = zxy2xyz(pred)
    return pred



def to_knossos_dataset(kd_p, kd_pred_p, cd_p, model_p,
                       imposed_patch_size, mfp_active=False):
    """

    Parameters
    ----------
    kd_p : str
    kd_pred_p : str
    cd_p : str
    model_p :
    imposed_patch_size :
    mfp_active :

    Returns
    -------

    """
    from elektronn2.neuromancer.model import modelload

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

    Parameters
    ----------
    raw : np.array
        volume [X, Y, Z]
    model : str or model object
        path to model (.mdl)
    override_mfp : bool
    imposed_patch_size : tuple
        in Z, X, Y FORMAT!

    Returns
    -------
    np.array
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


def chunk_pred(ch, model, debug=False):
    """
    Helper function to write chunks.

    Parameters
    ----------
    ch : Chunk
    model : str or model object
    """
    raw = ch.raw_data()
    pred = prediction_helper(raw, model) * 255
    pred = pred.astype(np.uint8)
    ch.save_chunk(pred, "pred", "pred", overwrite=True)
    if debug:
        ch.save_chunk(raw, "pred", "raw", overwrite=False)


class NeuralNetworkInterface(object):
    """
    Inference class for elektronn2 models, support will end at some point.
    Switching to 'InferenceModel' in elektronn3.model.base in the long run.
    """
    def __init__(self, model_path, arch='marvin', imposed_batch_size=1,
                 channels_to_load=(0, 1, 2, 3), normal=False, nb_labels=2,
                 normalize_data=False, normalize_func=None, init_gpu=None):
        self.imposed_batch_size = imposed_batch_size
        self.channels_to_load = channels_to_load
        self.arch = arch
        self._path = model_path
        self._fname = os.path.split(model_path)[1]
        self.nb_labels = nb_labels
        self.normal = normal
        self.normalize_data = normalize_data
        self.normalize_func = normalize_func

        if init_gpu is None:
            init_gpu = 'auto'
        from elektronn2.config import config as e2config
        if e2config.device is None:
            from elektronn2.utils.gpu import initgpu
            initgpu(init_gpu)
        elektronn2.logger.setLevel("ERROR")
        from elektronn2.neuromancer.model import modelload
        self.model = modelload(model_path, replace_bn='const',
                               imposed_batch_size=imposed_batch_size)
        self.original_do_rates = self.model.dropout_rates
        self.model.dropout_rates = ([0.0, ] * len(self.original_do_rates))

    def predict_proba(self, x, verbose=False):
        x = x.astype(np.float32)
        if self.normalize_data:
            if self.normalize_func is not None:
                x = self.normalize_func(x)
            else:
                x = naive_view_normalization(x)
        bs = self.imposed_batch_size
        # using floor now and remaining samples are treated later
        if self.arch == "rec_view":
            batches = [np.arange(i * bs, (i + 1) * bs) for i in
                       range(int(np.floor(x.shape[1] / bs)))]
            proba = np.ones((x.shape[1], 4, self.nb_labels))
        elif self.arch == "triplet":
            x = views2tripletinput(x)
            batches = [np.arange(i * bs, (i + 1) * bs) for i in
                       range(int(np.floor(len(x) / bs)))]
            # nb_labels represents latent space dim.; 3 -> view triplet
            proba = np.ones((len(x), self.nb_labels, 3))
        else:
            batches = [np.arange(i * bs, (i + 1) * bs) for i in
                       range(int(np.floor(len(x) / bs)))]
            proba = np.ones((len(x), self.nb_labels))
        if verbose:
            cnt = 0
            start = time.time()
            pbar = tqdm.tqdm(total=len(batches), ncols=80, leave=False,
                             unit='it', unit_scale=True, dynamic_ncols=False)
        for b in batches:
            if verbose:
                sys.stdout.write("\r%0.2f" % (float(cnt) / len(batches)))
                sys.stdout.flush()
                cnt += 1
                pbar.update()
            x_b = x[b]
            proba[b] = self.model.predict(x_b)[None, ]
        overhead = len(x) % bs
        # TODO: add proper axis handling, maybe introduce axistags
        if overhead != 0:
            new_x_b = x[-overhead:]
            if len(new_x_b) < bs:
                add_shape = list(new_x_b.shape)
                add_shape[0] = bs - len(new_x_b)
                new_x_b = np.concatenate((np.zeros((add_shape),
                                                   dtype=np.float32), new_x_b))
            proba[-overhead:] = self.model.predict(new_x_b)[-overhead:]
        if verbose:
            end = time.time()
            sys.stdout.write("\r%0.2f\n" % 1.0)
            sys.stdout.flush()
            print("Prediction of %d samples took %0.2fs; %0.4fs/sample." %\
                  (len(x), end-start, (end-start)/len(x)))
            pbar.close()
        if self.arch == "triplet":
            return proba[..., 0]
        return proba


def get_axoness_model():
    """
    Retrained with GP dendrites. May 2018.
    """
    m = NeuralNetworkInterface(global_params.config.mpath_axoness,
                               imposed_batch_size=200,
                               nb_labels=3, normalize_data=True)
    _ = m.predict_proba(np.zeros((1, 4, 2, 128, 256)))
    return m


def get_axoness_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new` """
    from elektronn3.models.base import InferenceModel
    path = global_params.config.mpath_axoness_e3
    m = InferenceModel(path, normalize_func=naive_view_normalization_new)
    return m


def get_glia_model():
    m = NeuralNetworkInterface(global_params.config.mpath_glia,
                               imposed_batch_size=200, nb_labels=2,
                               normalize_data=True)
    _ = m.predict_proba(np.zeros((1, 1, 2, 128, 256)))
    return m


def get_glia_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new` """
    from elektronn3.models.base import InferenceModel
    path = global_params.config.mpath_glia_e3
    m = InferenceModel(path, normalize_func=naive_view_normalization_new)
    return m


def get_celltype_model(init_gpu=None):
    """
    retrained on new GT on Jan. 13th, 2019
    Parameters
    ----------
    init_gpu

    Returns
    -------

    """
    # this model was trained with 'naive_view_normalization_new'
    m = NeuralNetworkInterface(global_params.config.mpath_celltype,
                               imposed_batch_size=2, nb_labels=4,
                               normalize_data=True,
                               normalize_func=naive_view_normalization_new,
                               init_gpu=init_gpu)
    _ = m.predict_proba(np.zeros((6, 4, 20, 128, 256)))
    return m


def get_celltype_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new`
     Unlike the other e3 InferenceModel instances, here the view normalization
     is applied in the downstream inference method (`predict_sso_celltype`)
      because the celltype model also gets scalar values as input which should
      not be normalized."""
    try:
        from elektronn3.models.base import InferenceModel
    except Exception as e:  # ImportError as e:
        log_main.error(
            "elektronn3 could not be imported ({}). Please see 'https://github."
            "com/ELEKTRONN/elektronn3' for more information.".format(e))
    path = global_params.config.mpath_celltype_e3
    m = InferenceModel(path)
    return m


def get_celltype_model_large_e3():
    """Those networks are typically trained with `naive_view_normalization_new`
     Unlike the other e3 InferenceModel instances, here the view normalization
     is applied in the downstream inference method (`predict_sso_celltype`)
      because the celltype model also gets scalar values as input which should
      not be normalized."""
    try:
        from elektronn3.models.base import InferenceModel
    except Exception as e:  # ImportError as e:
        log_main.error(
            "elektronn3 could not be imported ({}). Please see 'https://github."
            "com/ELEKTRONN/elektronn3' for more information.".format(e))
    path = global_params.config.mpath_celltype_large_e3
    m = InferenceModel(path)
    return m


def get_semseg_spiness_model():
    try:
        from elektronn3.models.base import InferenceModel
    except Exception as e:  # ImportError as e:
        log_main.error(
            "elektronn3 could not be imported ({}). Please see 'https://github."
            "com/ELEKTRONN/elektronn3' for more information.".format(e))
    path = global_params.config.mpath_spiness
    m = InferenceModel(path)
    m._path = path
    return m


def get_semseg_axon_model():
    try:
        from elektronn3.models.base import InferenceModel
    except Exception as e:  # ImportError as e:
        log_main.error(
            "elektronn3 could not be imported ({}). Please see 'https://github."
            "com/ELEKTRONN/elektronn3' for more information.".format(e))
    path = global_params.config.mpath_axonsem
    m = InferenceModel(path)
    m._path = path
    return m


def get_tripletnet_model_e3():
    """Those networks are typically trained with `naive_view_normalization_new` """
    try:
        from elektronn3.models.base import InferenceModel
    except Exception as e:  # ImportError as e:
        log_main.error(
            "elektronn3 could not be imported ({}). Please see 'https://github."
            "com/ELEKTRONN/elektronn3' for more information.".format(e))
    m_path = global_params.config.mpath_tnet
    m = InferenceModel(m_path)
    return m


def get_tripletnet_model_large_e3():
    """Those networks are typically trained with `naive_view_normalization_new` """
    try:
        from elektronn3.models.base import InferenceModel
    except Exception as e:  # ImportError as e:
        log_main.error(
            "elektronn3 could not be imported ({}). Please see 'https://github."
            "com/ELEKTRONN/elektronn3' for more information.".format(e))
    m_path = global_params.config.mpath_tnet_large
    m = InferenceModel(m_path)
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

    Parameters
    ----------
    fold : str
    fit_all : bool

    Returns
    -------

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

    Parameters
    ----------
    fold : str
    n_components : int
    fit_all : bool

    Returns
    -------

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


def _multi_gpu_ds_pred(kd_p, kd_pred_p, cd_p, model_p,
                       imposed_patch_size=None, gpu_ids=(0, 1)):
    # TODO: replace by QSUB_script
    import threading

    def start_partial_pred(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size,
                           gpuid, i, n):

        fpath = os.path.dirname(os.path.abspath(__file__))
        path, file = os.path.split(os.path.dirname(fpath))
        cmd = "python {0}/syconn/handler/partial_ds_pred.py {1} {2} {3} {4}" \
              " {5} {6} {7} {8}".format(path, kd_p, kd_pred_p, cd_p, model_p,
                                        imposed_patch_size, gpuid, i, n)
        os.system(cmd)

    for ii, gi in enumerate(gpu_ids):
        args = (kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size, gi, ii,
                len(gpu_ids))
        t = threading.Thread(target=start_partial_pred, args=args)
        t.daemon = True
        t.start()
