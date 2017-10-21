import re
from .basics import read_txt_from_zip
from collections import Counter
from knossos_utils.chunky import ChunkDataset, save_dataset
from knossos_utils.knossosdataset import KnossosDataset
from elektronn2.neuromancer.model import modelload
from elektronn2.utils.gpu import initgpu
from .compression import load_from_h5py, save_to_h5py
import numpy as np
import os
import sys


def load_gt_from_kzip(zip_fname, kd_p, raw_data_offset=75):
    """
    Loads ground truth from zip file, generated with Knossos. Corresponding
    dataset config file is locatet at kd_p.

    Parameters
    ----------
    zip_fname : str
    kd_p : str
    raw_data_offset : int
        additional offset for raw data to use full label volume

    Returns
    -------
    np.array, np.array
        raw data, label data
    """
    bb = parse_movement_area_from_zip(zip_fname)
    offset, size = bb[0], bb[1] - bb[0]
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_p)
    scaling = np.array(kd.scale, dtype=np.int)
    raw_data_offset = np.array(scaling[0] * raw_data_offset / scaling)
    print("Using scale adapted raw offset:", raw_data_offset)
    raw = kd.from_raw_cubes_to_matrix(size + 2 * raw_data_offset,
                                      offset - raw_data_offset, nb_threads=2,
                                      mag=1, show_progress=False)
    try:
        label = kd.from_kzip_to_matrix(zip_fname, size, offset, mag=1,
                                       verbose=False)
        label = label.astype(np.uint16)
    except:
        label = None
    return raw.astype(np.float32) / 255., label


def predict_kzip(kzip_p, m_path, kd_path, clf_thresh=0.5, mfp_active=False,
                 dest_path=None, overwrite=False, gpu_ix=0,
                 imposed_patch_size=None):
    """
    Extract objects from kzip

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
    if not os.path.isfile(dest_path + "/%s_data.h5" % cube_name) or overwrite:
        raw, labels = load_gt_from_kzip(kzip_p, kd_p=kd_path,
                                        raw_data_offset=0)
        raw = xyz2zxy(raw)
        initgpu(gpu_ix)
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
    assert vol.ndim == 3
    # adapt data to ELEKTRONN conventions (speed-up)
    vol = vol.swapaxes(1, 0)  # y x z
    vol = vol.swapaxes(0, 2)  # z x y
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
    assert vol.ndim == 3
    vol = vol.swapaxes(1, 0)  # x z y
    vol = vol.swapaxes(1, 2)  # x y z
    return vol


def create_h5_from_kzip(zip_fname, kd_p, foreground_ids=None):
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
    """
    raw, label = load_gt_from_kzip(zip_fname, kd_p)
    fname, ext = os.path.splitext(zip_fname)
    if fname[-2:] == ".k":
        fname = fname[:-2]
    create_h5_gt_file(fname, raw, label, foreground_ids)


def create_h5_gt_file(fname, raw, label, foreground_ids=None):
    """
    Create .h5 files for ELEKTRONN input from two arrays.
    Only supports binary labels (0=background, 1=foreground).

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
    """
    label = binarize_labels(label, foreground_ids)
    label = xyz2zxy(label)
    print("Raw:", raw.shape, raw.dtype, raw.min(), raw.max())
    print("Label:", label.shape, label.dtype, label.min(), label.max())
    print("-----------------\nGT Summary:\n%s\n" % \)
          str(Counter(label.flatten()).items())
    if not fname[:-2] == "h5":
        fname = fname + ".h5"
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
    np.array (2, 3)
        Movement Area
    """
    anno_str = read_txt_from_zip(zip_fname, "annotation.xml")
    line = re.findall("MovementArea (.*)/>", anno_str)
    assert len(line) == 1
    line = line[0]
    bb_min = np.array([re.findall('min.\w="(\d+)"', line)], dtype=np.uint)
    bb_max = np.array([re.findall('max.\w="(\d+)"', line)], dtype=np.uint)
    return np.concatenate([bb_min, bb_max])


def pred_dataset(kd_p, kd_pred_p, cd_p, model_p, imposed_patch_size=None,
                 mfp_active=False, gpu_ix=0):
    """
    Runs prediction on whole knossos dataset.
    Imposed patch size has to be given in Z, X, Y!

    Parameters
    ----------
    kd_p : str
        path to knossos dataset .conf file
    kd_pred_p : str
        path tho knossos dataset head folder which will contain the prediction
    cd_p : str
        destination folder for chunk dataset containing prediction
    model_p : str
        path tho ELEKTRONN2 model
    imposed_patch_size : tuple or None
        patch size (Z, X, Y) of the model
    mfp_active : bool
        activate max-fragment pooling (might be necessary to change patch_size)
    gpu_ix : int

    Returns
    -------

    """
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_p, fixed_mag=1)
    initgpu(gpu_ix)
    m = modelload(model_p, imposed_patch_size=list(imposed_patch_size)
    if isinstance(imposed_patch_size, tuple) else imposed_patch_size,
                  override_mfp_to_active=mfp_active, imposed_batch_size=1)
    original_do_rates = m.dropout_rates
    m.dropout_rates = ([0.0, ] * len(original_do_rates))
    # print kd.boundary
    offset = m.target_node.shape.offsets
    offset = np.array([offset[1], offset[2], offset[0]], dtype=np.int)
    cd = ChunkDataset()
    cd.initialize(kd, kd.boundary, [512, 512, 128], cd_p,
                  overlap=offset, box_coords=np.zeros(3))
    nb_ch = len(cd.chunk_dict.keys())
    print("Starting prediction of %d chunks.\n" % nb_ch)
    cnt = 0
    for chunk in cd.chunk_dict.values():
        sys.stdout.write("%d/%d" % (cnt, nb_ch))
        chunk_pred(chunk, m)
        cnt += 1
    save_dataset(cd)
    kd_pred = KnossosDataset()
    kd_pred.initialize_without_conf(kd_pred_p, kd.boundary, kd.scale,
                                    kd.experiment_name, mags=[1])
    cd.export_cset_to_kd(kd_pred, "pred", ["pred"], [2, 2], as_raw=True,
                         stride=[150, 150, 150])


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


def chunk_pred(ch, model):
    """
    Helper function to write chunks.

    Parameters
    ----------
    ch : Chunk
    model : str or model object
    """
    raw = ch.raw_data()
    pred = prediction_helper(raw, model)
    ch.save_chunk(pred, "pred", "pred", overwrite=True)