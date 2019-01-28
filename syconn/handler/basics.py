# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from collections import defaultdict
import collections
import numpy as np
import h5py
import os
import shutil
import tempfile
import zipfile
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from knossos_utils.skeleton import SkeletonAnnotation, SkeletonNode
import re
import signal
import contextlib
import tqdm

from . import log_handler
from .. import global_params

__all__ = ['load_from_h5py', 'save_to_h5py', 'crop_bool_array',
           'get_filepaths_from_dir', 'write_obj2pkl', 'load_pkl2obj',
           'write_data2kzip', 'remove_from_zip', 'chunkify', 'flatten_list',
           'get_skelID_from_path', 'write_txt2kzip', 'switch_array_entries',
           'parse_cc_dict_from_kzip', 'parse_cc_dict_from_kml', 'data2kzip',
           'safe_copy', 'temp_seed']


def load_from_h5py(path, hdf5_names=None, as_dict=False):
    """
    Loads data from a h5py File

    Parameters
    ----------
    path: str
    hdf5_names: list of str
        if None, all keys will be loaded
    as_dict: boolean
        if False a list is returned

    Returns
    -------
    data: dict or np.array

    """
    if as_dict:
        data = {}
    else:
        data = []
    try:
        f = h5py.File(path, 'r')
        if hdf5_names is None:
            hdf5_names = f.keys()
        for hdf5_name in hdf5_names:
            if as_dict:
                data[hdf5_name] = f[hdf5_name].value
            else:
                data.append(f[hdf5_name].value)
    except:
        raise Exception("Error at Path: %s, with labels:" % path, hdf5_names)
    f.close()
    return data


def save_to_h5py(data, path, hdf5_names=None):
    """
    Saves data to h5py File

    Parameters
    ----------
    data: list of np.arrays
    path: str
    hdf5_names: list of str
        has to be the same length as data

    Returns
    -------
    nothing

    """
    if (not type(data) is dict) and hdf5_names is None:
        raise Exception("hdf5names has to be set, when data is a list")
    if os.path.isfile(path):
        os.remove(path)
    f = h5py.File(path, "w")
    if type(data) is dict:
        for key in data.keys():
            f.create_dataset(key, data=data[key],
                             compression="gzip")
    else:
        if len(hdf5_names) != len(data):
            f.close()
            raise Exception("Not enough or to much hdf5-names given!")
        for nb_data in range(len(data)):
            f.create_dataset(hdf5_names[nb_data], data=data[nb_data],
                             compression="gzip")
    f.close()


def switch_array_entries(this_array, entries):
    entry_0 = this_array[entries[0]]
    this_array[entries[0]] = this_array[entries[1]]
    this_array[entries[1]] = entry_0
    return this_array


def crop_bool_array(arr):
    """
    Crops a bool array to its True region

    :param arr: 3d bool array
        array to crop
    :return: 3d bool array, list
        cropped array, offset
    """
    in_mask_indices = [np.flatnonzero(arr.sum(axis=(1, 2))),
                       np.flatnonzero(arr.sum(axis=(0, 2))),
                       np.flatnonzero(arr.sum(axis=(0, 1)))]

    return arr[in_mask_indices[0].min(): in_mask_indices[0].max() + 1,
               in_mask_indices[1].min(): in_mask_indices[1].max() + 1,
               in_mask_indices[2].min(): in_mask_indices[2].max() + 1],\
           [in_mask_indices[0].min(),
            in_mask_indices[1].min(),
            in_mask_indices[2].min()]


def group_ids_to_so_storage(ids, params, significant_digits=5):
    id_dict = defaultdict(list)
    param_dicts = [defaultdict(list) for _ in range(len(params))]
    for i_id in range(len(ids)):
        this_id = ids[i_id]
        this_id_str = "%.5d" % this_id
        id_dict[this_id_str[-significant_digits:]].append(this_id)
        for i_param in range(len(params)):
            param_dicts[i_param][this_id_str[-significant_digits:]].\
                append(params[i_param][i_id])

    return [id_dict] + param_dicts


def majority_element_1d(arr):
    """
    Returns most frequent element in 'arr'.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    scalar
    """
    uni_el, cnts = np.unique(arr, return_counts=True)
    return uni_el[np.argmax(cnts)]


def get_paths_of_skelID(id_list, traced_skel_dir):
    """Gather paths to kzip of skeletons with ID in id_list

    Parameters
    ----------
    id_list: list of str
        skeleton ID's
    traced_skel_dir: str
        directory of mapped skeletons

    Returns
    -------
    list of str
        paths of skeletons in id_list
    """
    mapped_skel_paths = get_filepaths_from_dir(traced_skel_dir)
    mapped_skel_ids = re.findall('iter_\d+_(\d+)', ''.join(mapped_skel_paths))
    wanted_paths = []
    for skelID in id_list:
        try:
            path = mapped_skel_paths[mapped_skel_ids.index(str(skelID))]
            wanted_paths.append(path)
        except:
            wanted_paths.append(None)
    return wanted_paths


def coordpath2anno(coords, scaling=None, add_edges=True):
    """
    Creates skeleton from scaled coordinates, assume coords are in order for
    edge creation.

    Parameters
    ----------
    coords : np.array
        scaled cooridnates
    scaling : tuple
    add_edges : bool

    Returns
    -------
    SkeletonAnnotation
    """
    if scaling is None:
        scaling = global_params.config.entries['Dataset']['scaling']
    anno = SkeletonAnnotation()
    anno.scaling = scaling
    scaling = np.array(scaling, dtype=np.int)
    rep_nodes = []
    coords = np.array(coords, dtype=np.int)
    for c in coords:
        unscaled_c = c / scaling
        n = SkeletonNode().from_scratch(anno, unscaled_c[0], unscaled_c[1],
                                        unscaled_c[2])
        anno.addNode(n)
        rep_nodes.append(n)
    if add_edges:
        for i in range(1, len(rep_nodes)):
            anno.addEdge(rep_nodes[i-1], rep_nodes[i])
    return anno


def get_filepaths_from_dir(directory, ending=('k.zip',), recursively=False,
                           exclude_endings=False, fname_includes=()):
    """
    Collect all files with certain ending from directory.

    Parameters
    ----------
    directory: str
        path to lookup directory
    ending: tuple/list/str
        ending(s) of files
    recursively: boolean
        add files from subdirectories
    fname_includes : str or list
        file names with this substring(s) will be added
    exclude_endings : bool
        filenames with endings defined in endings will not be added
    Returns
    -------
    list of str
        paths to files
    """
    # make it backwards compatible
    if type(ending) is str:
        ending = [ending]
    if type(fname_includes) is str:
        fname_includes = [fname_includes]
    files = []
    corr_incl = True
    corr_end = True
    if recursively:
        for r, s, fs in os.walk(directory):
            for f in fs:
                if len(ending) > 0:
                    corr_end = np.any(
                        [f[-len(end):] == end for end in ending])
                    if exclude_endings:
                        corr_end = not corr_end
                if len(fname_includes) > 0:
                    corr_incl = np.any([substr in f for substr in fname_includes])
                if corr_end and corr_incl:
                    files.append(os.path.join(r, f))

    else:
        for f in next(os.walk(directory))[2]:
            if len(ending) > 0:
                corr_end = np.any(
                    [f[-len(end):] == end for end in ending])
                if exclude_endings:
                    corr_end = not corr_end
            if len(fname_includes) > 0:
                corr_incl = np.any([substr in f for substr in fname_includes])
            if corr_end and corr_incl:
                files.append(os.path.join(directory, f))
    return files


def read_txt_from_zip(zip_fname, fname_in_zip):
    """
    Read text file from zip.

    Parameters
    ----------
    zip_fname : str
    fname_in_zip : str

    Returns
    -------
    bytes
    """
    with zipfile.ZipFile(zip_fname, allowZip64=True) as z:
        txt = z.read(fname_in_zip)
    return txt


def write_txt2kzip(kzip_path, text, fname_in_zip, force_overwrite=False):
    """
    Write string to file in k.zip.

    Parameters
    ----------
    kzip_path : str
    text : str or bytes
    fname_in_zip : str
        name of file when added to zip
    force_overwrite : bool
    """
    texts2kzip(kzip_path, [text], [fname_in_zip],
               force_overwrite=force_overwrite)


def texts2kzip(kzip_path, texts, fnames_in_zip, force_overwrite=False):
    """
    Write strings to files in k.zip.

    Parameters
    ----------
    kzip_path : str
    texts : List[str]
    fnames_in_zip : List[str]
        name of file when added to zip
    force_overwrite : bool
    """
    with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
        if os.path.isfile(kzip_path):
            try:
                if force_overwrite:
                    with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for i in range(len(texts)):
                            zf.writestr(fnames_in_zip[i], texts[i])
                else:
                    for i in range(len(texts)):
                        remove_from_zip(kzip_path, fnames_in_zip[i])
                    with zipfile.ZipFile(kzip_path, "a", zipfile.ZIP_DEFLATED) as zf:
                        for i in range(len(texts)):
                            zf.writestr(fnames_in_zip[i], texts[i])
            except Exception as e:
                log_handler.error("Couldn't open file %s for reading and" \
                      " overwriting." % kzip_path, e)
        else:
            try:
                with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i in range(len(texts)):
                        zf.writestr(fnames_in_zip[i], texts[i])
            except Exception as e:
                log_handler.error("Couldn't open file %s for writing." % kzip_path, e)


def write_data2kzip(kzip_path, fpath, fname_in_zip=None, force_overwrite=False):
    """
    Write file to k.zip.

    Parameters
    ----------
    kzip_path : str
    fpath : str
    fname_in_zip : str
        name of file when added to zip
    force_overwrite : bool
    """
    data2kzip(kzip_path, [fpath], [fname_in_zip], force_overwrite)


def data2kzip(kzip_path, fpaths, fnames_in_zip=None, force_overwrite=True,
              verbose=False):
    """
    Write files to k.zip.

    Parameters
    ----------
    kzip_path : str
    fpaths : List[str]
    fnames_in_zip : List[str]
        name of file when added to zip
    verbose : bool
    force_overwrite : bool
    """
    # This should now work
    # if not force_overwrite:
    #     log_handler.warning('Currently modification of data '
    #                         'in already existing kzip is still tested. "remove_from_zip" has to be adapted to'
    #                         ' work on all files in kzip.')
    nb_files = len(fpaths)
    if verbose:
        log_handler.info('Writing {} files to .zip.'.format(nb_files))
        pbar = tqdm.tqdm(total=nb_files)
    with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
        if os.path.isfile(kzip_path):
            try:
                if force_overwrite:
                    with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED,
                                         allowZip64=True) as zf:
                        for ii in range(nb_files):
                            file_name = os.path.split(fpaths[ii])[1]
                            if fnames_in_zip[ii] is not None:
                                file_name = fnames_in_zip[ii]
                            zf.write(fpaths[ii], file_name)
                            if verbose:
                                pbar.update()
                else:
                    for ii in range(nb_files):
                        file_name = os.path.split(fpaths[ii])[1]
                        if fnames_in_zip[ii] is not None:
                            file_name = fnames_in_zip[ii]
                        remove_from_zip(kzip_path, file_name)
                    with zipfile.ZipFile(kzip_path, "a", zipfile.ZIP_DEFLATED,
                                         allowZip64=True) as zf:
                        for ii in range(nb_files):
                            file_name = os.path.split(fpaths[ii])[1]
                            if fnames_in_zip[ii] is not None:
                                file_name = fnames_in_zip[ii]
                            zf.write(fpaths[ii], file_name)
                            if verbose:
                                pbar.update()
            except Exception as e:
                log_handler.error("Couldn't open file %s for reading and"
                                  " overwriting. Error: {}".format(kzip_path, e))
        else:
            try:
                with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED,
                                     allowZip64=True) as zf:
                    for ii in range(nb_files):
                        file_name = os.path.split(fpaths[ii])[1]
                        if fnames_in_zip[ii] is not None:
                            file_name = fnames_in_zip[ii]
                        zf.write(fpaths[ii], file_name)
                        if verbose:
                            pbar.update()
            except Exception as e:
                log_handler.error("Couldn't open file %s for writing. Error: {}".format(kzip_path, e))
        for ii in range(nb_files):
            os.remove(fpaths[ii])
        if verbose:
            log_handler.info('Done writing {} files to .zip.'.format(nb_files))
            pbar.close()


def remove_from_zip(zipfname, *filenames):
    """Removes filenames from zipfile

    Parameters
    ----------
    zipfname : str
        Path to zipfile
    filenames : list of str
        files to delete
    """
    with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
        tempdir = tempfile.mkdtemp()
        try:
            tempname = os.path.join(tempdir, 'new.zip')
            with zipfile.ZipFile(zipfname, 'r', allowZip64=True) as zipread:
                with zipfile.ZipFile(tempname, 'w', allowZip64=True) as zipwrite:
                    for item in zipread.infolist():
                        if item.filename not in filenames:
                            data = zipread.read(item.filename)
                            zipwrite.writestr(item, data)
            shutil.move(tempname, zipfname)
        finally:
            shutil.rmtree(tempdir)


def write_obj2pkl(path, objects):
    """Writes object to pickle file

    Parameters
    ----------
    objects : object
    path : str
        destianation
    """
    with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
        if isinstance(path, str):
            with open(path, 'wb') as output:
                pkl.dump(objects, output, -1)
        else:
            log_handler.warn("Write_obj2pkl takes arguments 'path' (str) and "
                             "'objects' (python object).")
            with open(objects, 'wb') as output:
                pkl.dump(path, output, -1)


def load_pkl2obj(path):
    """Loads pickle file of object

    Parameters
    ----------
    path: str
        path of source file

    Returns
    -------
    SegmentationDatasetObject
    """
    try:
        with open(path, 'rb') as inp:
            objects = pkl.load(inp)
    except UnicodeDecodeError: # python3 compatibility
        with open(path, 'rb') as inp:
            objects = pkl.loads(inp.read(), encoding='bytes')
        objects = convert_keys_byte2str(objects)
    return objects


def convert_keys_byte2str(dc):
    if type(dc) is not dict:
        return dc
    for k in list(dc.keys()):
        v = convert_keys_byte2str(dc[k])
        if type(k) is bytes:
            dc[k.decode('utf-8')] = v
            del dc[k]
    return dc


def chunkify(lst, n):
    """
    Splits list into n sub-lists.

    Parameters
    ----------
    lst : List
    n : int

    Returns
    -------

    """
    if len(lst) < n:
        n = len(lst)
    return [lst[i::n] for i in range(n)]


def flatten_list(lst):
    """
    Flattens list of lists. Same ordering as np.concatenate

    Parameters
    ----------
    lst : list of lists

    Returns
    -------
    list
    """
    res = np.array([el for sub in lst for el in sub])
    return res


def flatten(x):
    """
    Replacement for compiler.ast.flatten - this performs
    recursive flattening in comparison to the function above.
    Public domain code:
    https://stackoverflow.com/questions/16176742/
    python-3-replacement-for-deprecated-compiler-ast-flatten-function

    :param x:
    :return: flattend x

    """

    def iselement(e):
        return not(isinstance(e, collections.Iterable) and not isinstance(e, str))
    for el in x:
        if iselement(el):
            yield el
        else:
            # py2 compat
            # yield from flatten(el)
            for subel in flatten(el):
                yield subel


def get_skelID_from_path(skel_path):
    """
    Parse skeleton ID from filename.

    Parameters
    ----------
    skel_path : str
        path to skeleton

    Returns
    -------
    int
        skeleton ID
    """
    return int(re.findall('iter_0_(\d+)', skel_path)[0])


def safe_copy(src, dest, safe=True):
    """
    Copies file and throws exception if destination exists. Taken from
    Misandrist on Stackoverflow (03/31/17).

    Parameters
    ----------
    src : str
        path to source file
    dest : str
        path to destination file
    safe : bool
        If False, copies file with replacement
    """
    if safe:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        # Copy the file and automatically close files at the end
        with os.fdopen(fd, 'wb') as f:
            with open(src, 'rb') as sf:
                shutil.copyfileobj(sf, f)
    else:
        shutil.copy(src, dest)


# https://gist.github.com/tcwalther/ae058c64d5d9078a9f333913718bba95
# class based on: http://stackoverflow.com/a/21919644/487556
class DelayedInterrupt(object):
    def __init__(self, signals):
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals        

    def __enter__(self):
        self.signal_received = {}
        self.old_handlers = {}
        for sig in self.sigs:
            self.signal_received[sig] = False
            self.old_handlers[sig] = signal.getsignal(sig)
            def handler(s, frame):
                self.signal_received[sig] = (s, frame)
                # Note: in Python 3.5, you can use signal.Signals(sig).name
                log_handler.info('Signal %s received. Delaying KeyboardInterrupt.' % sig)
            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def __exit__(self, type, value, traceback):
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])


def prase_cc_dict_from_txt(txt):
    """
    Parse connected components from knossos mergelist text file

    Parameters
    ----------
    txt : str or bytes

    Returns
    -------
    dict
    """
    cc_dict = {}
    for line in txt.splitlines()[::4]:
        if type(line) is bytes:
            curr_line = line.decode()
        else:
            curr_line = line
        line_nb = np.array(re.findall("(\d+)", curr_line), dtype=np.uint)
        curr_ixs = line_nb[3:]
        cc_ix = line_nb[0]
        curr_ixs = curr_ixs[curr_ixs != 0]
        cc_dict[cc_ix] = curr_ixs
    return cc_dict


def parse_cc_dict_from_kml(kml_path):
    """
    Parse connected components from knossos mergelist text file

    Parameters
    ----------
    kml_path : str

    Returns
    -------
    dict
    """
    txt = open(kml_path, "rb").read().decode()
    return prase_cc_dict_from_txt(txt)


def parse_cc_dict_from_kzip(k_path):
    """

    Parameters
    ----------
    k_path : str

    Returns
    -------
    dict
    """
    txt = read_txt_from_zip(k_path, "mergelist.txt").decode()
    return prase_cc_dict_from_txt(txt)


@contextlib.contextmanager
def temp_seed(seed):
    """
    From https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    Parameters
    ----------
    seed :

    Returns
    -------

    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
