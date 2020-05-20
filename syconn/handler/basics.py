# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from . import log_handler
from .. import global_params

import collections
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import re
import gc
import signal
import contextlib
import glob
import tqdm
import numpy as np
import networkx as nx
from typing import List, Union
from plyfile import PlyData
from knossos_utils.skeleton import SkeletonAnnotation, SkeletonNode
from knossos_utils import KnossosDataset


def kd_factory(kd_path: str, channel: str = 'jpg'):
    """
    Initializes a KnossosDataset at the given `kd_path`.

    Notes:
        * Prioritizes pyk.conf files.

    Todo:
        * Requires additional adjustment of the data type,
          i.e. setting the channel explicitly currently leads to uint32 <->
          uint64 issues in the CS segmentation.

    Args:
        kd_path: Path to the KnossosDataset.
        channel: Channel which to use. Currently not used.

    Returns:

    """
    kd = KnossosDataset()
    # TODO: set appropriate channel
    # # kd.set_channel(channel)

    if os.path.isfile(kd_path):
        kd.initialize_from_conf(kd_path)
    elif len(glob.glob(f'{kd_path}/*.pyk.conf')) == 1:
        pyk_confs = glob.glob(f'{kd_path}/*.pyk.conf')
        kd.initialize_from_pyknossos_path(pyk_confs[0])
    elif os.path.isfile(kd_path + "/mag1/knossos.conf"):
        # Initializes the dataset by parsing the knossos.conf in path + "mag1"
        kd_path += "/mag1/knossos.conf"
        kd.initialize_from_knossos_path(kd_path)
    else:
        raise ValueError(f'Could not find KnossosDataset config at {kd_path}.')

    return kd


def switch_array_entries(this_array, entries):
    entry_0 = this_array[entries[0]]
    this_array[entries[0]] = this_array[entries[1]]
    this_array[entries[1]] = entry_0
    return this_array


def crop_bool_array(arr):
    """
    Crops a bool array to its True region

    Args:
        arr: 3d bool array
            array to crop

    Returns: d bool array, list
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

    Args:
        arr: np.array

    Returns: scalar

    """
    uni_el, cnts = np.unique(arr, return_counts=True)
    return uni_el[np.argmax(cnts)]


def get_paths_of_skelID(id_list, traced_skel_dir):
    """
    Gather paths to kzip of skeletons with ID in id_list

    Args:
        id_list: list of str
            skeleton ID's
        traced_skel_dir: str
            directory of mapped skeletons

    Returns: list of str
        paths of skeletons in id_list

    """
    mapped_skel_paths = get_filepaths_from_dir(traced_skel_dir)
    mapped_skel_ids = re.findall(r'iter_\d+_(\d+)', ''.join(mapped_skel_paths))
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

    Args:
        coords: np.array
            scaled cooridnates
        scaling: tuple
        add_edges: bool

    Returns: SkeletonAnnotation

    """
    if scaling is None:
        scaling = global_params.config['scaling']
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

    Args:
        directory: str
            path to lookup directory
        ending: tuple/list/str
            ending(s) of files
        recursively: boolean
            add files from subdirectories
        exclude_endings: bool
            filenames with endings defined in endings will not be added
        fname_includes: str or list
            file names with this substring(s) will be added

    Returns: list of str
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

    Args:
        zip_fname: str
        fname_in_zip: str

    Returns: bytes

    """
    with zipfile.ZipFile(zip_fname, allowZip64=True) as z:
        txt = z.read(fname_in_zip)
    return txt


def read_mesh_from_zip(zip_fname, fname_in_zip):
    """
    Read ply file from zip. Currently does not support normals!

    Args:
        zip_fname: str
        fname_in_zip: str

    Returns: np.array, np.array, np.array

    """
    with zipfile.ZipFile(zip_fname, allowZip64=True) as z:
        txt = z.open(fname_in_zip)
        plydata = PlyData.read(txt)
        vert = plydata['vertex'].data
        vert = vert.view((np.float32, len(vert.dtype.names))).flatten()
        ind = np.array(plydata['face'].data['vertex_indices'].tolist()).flatten()
        # TODO: support normals
        # norm = plydata['normals'].data
        # norm = vert.view((np.float32, len(vert.dtype.names))).flatten()
    return [ind, vert, None]


def write_txt2kzip(kzip_path, text, fname_in_zip, force_overwrite=False):
    """
    Write string to file in k.zip.

    Args:
        kzip_path: str
        text: str or bytes
        fname_in_zip: str
            name of file when added to zip
        force_overwrite: bool

    Returns:

    """
    texts2kzip(kzip_path, [text], [fname_in_zip],
               force_overwrite=force_overwrite)


def texts2kzip(kzip_path, texts, fnames_in_zip, force_overwrite=False):
    """
    Write strings to files in k.zip.

    Args:
        kzip_path: str
        texts: List[str]
        fnames_in_zip: List[str]
            name of file when added to zip
        force_overwrite: bool

    Returns:

    """
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
            log_handler.error("Couldn't open file {} for reading and overwri"
                              "ting. {}".format(kzip_path, e))
    else:
        try:
            with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i in range(len(texts)):
                    zf.writestr(fnames_in_zip[i], texts[i])
        except Exception as e:
            log_handler.error("Couldn't open file {} for writing. {}"
                              "".format(kzip_path, e))


def write_data2kzip(kzip_path, fpath, fname_in_zip=None, force_overwrite=False):
    """
    Write file to k.zip.

    Args:
        kzip_path: str
        fpath: str
        fname_in_zip: str
            name of file when added to zip
        force_overwrite: bool

    Returns:

    """
    data2kzip(kzip_path, [fpath], [fname_in_zip], force_overwrite)


def data2kzip(kzip_path, fpaths, fnames_in_zip=None, force_overwrite=True,
              verbose=False):
    """
    Write files to k.zip. Finally removes files at `fpaths`.

    Args:
        kzip_path: str
        fpaths: List[str]
        fnames_in_zip: List[str]
            name of file when added to zip
        force_overwrite: bool
        verbose: bool

    Returns:

    """
    nb_files = len(fpaths)
    if verbose:
        log_handler.info('Writing {} files to .zip.'.format(nb_files))
        pbar = tqdm.tqdm(total=nb_files, leave=False)
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
            log_handler.error("Couldn't open file {} for reading and"
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
            log_handler.error("Couldn't open file {} for writing. Error: "
                              "{}".format(kzip_path, e))
    for ii in range(nb_files):
        os.remove(fpaths[ii])
    if verbose:
        log_handler.info('Done writing {} files to .zip.'.format(nb_files))
        pbar.close()


def remove_from_zip(zipfname, *filenames):
    """
    Removes filenames from zipfile

    Args:
        zipfname: str
            Path to zipfile
        *filenames: list of str
            files to delete

    Returns:

    """
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
    """
    Writes object to pickle file

    Args:
        path: str
            Destination.
        objects: object

    Returns:

    """
    gc.disable()
    if isinstance(path, str):
        with open(path, 'wb') as output:
            pkl.dump(objects, output, -1)
    else:
        log_handler.warn("Write_obj2pkl takes arguments 'path' (str) and "
                         "'objects' (python object).")
        with open(objects, 'wb') as output:
            pkl.dump(path, output, -1)
    gc.enable()


def load_pkl2obj(path):
    """
    Loads pickle file of object

    Args:
        path: str
            path of source file

    Returns:

    """
    gc.disable()
    try:
        with open(path, 'rb') as inp:
            objects = pkl.load(inp)
    except UnicodeDecodeError:  # python3 compatibility
        with open(path, 'rb') as inp:
            objects = pkl.loads(inp.read(), encoding='bytes')
        objects = convert_keys_byte2str(objects)
    gc.enable()
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


def chunkify(lst: Union[list, np.ndarray], n: int) -> List[list]:
    """
    Splits list into ``np.min([n, len(lst)])`` sub-lists.

    Args:
        lst:
        n:
    Examples:
        >>> chunkify(np.arange(10), 2)
        >>> chunkify(np.arange(10), 100)

    Returns:
        List of chunks. Length is ``np.min([n, len(lst)])``.
    """
    if len(lst) < n:
        n = len(lst)
    return [lst[i::n] for i in range(n)]


def chunkify_weighted(lst, n, weights):
    """
    splits list into n sub-lists according to weights.

    Args:
        lst: list
        n: int
        weights: array

    Returns:

    """
    if len(lst) < n:
        n = len(lst)
        return [lst[i::n] for i in range(n)]  # no weighting needed
    ordered = np.argsort(weights)
    lst = lst[ordered[::-1]]
    return [lst[i::n] for i in range(n)]


def chunkify_successive(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten_list(lst):
    """
    Flattens list of lists. Same ordering as np.concatenate

    Args:
        lst: list of lists

    Returns: list

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

    Args:
        x:

    Returns: flattend x

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

    Args:
        skel_path: str
            path to skeleton

    Returns: int
        skeleton ID

    """
    return int(re.findall(r'iter_0_(\d+)', skel_path)[0])


def safe_copy(src, dest, safe=True):
    """
    Copies file and throws exception if destination exists. Taken from
    Misandrist on Stackoverflow (03/31/17).

    Args:
        src: str
            path to source file
        dest: str
            path to destination file
        safe: bool
            If False, copies file with replacement

    Returns:

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

    Args:
        txt: str or bytes

    Returns: dict

    """
    cc_dict = {}
    for line in txt.splitlines()[::4]:
        if type(line) is bytes:
            curr_line = line.decode()
        else:
            curr_line = line
        line_nb = np.array(re.findall(r"(\d+)", curr_line), dtype=np.uint)
        curr_ixs = line_nb[3:]
        cc_ix = line_nb[0]
        curr_ixs = curr_ixs[curr_ixs != 0]
        cc_dict[cc_ix] = curr_ixs
    return cc_dict


def parse_cc_dict_from_kml(kml_path):
    """
    Parse connected components from knossos mergelist text file

    Args:
        kml_path: str

    Returns: dict

    """
    txt = open(kml_path, "rb").read().decode()
    return prase_cc_dict_from_txt(txt)


def parse_cc_dict_from_g(g):
    cc_dict = {}
    # use minimum ID in CC as SSV ID
    for cc in sorted(nx.connected_components(g), key=len, reverse=True):
        cc_dict[cc[0]] = cc
    return cc_dict


def parse_cc_dict_from_kzip(k_path):
    """

    Args:
        k_path: str

    Returns: dict

    """
    txt = read_txt_from_zip(k_path, "mergelist.txt").decode()
    return prase_cc_dict_from_txt(txt)


@contextlib.contextmanager
def temp_seed(seed):
    """
    From https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed

    Args:
        seed:

    Returns:

    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def str_delta_sec(seconds: int) -> str:
    """
    String time formatting - omits time units which are zero.

    Examples:
        >>> sec = 2 * 24 * 3600 + 12 * 3600 + 5 * 60 + 1
        >>> str_rep = str_delta_sec(sec)
        >>> assert str_rep == '2d:12h:05min:01s'
        >>> assert str_delta_sec(4 * 3600 + 20 * 60 + 10) == '4h:20min:10s'

    Args:
        seconds: Number of seconds, e.g. result of a time delta.

    Returns:
        String representation, e.g. ``'2d:12h:05min:01s'`` for
        ``sec = 1 + 5 * 60 + 12 * 3600 + 2 * 24 * 3600``.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    str_rep = ''
    if d > 0:
        str_rep += f'{d:d}d:'
    if h > 0:
        str_rep += f'{h:d}h:'
    if m > 0:
        str_rep += f'{m:02d}min:'
    str_rep += f'{s:02d}s'
    return str_rep
