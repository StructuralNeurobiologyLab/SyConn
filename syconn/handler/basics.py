# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import collections
import contextlib
import gc
import glob
import os
import pickle as pkl
import re
import shutil
import signal
import tempfile
import zipfile
from collections import defaultdict
from typing import List, Union

import networkx as nx
import numpy as np
import tqdm
from knossos_utils import KnossosDataset
from knossos_utils.skeleton import SkeletonAnnotation, SkeletonNode
from plyfile import PlyData

from . import log_handler
from .. import global_params


def kd_factory(kd_path: str, channel: str = 'jpg'):
    """
    Initializes a KnossosDataset at the given `kd_path`.
    
    This function attempts to initialize a KnossosDataset by searching for
    configuration files in the specified path. It prioritizes pyk.conf files
    and attempts to handle different scenarios where the configuration might
    be located.
    
    Notes:
        * Prioritizes pyk.conf files.
    
    Todo:
        * Requires additional adjustment of the data type,
          i.e., setting the channel explicitly currently leads to uint32 <->
          uint64 issues in the CS segmentation.
    
    Args:
        kd_path: The file system path where the KnossosDataset configuration
                 is expected to be found.
        channel: The channel to use for the dataset. This argument is currently
                 not used in the function.
    
    Returns:
        An initialized KnossosDataset object. If the initialization fails due
        to missing configuration files, a ValueError is raised.
    
    Raises:
        ValueError: If no configuration file can be found at the specified path.
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
    """
    Switches two specified entries in an array.
    
    This function swaps the values at two specified indices in the given array.
    
    Args:
        this_array: The array in which the entries will be switched.
        entries: A list or tuple containing two indices whose corresponding
                 values in `this_array` will be swapped.
    
    Returns:
        The array with the two entries switched.
    """
    entry_0 = this_array[entries[0]]
    this_array[entries[0]] = this_array[entries[1]]
    this_array[entries[1]] = entry_0
    return this_array


def crop_bool_array(arr):
    """
    Crops a bool array to its True region.
    
    This function finds the bounding box of the True region in a 3D boolean
    array and returns the cropped array along with the offset of the crop.
    
    Args:
        arr: 3D boolean array
            The array to be cropped.
    
    Returns:
        A tuple containing the cropped 3D boolean array and a list representing
        the offset of the crop in the format [x_min, y_min, z_min].
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
    """
    Groups IDs and corresponding parameters for storage optimization.
    
    This function creates a dictionary where keys are strings representing the
    last `significant_digits` of each ID, and values are lists of IDs that
    share the same key. It also groups corresponding parameters in the same
    manner.
    
    Args:
        ids: A list of integer IDs to be grouped.
        params: A list of parameters corresponding to each ID.
        significant_digits: The number of digits from the end of the ID to use
                            for grouping.
    
    Returns:
        A list containing the dictionary of grouped IDs and dictionaries of
        grouped parameters.
    """
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
    Finds the most frequent element in a 1D array.
    
    This function returns the element that appears most frequently in the
    provided array. If multiple elements have the same highest frequency,
    the function returns the first one encountered.
    
    Args:
        arr: np.array
    
    Returns:
        scalar - The most frequent element in the array.
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
        paths of skeletons in id_list. If a skeleton ID does not have a
        corresponding file, `None` is returned in its place.
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
    Creates a SkeletonAnnotation from a path of coordinates.
    
    This function generates a SkeletonAnnotation object from a list of
    coordinates, optionally scaling them and adding edges between consecutive
    nodes. The assumption is made that the coordinates are scaled and in order
    for edge creation.
    
    Args:
        coords: A numpy array of scaled coordinates.
        scaling: A tuple representing the scaling factors for each coordinate
                 axis. If not provided, the global scaling parameters are used.
        add_edges: A boolean indicating whether to add edges between
                   consecutive nodes.
    
    Returns:
        A SkeletonAnnotation object representing the skeleton formed by the
        coordinates.
    """
    if scaling is None:
        scaling = global_params.config['scaling']
    anno = SkeletonAnnotation()
    anno.scaling = scaling
    scaling = np.array(scaling, dtype=np.int32)
    rep_nodes = []
    coords = np.array(coords, dtype=np.int32)
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
    Retrieves file paths with specific endings from a directory.
    
    This function collects all file paths from a given directory that have
    specified endings. It can search recursively and include or exclude files
    based on the endings and substrings in the filenames.
    
    Args:
        directory: The directory to search for files.
        ending: A tuple, list, or string specifying the file endings to include.
        recursively: A boolean indicating whether to search subdirectories.
        exclude_endings: A boolean indicating whether to exclude files with the
                         specified endings.
        fname_includes: A string or list of substrings that must be included in
                        the filenames.
    
    Returns:
        A list of strings representing the paths to files that match the
        specified criteria.
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
    Reads a text file from a zip archive.
    
    This function extracts and reads the contents of a text file stored within
    a zip archive.
    
    Args:
        zip_fname (str): The path to the zip file.
        fname_in_zip (str): The name of the text file within the zip archive.
    
    Returns:
        bytes: The content of the text file.
    """
    with zipfile.ZipFile(zip_fname, allowZip64=True) as z:
        txt = z.read(fname_in_zip)
    return txt


def read_mesh_from_zip(zip_fname, fname_in_zip):
    """
    Reads a PLY mesh file from a zip archive.
    
    This function extracts and reads the vertex and face data from a PLY file
    stored within a zip archive. Currently, it does not support reading normals.
    
    Args:
        zip_fname: The path to the zip file.
        fname_in_zip: The name of the PLY file within the zip archive.
    
    Returns:
        A tuple containing three np.array objects for indices, vertices, and
        normals (the latter is not supported and will be `None`).
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


def read_meshes_from_zip(zip_fname, fnames_in_zip):
    """
    Reads multiple PLY mesh files from a zip archive.
    
    This function extracts and reads the vertex and face data from multiple PLY
    files stored within a zip archive. Currently, it does not support reading
    normals or other additional data.
    
    Args:
        zip_fname: The path to the zip file containing PLY files.
        fnames_in_zip: A list of filenames of the PLY files within the zip.
    
    Returns:
        Three numpy arrays containing the vertices, faces, and a placeholder for
        normals (which is currently set to `None`) for each PLY file.
    """
    meshes = []
    with zipfile.ZipFile(zip_fname, allowZip64=True) as z:
        for fname_in_zip in fnames_in_zip:
            txt = z.open(fname_in_zip)
            plydata = PlyData.read(txt)
            vert = plydata['vertex'].data
            vert = vert.view((np.float32, len(vert.dtype.names))).flatten()
            ind = np.array(plydata['face'].data['vertex_indices'].tolist()).flatten()
            # TODO: support normals
            # norm = plydata['normals'].data
            # norm = vert.view((np.float32, len(vert.dtype.names))).flatten()
            meshes.append((ind, vert, None))
    return meshes


def write_txt2kzip(kzip_path, text, fname_in_zip, force_overwrite=False):
    """
    Writes a text string to a file within a k.zip archive.
    
    This function creates or updates a k.zip archive by adding a text or byte 
    file with the specified content. It can optionally overwrite existing 
    files if force_overwrite is set to True.
    
    Args:
        kzip_path: The path to the k.zip archive.
        text: The text or bytes content to write to the file within the archive.
        fname_in_zip: The name of the file to be created or updated within the
                      archive.
        force_overwrite: A boolean indicating whether to overwrite existing files
                         with the same name in the archive.
    
    Returns:
        None.
    """
    texts2kzip(kzip_path, [text], [fname_in_zip],
               force_overwrite=force_overwrite)


def texts2kzip(kzip_path, texts, fnames_in_zip, force_overwrite=False):
    """
    Writes multiple text strings to files within a k.zip archive.
    
    This function creates or updates a k.zip archive by adding multiple text
    files with the specified contents. It can optionally overwrite existing files
    if the 'force_overwrite' parameter is set to True.
    
    Args:
        kzip_path (str): The path to the k.zip archive.
        texts (List[str]): A list of text contents to write to the files within the
                           archive.
        fnames_in_zip (List[str]): A list of names for the files to be created or
                                   updated within the archive, indicating the name of
                                   the file when added to the zip.
        force_overwrite (bool): A boolean indicating whether to overwrite existing
                                files with the same names.
    
    Returns:
        None.
    """
    if not kzip_path.endswith('.k.zip'):
        kzip_path += '.k.zip'
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
    Writes a file to a k.zip archive.
    
    This function adds a file to a k.zip archive, optionally overwriting an
    existing file with the same name.
    
    Args:
        kzip_path (str): The path to the k.zip archive.
        fpath (str): The path to the file to be added to the archive.
        fname_in_zip (str): The name of the file within the archive. If not
                            provided, the original file name is used.
        force_overwrite (bool): A boolean indicating whether to overwrite an
                                existing file with the same name.
    
    Returns:
        None.
    """
    data2kzip(kzip_path, [fpath], [fname_in_zip], force_overwrite)


def data2kzip(kzip_path: str, fpaths, fnames_in_zip=None, force_overwrite=True,
              verbose=False):
    """
    Writes multiple files to a k.zip archive and optionally removes original files.
    
    This function adds multiple files to a k.zip archive and can optionally 
    overwrite existing files with the same names in the archive. If 
    `force_overwrite` is set to True, it will overwrite files. After adding 
    the files to the archive, it removes the original files specified by 
    `fpaths` if not contradicted by the calling code.
    
    Args:
        kzip_path: The path to the k.zip archive.
        fpaths: A list of paths to the files to be added to the archive.
        fnames_in_zip: A list of names for the files within the archive. If not
                       provided, the original file names are used.
        force_overwrite: A boolean indicating whether to overwrite existing files
                         in the archive with the same names.
        verbose: A boolean indicating whether to print progress information.
    
    Returns:
        None.
    """
    if not kzip_path.endswith('.k.zip'):
        kzip_path += '.k.zip'
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
        pbar.close()
        log_handler.info('Done writing files to .zip.')


def remove_from_zip(zipfname, *filenames):
    """
    Removes specified files from a zip archive.
    
    This function deletes files with the given names from a zip archive.
    
    Args:
        zipfname: The path to the zip archive.
        *filenames: A variable number of strings representing the names of the
                    files to be removed from the archive.
    
    Returns:
        None.
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
    Writes object to pickle file.
    
    This function writes a given object to a pickle file at the specified path.
    
    Args:
        path (str): Destination.
        objects (object): The object to be serialized and written to the file.
    
    Returns:
        None.
    """
    gc.disable()
    if isinstance(path, str):
        with open(path + ".tmp", 'wb') as output:
            pkl.dump(objects, output, protocol=pkl.HIGHEST_PROTOCOL)
        shutil.move(path + ".tmp", path)
    else:
        log_handler.warn("Write_obj2pkl takes arguments 'path' (str) and "
                         "'objects' (python object).")
        with open(objects + ".tmp", 'wb') as output:
            pkl.dump(path, output, protocol=pkl.HIGHEST_PROTOCOL)
        shutil.move(objects + ".tmp", objects)
    gc.enable()


def load_pkl2obj(path):
    """
    Deserializes and loads an object from a pickle file.
    
    This function reads a pickle file from the specified path and returns the
    deserialized object.
    
    Args:
        path (str): The path to the pickle file.
    
    Returns:
        The object deserialized from the pickle file.
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
    """
    Converts byte string keys in a dictionary to regular strings.
    
    This function recursively traverses a dictionary and converts all keys that
    are byte strings to regular strings.
    
    Args:
        dc: The dictionary with byte string keys.
    
    Returns:
        The dictionary with all keys converted to regular strings.
    """
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
    Splits a list or array into a specified number of approximately equal-sized chunks.
    
    This function divides a list or array into `n` chunks, where `n` is the
    minimum of the specified number and the length of the list. Each chunk
    contains consecutive elements from the original list.
    
    Args:
        lst: The list or numpy array to be chunked.
        n: The desired number of chunks.
    
    Examples:
        >>> chunkify(np.arange(10), 2)
        >>> chunkify(np.arange(10), 100)
    
    Returns:
        A list of chunks. Length is `np.min([n, len(lst)])`.
    """
    if len(lst) < n:
        n = len(lst)
    return [lst[i::n] for i in range(n)]


def chunkify_weighted(lst, n, weights):
    """
    Splits a list into weighted sub-lists.
    
    This function divides a list into `n` sub-lists based on the provided
    weights. The weights are not necessarily used for sorting; they determine
    the distribution of elements across the sub-lists.
    
    Args:
        lst: The list to be chunked.
        n: The number of chunks to create.
        weights: An array of weights corresponding to the elements of `lst`.
    
    Returns:
        A list of `n` chunks, where each chunk is a sublist of the original list.
    """
    if len(lst) < n:
        n = len(lst)
        return [lst[i::n] for i in range(n)]  # no weighting needed
    ordered = np.argsort(weights)
    lst = lst[ordered[::-1]]
    return [lst[i::n] for i in range(n)]


def chunkify_successive(l, n):
    """
    Yield successive n-sized chunks from l.
    
    This generator function divides a list into chunks of size `n` and yields
    each chunk in turn, ensuring that all elements in the list are 
    presented in these chunks.
    
    Args:
        l: The list to be chunked.
        n: The size of each chunk.
    
    Yields:
        A generator of successive n-sized chunks of the list `l`.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten_list(lst):
    """
    Flattens a list of lists into a single list.
    
    This function takes a list of lists and concatenates their elements into a
    single list, preserving the order of elements, similar to `np.concatenate`.
    
    Args:
        lst: A list of lists to be flattened.
    
    Returns:
        A single list containing all the elements of the sublists.
    """
    res = np.array([el for sub in lst for el in sub])
    return res


def flatten(x):
    """
    Recursively flattens a nested iterable into a flat list.
    
    This function replaces the deprecated compiler.ast.flatten by performing
    recursive flattening. It takes a nested iterable (e.g., list of lists
    of lists) and flattens it into a single list containing all the
    non-iterable elements. Originally shared under Public domain code:
    https://stackoverflow.com/questions/16176742/python-3-replacement-for-
    deprecated-compiler-ast-flatten-function
    
    Args:
        x: The nested iterable to be flattened.
    
    Returns:
        A flat list containing all the non-iterable elements of the input.
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
    Extracts the skeleton ID from a file path.
    
    This function parses the file path of a skeleton file to extract the
    skeleton ID, which is represented as an integer.
    
    Args:
        skel_path: str
            The file path of the skeleton.
    
    Returns:
        int: The ID of the skeleton.
    """
    return int(re.findall(r'iter_0_(\d+)', skel_path)[0])


def safe_copy(src, dest, safe=True):
    """
    Copies file and optionally throws an exception if destination exists.
    
    This function copies a file from the specified source path (`src`) to the
    destination path (`dest`). If the `safe` parameter is True, the function
    checks if the destination file exists and throws an exception to prevent
    overwriting. If `safe` is False, the file is copied and any existing file at
    the destination is replaced.
    
    Note: Credit to Misandrist on Stackoverflow for the original implementation
          date 03/31/17.
    
    Args:
        src (str): The source file path.
        dest (str): The destination file path.
        safe (bool): If True, raise an exception if the destination file exists.
                     If False, allows overwriting of the destination file.
    
    Returns:
        None.
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
        """
        Initializes a context manager to delay interrupts.
        
        This context manager is used to delay handling of specified signals
        until the context block is exited.
        
        Args:
            signals: A list or tuple of signal numbers to be delayed.
        """
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals        

    def __enter__(self):
        """
        Enters the context, setting up the delay for the specified signals.
        
        This method sets up handlers for the specified signals to delay their
        processing until the context is exited.
        
        Returns:
            None.
        """
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
        """
        Exits the context, restoring original signal handlers and processing any
        received signals.
        
        This method restores the original signal handlers and, if any signals
        were received during the context, processes them using the original
        handlers.
        
        Args:
            type: The type of the exception, if any occurred.
            value: The value of the exception, if any occurred.
            traceback: The traceback of the exception, if any occurred.
        
        Returns:
            None.
        """
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])


def prase_cc_dict_from_txt(txt):
    """
    Parse connected components from a Knossos mergelist text file.
    
    Args:
        txt (str or bytes): The content of a Knossos mergelist text file.
    
    Returns:
        dict: A mapping of each component ID to its associated node IDs.
    """
    cc_dict = {}
    for line in txt.splitlines()[::4]:
        if type(line) is bytes:
            curr_line = line.decode()
        else:
            curr_line = line
        line_nb = np.array(re.findall(r"(\d+)", curr_line), dtype=np.uint64)
        curr_ixs = line_nb[3:]
        cc_ix = line_nb[0]
        curr_ixs = curr_ixs[curr_ixs != 0]
        cc_dict[cc_ix] = curr_ixs
    return cc_dict


def parse_cc_dict_from_kml(kml_path):
    """
    Parses connected components from a Knossos mergelist file specified by the path and
    returns a dictionary mapping each component ID to its associated node IDs.
    
    Args:
        kml_path (str): The file path to the Knossos mergelist file.
    
    Returns:
        dict: A dictionary where each key is a component ID and the corresponding value is
              a list of node IDs belonging to that component. The information regarding the
              numpy array in the generated docstring has been replaced with 'list' from the
              old docstring to resolve the conflict in the return type information.
    """
    txt = open(kml_path, "rb").read().decode()
    return prase_cc_dict_from_txt(txt)


def parse_cc_dict_from_g(g):
    """
    Parses connected components from a graph object and returns a dictionary mapping each
    component ID to its associated node IDs.
    
    Args:
        g (networkx.Graph): The graph object containing connected components.
    
    Returns:
        dict: A dictionary where each key is a component ID and the corresponding value is a
              set of node IDs belonging to that component.
    """
    cc_dict = {}
    # use minimum ID in CC as SSV ID
    for cc in sorted(nx.connected_components(g), key=len, reverse=True):
        cc_dict[cc[0]] = cc
    return cc_dict


def parse_cc_dict_from_kzip(k_path):
    """
    Parses connected components from a Knossos mergelist text file within a zip archive and
    returns a dictionary mapping each component ID to its associated node IDs.
    
    Args:
        k_path (str): The file path to the zip archive containing the Knossos mergelist file.
    
    Returns:
        dict: A dictionary where each key is a component ID and the corresponding value is a
              numpy array of node IDs belonging to that component.
    """
    txt = read_txt_from_zip(k_path, "mergelist.txt").decode()
    return prase_cc_dict_from_txt(txt)


@contextlib.contextmanager
def temp_seed(seed):
    """
    A context manager for temporarily setting the random seed within a block of
    code to ensure reproducibility of random operations.
    (From https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed)
    
    Args:
        seed (int): The seed value to set for random number generation.
    
    Returns:
        None: This context manager does not return any value but ensures that the
              random state is reset to its original state after the block of code
              is executed.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def str_delta_sec(seconds: int) -> str:
    """
    Converts a time duration in seconds to a human-readable string, omitting time units
    that are zero.
    
    Examples:
        >>> sec = 2 * 24 * 3600 + 12 * 3600 + 5 * 60 + 1
        >>> str_rep = str_delta_sec(sec)
        >>> assert str_rep == '2d:12h:05min:01s'
        >>> assert str_delta_sec(4 * 3600 + 20 * 60 + 10) == '4h:20min:10s'
    
    Args:
        seconds (int): The time duration in seconds, e.g., result of a time delta.
    
    Returns:
        str: A human-readable string representation of the time duration, formatted as
             'Xd:Xh:XXmin:XXs' where X represents non-zero time units, e.g.
             '2d:12h:05min:01s' for sec = 1 + 5 * 60 + 12 * 3600 + 2 * 24 * 3600.
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


