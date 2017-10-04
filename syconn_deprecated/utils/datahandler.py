# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np
import re
import h5py
import os
import shutil
import tempfile
import zipfile
from multiprocessing import cpu_count
from basics import *
from segmentationdataset import load_dataset
from knossos_utils import skeleton_utils as su
from ..utils.segmentationdataset import UltrastructuralDataset
from knossos_utils.skeleton import SkeletonAnnotation
import cPickle as pickle


class DataHandler(object):
    """Initialized with paths or cell components (SegmentationObjects), path to
    membrane prediction and source path of traced skeletons (to be computed).
    DataHandler is needed for further processing.

    Attributes
    ----------
    wd : str
        path to working directory, by default all other paths will lead to
        subfolders of this directory
    mem_path : str
        path to barrier prediction
    cs_path : str
        path to supervoxel data
    myelin_ds_path : str
        path to myelin prediction
    skeleton_path : str
        path to cell tracings
    data_path : str
        output directory for mapped cell tracings
    scaling : np.array
        scaling of dataset, s.t. after applying coordinates are isotropic and
        in nm
    mem : KnossosDataset
        will be assigned automatically
    mem_offset :
        optional offeset of dataset
    mitos/vc/sj : UltrastructuralDataset
        Dataset which contains cell objects of mitochondria, vesicle clouds and
        synaptic junctions respectively
    """
    def __init__(self, wd, scaling=(9., 9., 20.)):
        """
        Parameters
        ----------
        wd : str
            path to working directory
        scaling : tuple of ints
            scaling of data set, s.t. data is isotropic
        """
        self.wd = wd
        vc_source = wd + '/chunkdataset/obj_vc_ARGUS_5/'
        sj_source = wd + '/chunkdataset/obj_sj_ARGUS_3/'
        mito_source = wd + '/chunkdataset/obj_mi_ARGUS_8/'
        self.mem_path = wd + '/knossosdatasets/rrbarrier/'
        self.cs_path = wd + '/chunkdataset/j0126_watershed_map/'
        self.myelin_ds_path = wd + "/knossosdatasets/myelin/"
        self.data_path = wd + '/neurons/'
        self.skeleton_path = wd + '/tracings/'
        self.nb_cpus = np.min((cpu_count(), 2))
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.scaling = arr(scaling)
        self.mem = None
        self.mem_offset = arr([0, 0, 0])
        object_dict = {0: "mitos", 1: "vc", 2: "sj"}
        objects = [None, None, None]
        for i, source in enumerate([mito_source, vc_source, sj_source]):
            if type(source) is str:
                try:
                    obj = load_dataset(source)
                    obj.init_properties()
                    # print "Initialized %s objects." % object_dict[i]
                except IOError:
                    obj = None
            else:
                obj = source
            objects[i] = obj
        self.mitos = objects[0]
        self.vc = objects[1]
        self.sj = objects[2]


def load_ordered_mapped_skeleton(path):
    """Load nml of mapped skeleton and order trees

    Parameters
    ----------
    path : str
        Path to nml

    Returns
    -------
    list
        List of trees with order [skeleton, mitos, vc, sj, soma]
    """
    anno_dict = {"skeleton": 0, "mitos": 1, "vc": 2, "sj": 3, "soma": 4}
    annotation = su.loadj0126NML(path)
    ordered_annotation = list([[], [], [], [], []])
    for name in anno_dict.keys():
        init_anno = SkeletonAnnotation()
        init_anno.scaling = [9, 9, 20]
        init_anno.appendComment(name)
        ordered_annotation[anno_dict[name]] = init_anno
    if len(annotation) == 1 and not np.any([key in annotation[0].getComment()
                                            for key in anno_dict.keys()]):
        ordered_annotation[0] = annotation[0]
    else:
        for anno in annotation:
            c = anno.getComment()
            if c == '':
                continue
            if ('vc' in c) or ('p4' in c):
                c = 'vc'
            elif ('sj' in c) or ('az' in c):
                c = 'sj'
            elif 'mitos' in c:
                c = 'mitos'
            elif 'soma' in c:
                c = 'soma'
            elif 'skeleton' in c:
                c = 'skeleton'
            else:
                print "Using %s as skeleton tree." % c
                c = 'skeleton'
            try:
                ordered_annotation[anno_dict[c]] = anno
            except KeyError:
                print "Found strange tree %s in skeleton at %s." % (c, path)
    return ordered_annotation


def load_files_from_kzip(path, load_mitos):
    """Load all available files from annotation kzip

    Parameters
    ----------
    path : str
        path to kzip
    load_mitos : bool
        load mito hull points and ids

    Returns
    -------
    list of np.array
        coordinates [hull, mitos, vc, sj], id lists, normals
    """
    coord_list = [np.zeros((0, 3)), np.zeros((0, 3)),
                  np.zeros((0, 3)), np.zeros((0, 3))]
    hull_normals = [np.zeros((0, 3)), np.zeros((0, 3)),
                    np.zeros((0, 3)), np.zeros((0, 3))]
    # id_list of cell objects
    id_list = [np.zeros((0, )), np.zeros((0, )), np.zeros((0, ))]
    zf = zipfile.ZipFile(path, 'r')
    for i, filename in enumerate(['hull_points.xyz', 'mitos.txt', 'vc.txt',
                                  'sj.txt']):
        try:
            if i == 1 and load_mitos is not True:
                continue
            data = np.fromstring(zf.read(filename), sep=' ')
            if len(data) == 0:
                continue
            if np.isscalar(data[0]) and data[0] == -1:
                continue
            if i == 0:
                hull_normals = data.reshape(data.shape[0]/6, 6)[:, 3:]
                data = data.reshape(data.shape[0]/6, 6)[:, :3]
            else:
                data = data.reshape(data.shape[0]/3, 3)
            coord_list[i] = data.astype(np.uint32)
        except (IOError, ImportError, KeyError):
            pass
    for i, filename in enumerate(['mitos_id.txt', 'vc_id.txt', 'sj_id.txt']):
        try:
            if i == 0 and load_mitos is not True:
                continue
            data = np.fromstring(zf.read(filename), sep=' ')
            if len(data) == 0:
                continue
            if data[0] == -1:
                continue
            id_list[i] = data.astype(np.uint32)
        except (IOError, ImportError, KeyError):
            pass
    zf.close()
    return coord_list, id_list, hull_normals


def load_objpkl_from_kzip(path):
    """Loads object pickle file from mapped skeleton kzip

    Parameters
    ----------
    path : str
        Path to kzip

    Returns
    -------
    list of SegmentationObjectDataset
    """
    zf = zipfile.ZipFile(path, 'r')
    object_datasets = []
    for ix, filename in enumerate(['mitos.pkl', 'vc.pkl', 'sj.pkl']):
        object_datasets.append(UltrastructuralDataset('', '', ''))
        try:
            temp = tempfile.TemporaryFile()
            temp.write(zf.read(filename))
            temp.seek(0)
            obj_ds = pickle.load(temp)
            object_datasets[ix] = obj_ds
        except (IOError, ImportError, KeyError):
            pass
    return object_datasets


def load_anno_list(fpaths, load_mitos=True, append_obj=True):
    """Load nml's given in nml_list and append according hull information from
    xyz-file, as well as object coordinates from txt files in kzip (if given).

    Parameters
    ----------
    fpaths : list of str
        paths to tracings
    load_mitos: bool
        load mito hull points and ids
    append_obj : bool
        load mapped cell objects

    Returns
    -------
    list of list of SkeletonAnnotations
        list of mapped cell tracings
    """
    anno_list = []
    cnt = 0
    for path in fpaths:
        cnt += 1
        mapped_skel = load_mapped_skeleton(path, append_obj, load_mitos)
        if mapped_skel is None:
            continue
        anno_list.append(mapped_skel)
    return anno_list


def load_mapped_skeleton(path, append_obj, load_mitos):
    """Load mapped cell tracing and append mapped cell objects

    Parameters
    ----------
    path : str
        path to tracing kzip
    append_obj : bool
        append cell objects
    load_mitos : bool
        load mitochondria objects

    Returns
    -------
    list of SkeletonAnnotations
        mapped cell tracings (skel, mito, vc, sj, soma)
    """
    mapped_skel = load_ordered_mapped_skeleton(path)
    if append_obj:
        skel = mapped_skel[0]
        obj_dicts = load_objpkl_from_kzip(path)
        skel.mitos = obj_dicts[0]
        skel.vc = obj_dicts[1]
        skel.sj = obj_dicts[2]
        if 'k.zip' in os.path.basename(path):
            path = path[:-5]
        if 'nml' in os.path.basename(path):
            path = path[:-3]
        coord_list, id_list, hull_normals = \
            load_files_from_kzip(path + 'k.zip', load_mitos)
        mito_hull_ids, vc_hull_ids, sj_hull_ids = id_list
        hull, mitos, vc, sj = coord_list
        skel.hull_coords = hull
        skel.mito_hull_coords = mitos
        skel.mito_hull_ids = mito_hull_ids
        skel.vc_hull_coords = vc
        skel.vc_hull_ids = vc_hull_ids
        skel.sj_hull_coords = sj
        skel.sj_hull_ids = sj_hull_ids
        skel.hull_normals = hull_normals
    return mapped_skel


def get_filepaths_from_dir(directory, ending='k.zip', recursively=False):
    """Collect all files with certain ending from directory.

    Parameters
    ----------
    directory: str
        path to lookup directory
    ending: str
        ending of files
    recursively: boolean
        add files from subdirectories

    Returns
    -------
    list of str
        paths to files
    """
    if recursively:
        files = [os.path.join(r, f) for r,s ,fs in
                 os.walk(directory) for f in fs if ending in f[-len(ending):]]
    else:
        files = [os.path.join(directory, f) for f in next(os.walk(directory))[2]
                 if ending in f[-len(ending):]]
    return files


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
            pass
    return wanted_paths


def supp_fname_from_fpath(fpath):
    """Returns supported filename from path to written file to write it to
    kzip

    Parameters
    ----------
    fpath : str
        path to file

    Returns
    -------
    str
        filename if supported, else ValueError
    """
    file_name = os.path.basename(fpath)
    if '.nml' in file_name or '.xml' in file_name:
        file_name = 'annotation.xml'
    elif '.xyz' in file_name:
        file_name = 'hull_points.xyz'
    elif 'mitos.txt' in file_name:
        file_name = 'mitos.txt'
    elif 'vc.txt' in file_name:
        file_name = 'vc.txt'
    elif 'sj.txt' in file_name:
        file_name = 'sj.txt'
    elif 'mitos_id.txt' in file_name:
        file_name = 'mitos_id.txt'
    elif 'vc_id.txt' in file_name:
        file_name = 'vc_id.txt'
    elif 'sj_id.txt' in file_name:
        file_name = 'sj_id.txt'
    elif 'mitos.pkl' in file_name:
        file_name = 'mitos.pkl'
    elif 'vc.pkl' in file_name:
        file_name = 'vc.pkl'
    elif 'sj.pkl' in file_name:
        file_name = 'sj.pkl'
    elif 'axoness_feat.csv' in file_name:
        file_name = 'axoness_feat.csv'
    elif 'spiness_feat.csv' in file_name:
        file_name = 'spiness_feat.csv'
    elif 'mergelist.txt' in file_name:
        file_name = 'mergelist.txt'
    else:
        raise ValueError
    return file_name


def write_data2kzip(kzip_path, fpath, force_overwrite=False):
    """Write supported files to kzip of mapped annotation

    Parameters
    ----------
    kzip_path : str
    fpath : str
    force_overwrite : bool
    """
    file_name = supp_fname_from_fpath(fpath)
    if os.path.isfile(kzip_path):
        try:
            if force_overwrite:
                with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(fpath, file_name)
            else:
                remove_from_zip(kzip_path, file_name)
                with zipfile.ZipFile(kzip_path, "a", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(fpath, file_name)
        except Exception, e:
            print "Couldn't open file %s for reading and" \
                  " overwriting." % kzip_path, e
    else:
        try:
            with zipfile.ZipFile(kzip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(fpath, file_name)
        except Exception, e:
            print "Couldn't open file %s for writing." % kzip_path, e
    os.remove(fpath)


def remove_from_zip(zipfname, *filenames):
    """Removes filenames from zipfile

    Parameters
    ----------
    zipfname : str
        Path to zipfile
    filenames : list of str
        files to delete
    """
    tempdir = tempfile.mkdtemp()
    try:
        tempname = os.path.join(tempdir, 'new.zip')
        with zipfile.ZipFile(zipfname, 'r') as zipread:
            with zipfile.ZipFile(tempname, 'w') as zipwrite:
                for item in zipread.infolist():
                    if item.filename not in filenames:
                        data = zipread.read(item.filename)
                        zipwrite.writestr(item, data)
        shutil.move(tempname, zipfname)
    finally:
        shutil.rmtree(tempdir)


def get_skelID_from_path(skel_path):
    """Parse skeleton id from filename

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


def connect_soma_tracing(soma):
    """Connect tracings of soma, s.t. a lot of rays are emitted. Connects nearby
     (kd-tree) nodes of soma tracings with edges

    Parameters
    ----------
    soma: SkeletonAnnotation
        Soma tracing

    Returns
    -------
    SkeletonAnnotation
        Sparsely connected soma tracing
    """
    node_list = np.array([node for node in soma.getNodes()])
    coords = np.array([node.getCoordinate_scaled() for node in node_list])
    if len(coords) == 0:
        return soma
    tree = spatial.cKDTree(coords)
    near_nodes_list = tree.query_ball_point(coords, 4000)
    for ii, node in enumerate(node_list):
        near_nodes_ids = near_nodes_list[ii]
        near_nodes = node_list[near_nodes_ids]
        for nn in near_nodes:
            soma.addEdge(node, nn)
    return soma


def cell_object_id_parser(obj_trees):
    """Extracts unique object ids from object tree list for cell objects
    'mitos', 'vc' and 'sj'

    Parameters
    ----------
    obj_trees : list of annotation objects ['mitos', 'vc', 'sj']

    Returns
    -------
    dict
        keys 'mitos', 'vc' and 'sj' containing unique object ID's as values
    """
    mito_ids = []
    vc_ids = []
    sj_ids = []
    for node in obj_trees[0].getNodes():
        comment = node.getComment()
        match = re.search('%s-([^,]+)' % 'mitos', comment)
        mito_ids.append(int(match.group(1)))
    for node in obj_trees[1].getNodes():
        comment = node.getComment()
        match = re.search('%s-([^,]+)' % 'vc', comment)
        vc_ids.append(int(match.group(1)))
    for node in obj_trees[2].getNodes():
        comment = node.getComment()
        match = re.search('%s-([^,]+)' % 'sj', comment)
        sj_ids.append(int(match.group(1)))
    obj_id_dict = {}
    obj_id_dict['mitos'] = mito_ids
    obj_id_dict['vc'] = vc_ids
    obj_id_dict['sj'] = sj_ids
    return obj_id_dict


def write_obj2pkl(objects, path):
    """Writes object to pickle file

    Parameters
    ----------
    objects : UltrastructuralDatasetObject
    path : str
        destianation
    """
    with open(path, 'wb') as output:
        pickle.dump(objects, output, -1)


def load_pkl2obj(path):
    """Loads pickle file of object

    Parameters
    ----------
    path: str
        path of source file

    Returns
    -------
    UltrastructuralDatasetObject
    """
    with open(path, 'rb') as inp:
        objects = pickle.load(inp)
    return objects


def helper_get_voxels(obj):
    """Helper function to receive object voxel

    Parameters
    ----------
    obj : SegmentationObject

    Returns
    -------
    np.array
        object voxels
    """
    try:
        voxels = obj.voxels
    except KeyError:
        return np.array([])
    return voxels


def helper_get_hull_voxels(obj):
    """Helper function to receive object hull voxels.
    Parameters
    ----------
    obj : SegmentationObject

    Returns
    -------
    np.array
        object hull voxels
    """

    return obj.hull_voxels


def hull2text(hull_coords, normals, path):
    """Writes hull coordinates and normals to xyz file. Each line corresponds to
    coordinates and normal vector of one point x y z n_x n_y n_z.

    Parameters
    ----------
    hull_coords : np.array
    normals : np.array
    path : str
    """
    # add ray-end-points to nml and to txt file (incl. normals)
    f = open(path, 'wb')
    for i in range(hull_coords.shape[0]):
        end_point = hull_coords[i]
        normal = normals[i]
        f.write("%d %d %d %0.4f %0.4f %0.4f\n" % (end_point[0], end_point[1],
                                                  end_point[2], normal[0],
                                                  normal[1], normal[2]))
    f.close()


def obj_hull2text(id_list, hull_coords_list, path):
    """Writes object hull coordinates and corresponding object ids to txt file.
    Each line corresponds to id and coordinate vector of one object: id x y z

    Parameters
    ----------
    id_list : np.array
    hull_coords_list : np.array
    path : str
    """
    # add ray-end-points to nml and to txt file (incl. normals)
    f = open(path, 'wb')
    for i in range(len(hull_coords_list)):
        coord = hull_coords_list[i]
        f.write("%d %d %d\n" % (coord[0], coord[1], coord[2]))
    f.close()
    if id_list is []:
        return
    f = open(path[:-4]+'_id.txt', 'wb')
    for i in range(len(hull_coords_list)):
        ix = id_list[i]
        f.write("%d\n" % ix)
    f.close()


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
    """
    Switches to array entries

    Parameters
    ----------
    this_array: np.array
    entries: list of int

    Returns
    -------
    this_array: np.array
    """
    entry_0 = this_array[entries[0]]
    this_array[entries[0]] = this_array[entries[1]]
    this_array[entries[1]] = entry_0
    return this_array


def cut_array_in_one_dim(array, start, end, dim):
    """
    Cuts an array along a dimension

    Parameters
    ----------
    array: np.array
    start: int
    end: int
    dim: int

    Returns
    -------
    array: np.array

    """
    start = int(start)
    end = int(end)
    if dim == 0:
        array = array[start: end, :, :]
    elif dim == 1:
        array = array[:, start: end, :]
    elif dim == 2:
        array = array[:, :, start:end]
    else:
        raise NotImplementedError()

    return array
