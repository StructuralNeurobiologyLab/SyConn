import os
from multiprocessing import cpu_count

from syconn.utils import annotationUtils as au

try:
    import ChunkUtils as cu
except:
    import Sven.functional.ChunkUtils as cu
from basics import *
import zipfile
import numpy as np
from syconn.utils.newskeleton import SkeletonAnnotation
import cPickle as pickle
import tempfile
from numpy import array as arr
import shutil
from scipy import spatial
__author__ = 'pschuber'


class DataHandler(object):
    """Initialized with paths or cell components (SegmentationObjects), path to
    membrane prediction and source path of traced skeletons (to be computed).
    DataHandler is needed for further processing.
    :param datapath: Used as output path for all computations
    """
    def __init__(self, wd, scaling=(9., 9., 20.)):

        p4_source = wd + '/obj_p4/'
        az_source = wd + '/obj_az'
        mito_source = wd + '/obj_mito/',
        skeleton_source = wd + '/tracings/',
        mempath = "/lustre/sdorkenw/j0126_3d_rrbarrier/", #TODO
        datapath = wd + '/neurons/'
        self.nb_cpus = cpu_count()
        self.data_path = datapath
        self.skeleton_path = skeleton_source
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        self.mem_path = mempath
        self.scaling = arr(scaling)
        self.skeletons = {}
        self.mem = None
        self.mem_offset = arr([0, 0, 0])
        object_dict = {0: "mitos", 1: "p4", 2: "az"}
        objects = [None, None, None]
        for i, source in enumerate([mito_source, p4_source, az_source]):
            if type(source) is str:
                obj = cu.load_dataset(source)
                obj.init_properties()
                print "Initialized %s objects." % object_dict[i]
            else:
                obj = source
            objects[i] = obj
        self.mitos = objects[0]
        self.p4 = objects[1]
        self.az = objects[2]


def load_ordered_mapped_skeleton(path):
    """
    Load nml of mapped skeleton and order trees.
    :param path: Path to nml
    :type path: list
    :return: List of trees with order [skeleton, mitos, p4, az, soma]
    """
    anno_dict = {"skeleton": 0, "mitos": 1, "p4": 2, "az": 3, "soma": 4}
    annotation = au.loadj0126NML(path)
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
            if 'p4' in c:
                c = 'p4'
            elif 'az' in c:
                c = 'az'
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
    """
    Load all available files from annotation kzip.
    :param path: string path to kzip
    :param load_mitos: bool Load mito hull points and ids
    :return: list of coordinates [hull, mitos, p4, az],
    """
    coord_list = [np.zeros((0, 3)), np.zeros((0, 3)),
                  np.zeros((0, 3)), np.zeros((0, 3))]
    hull_normals = [np.zeros((0, 3)), np.zeros((0, 3)),
                  np.zeros((0, 3)), np.zeros((0, 3))]
    # id_list of cell objects
    id_list = [np.zeros((0, )), np.zeros((0, )), np.zeros((0, ))]
    zf = zipfile.ZipFile(path, 'r')
    for i, filename in enumerate(['hull_points.xyz', 'mitos.txt', 'p4.txt',
                                  'az.txt']):
        if i == 1 and load_mitos != True:
            continue
        data = np.fromstring(zf.read(filename), sep=' ')
        if np.isscalar(data[0]) and data[0] == -1:
            continue
        if i == 0:
            hull_normals = data.reshape(data.shape[0]/6, 6)[:, 3:]
            data = data.reshape(data.shape[0]/6, 6)[:, :3]
        else:
            data = data.reshape(data.shape[0]/3, 3)
        coord_list[i] = data.astype(np.uint32)
    for i, filename in enumerate(['mitos_id.txt', 'p4_id.txt', 'az_id.txt']):
        if i == 0 and load_mitos is not True:
            continue
        data = np.fromstring(zf.read(filename), sep=' ')
        if data[0] == -1:
            continue
        id_list[i] = data.astype(np.uint32)
    zf.close()
    return coord_list, id_list, hull_normals


def load_objpkl_from_kzip(path):
    """
    Loads object pickle file from mapped skeleton kzip.
    :param path: str Path to kzip
    :return: list of SegmentationObjectDataset
    """
    zf = zipfile.ZipFile(path, 'r')
    object_datasets = []
    for filename in ['mitos.pkl', 'p4.pkl', 'az.pkl']:
        temp = tempfile.TemporaryFile()
        temp.write(zf.read(filename))
        temp.seek(0)
        obj_ds = pickle.load(temp)
        object_datasets.append(obj_ds)
    return object_datasets


def load_anno_list(nml_list, load_mitos=True, append_obj=True):
    """
    Load nml's given in nml_list and append according hull information from
    xyz-file, as well as object coordinates from txt files in kzip (if given).
    :param load_mitos: bool Load mito hull points and ids
    :return: List of tree lists.
    """
    anno_list = []
    cnt = 0
    for path in nml_list:
        cnt += 1
        mapped_skel = load_mapped_skeleton(path, append_obj, load_mitos)
        if mapped_skel is None:
            continue
        anno_list.append(mapped_skel)
    return anno_list


def load_mapped_skeleton(path, append_obj, load_mitos):
    """

    :param path:
    :param append_obj:
    :param load_mitos:
    :return:
    """
    try:
        mapped_skel = load_ordered_mapped_skeleton(path)
    except IOError as e:
        print "Skipped", path
        print e
        return
    if append_obj:
        skel = mapped_skel[0]
        if 'k.zip' in os.path.basename(path):
            path = path[:-5]
        if 'nml' in os.path.basename(path):
            path = path[:-3]
        coord_list, id_list, hull_normals = \
            load_files_from_kzip(path + 'k.zip', load_mitos)
        mito_hull_ids, p4_hull_ids, az_hull_ids = id_list
        hull, mitos, p4, az = coord_list
        skel._hull_coords = hull
        skel.mito_hull_coords = mitos
        skel.mito_hull_ids = mito_hull_ids
        skel.p4_hull_coords = p4
        skel.p4_hull_ids = p4_hull_ids
        skel.az_hull_coords = az
        skel.az_hull_ids = az_hull_ids
        skel._hull_normals = hull_normals
    return mapped_skel


def get_filepaths_from_dir(dir, ending='k.zip'):
    """
    Collect all files with certain ending from directory.
    :param dir: str Path to lookup directory
    :param ending: str Ending of files
    :return: list of paths to files
    """
    files = [os.path.join(dir, f) for f in next(os.walk(dir))[2] if ending in f]
    return files


def get_paths_of_skelID(id_list, traced_skel_dir):
    """
    Gather paths to kzip of skeletons with ID in id_list
    :param id_list: list of str of skeleton ID's
    :param traced_skel_dir: dir of mapped skeletons
    :return: list of paths of skeletons in id_list
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
    """
    Returns supported filename from path to written file to write it to
    kzip.
    :param fpath: path to file
    :return: filename if supported, else ValueError
    """
    file_name = os.path.basename(fpath)
    if '.nml' in file_name or '.xml' in file_name:
        file_name = 'annotation.xml'
    elif '.xyz' in file_name:
        file_name = 'hull_points.xyz'
    elif 'mitos.txt' in file_name:
        file_name = 'mitos.txt'
    elif 'p4.txt' in file_name:
        file_name = 'p4.txt'
    elif 'az.txt' in file_name:
        file_name = 'az.txt'
    elif 'mitos_id.txt' in file_name:
        file_name = 'mitos_id.txt'
    elif 'p4_id.txt' in file_name:
        file_name = 'p4_id.txt'
    elif 'az_id.txt' in file_name:
        file_name = 'az_id.txt'
    elif 'mitos.pkl' in file_name:
        file_name = 'mitos.pkl'
    elif 'p4.pkl' in file_name:
        file_name = 'p4.pkl'
    elif 'az.pkl' in file_name:
        file_name = 'az.pkl'
    elif 'axoness_feat.csv' in file_name:
        file_name = 'axoness_feat.csv'
    elif 'spiness_feat.csv' in file_name:
        file_name = 'spiness_feat.csv'
    else:
        raise(ValueError)
    return file_name


def write_data2kzip(kzip_path, fpath, force_overwrite=False):
    """
    Write supported files to kzip of mapped annotation
    :param kzip_path:
    :param fpath:
    :param force_overwrite:
    :return:
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
    """
    Removes filenames from zipfile.
    :param zipfname: str Path to zipfile
    :param filenames: list of str Files to delete
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
    """
    :param skel_path: path to skeleton
    :return: int skeleton ID
    """
    return int(re.findall('iter_0_(\d+)', skel_path)[0])


def connect_soma_tracing(soma):
    """
    :param soma: Soma tracing
    :type soma: AnnotationObject
    :return: Sparsely connected soma tracing
    Connect nearby (kd-tree) nodes of soma tracings with edges
    """
    node_list = np.array([node for node in soma.getNodes()])
    coords = np.array([node.getCoordinate_scaled() for node in node_list])
    if len(coords) == 0:
        return soma
    print "Connecting nearby soma nodes with kd-tree (radius=4000)."
    tree = spatial.cKDTree(coords)
    near_nodes_list = tree.query_ball_point(coords, 4000)
    for ii, node in enumerate(node_list):
        near_nodes_ids = near_nodes_list[ii]
        near_nodes = node_list[near_nodes_ids]
        for nn in near_nodes:
            soma.addEdge(node, nn)
    return soma


def cell_object_id_parser(obj_trees):
    """
    Extracts unique object ids from object tree list for cell objects
    'mitos', 'p4' and 'az'.
    :param obj_trees: list of annotation objects ['mitos', 'p4', 'az']
    :return: dictionary with keys 'mitos', 'p4' and 'az' containing unique ID's
    as values
    """
    mito_ids = []
    p4_ids = []
    az_ids = []
    for node in obj_trees[0].getNodes():
        comment = node.getComment()
        match = re.search('%s-([^,]+)' % 'mitos', comment)
        mito_ids.append(int(match.group(1)))
    for node in obj_trees[1].getNodes():
        comment = node.getComment()
        match = re.search('%s-([^,]+)' % 'p4', comment)
        p4_ids.append(int(match.group(1)))
    for node in obj_trees[2].getNodes():
        comment = node.getComment()
        match = re.search('%s-([^,]+)' % 'az', comment)
        az_ids.append(int(match.group(1)))
    print "Found %d mitos, %d az and %d p4." % (len(mito_ids), len(p4_ids),
                                                len(az_ids))
    obj_id_dict = {}
    obj_id_dict['mitos'] = mito_ids
    obj_id_dict['p4'] = p4_ids
    obj_id_dict['az'] = az_ids
    return obj_id_dict


def write_obj2pkl(objects, path):
    """
    Writes SegmentationObjectDataset to pickle file.
    :param objects: SOD
    :param path: str to destianation
    """
    with open(path, 'wb') as output:
       pickle.dump(objects, output, -1)


def load_pkl2obj(path):
    """
    Loads pickle file of SegmentationObjectDataset
    :param path: str to source file.
    :return: SegmentatioObjectDataset
    """
    with open(path, 'rb') as input:
        objects = pickle.load(input)
    return objects


def helper_get_voxels(obj):
    """
    Helper function to receive object voxels.
    :param obj: SegmentationObject
    :return: array voxels
    """
    try:
        voxels = obj.voxels
    except KeyError:
        return np.array([])
    return voxels


def helper_get_hull_voxels(obj):
    """
    Helper function to receive object hull voxels.
    :param obj: SegmentationObject
    :return: array hull voxels
    """

    return obj.hull_voxels


def hull2text(hull_coords, normals, path):
    """
    Writes hull coordinates and normals to xyz file. Each line corresponds to
    coordinates and normal vector of one point x y z n_x n_y n_z.
    :param hull_coords: array
    :param normals: array
    :param path: str
    """
    print "Writing hull to .xyz file.", path
    # add ray-end-points to nml and to txt file (incl. normals)
    file = open(path, 'wb')
    for i in range(hull_coords.shape[0]):
        end_point = hull_coords[i]
        normal = normals[i]
        file.write("%d %d %d %0.4f %0.4f %0.4f\n" %
                   (end_point[0], end_point[1], end_point[2], normal[0],
                    normal[1], normal[2]))
    file.close()


def obj_hull2text(id_list, hull_coords_list, path):
    """
    Writes object hull coordinates and corresponding object ids to txt file.
    Each line corresponds to id and coordinate vector of one object:
     id x y z
    :param id_list: array
    :param hull_coords_list: array
    :param path: str
    """
    print "Writing object hull to .txt file.", path
    # add ray-end-points to nml and to txt file (incl. normals)
    file = open(path, 'wb')
    for i in range(len(hull_coords_list)):
        coord = hull_coords_list[i]
        file.write("%d %d %d\n" % (coord[0], coord[1], coord[2]))
    file.close()
    if id_list == []:
        return
    file = open(path[:-4]+'_id.txt', 'wb')
    for i in range(len(hull_coords_list)):
        id = id_list[i]
        file.write("%d\n" % id)
    file.close()
