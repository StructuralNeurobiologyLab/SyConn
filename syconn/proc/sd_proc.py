import cPickle as pkl
import glob
import numpy as np
import os
from collections import defaultdict
from .image import single_conn_comp_img
from knossos_utils import knossosdataset
from ..mp import qsub_utils as qu
from ..mp import shared_mem as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../QSUB_scripts/")
from ..handler.compression import VoxelDict, AttributeDict
from ..reps import segmentation
from ..handler import basics


def dataset_analysis(sd, recompute=True, stride=100, qsub_pe=None,
                     qsub_queue=None, nb_cpus=1, n_max_co_processes=100):
    """ Analyses the whole dataset and extracts and caches key information

    :param sd: SegmentationDataset
    :param recompute: bool
        whether or not to (re-)compute key information of each object
        (rep_coord, bounding_box, size)
    :param stride: int
        number of voxel / attribute dicts per thread
    :param qsub_pe: str
        qsub parallel environment
    :param qsub_queue: str
        qsub queue
    :param nb_cpus: int
        number of cores used for multithreading
        number of cores per worker for qsub jobs
    :param n_max_co_processes: int
        max number of workers running at the same time when using qsub
    """

    paths = sd.so_dir_paths

    # Partitioning the work

    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, sd.type, sd.version,
                             sd.working_dir, recompute])

    # Running workers

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_dataset_analysis_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "dataset_analysis",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file) as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")

    # Creating summaries
    # This is a potential bottleneck for very large datasets

    attr_dict = {}
    for this_attr_dict in results:
        for attribute in this_attr_dict:
            if not attribute in attr_dict:
                attr_dict[attribute] = []

            attr_dict[attribute] += this_attr_dict[attribute]

    for attribute in attr_dict:
        np.save(sd.path + "/%ss.npy" % attribute, attr_dict[attribute])


def _dataset_analysis_thread(args):
    """ Worker of dataset_analysis """

    paths = args[0]
    obj_type = args[1]
    version = args[2]
    working_dir = args[3]
    recompute = args[4]

    global_attr_dict = dict(id=[], size=[], bounding_box=[], rep_coord=[])

    for p in paths:
        print(p)
        if not len(os.listdir(p)) > 0:
            os.rmdir(p)
        else:
            this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                         read_only=not recompute, timeout=3600)
            if recompute:
                this_vx_dc = VoxelDict(p + "/voxel.pkl",
                                       read_only=True, timeout=3600)
                so_ids = this_vx_dc.keys()
            else:
                so_ids = this_attr_dc.keys()

            print(so_ids)

            for so_id in so_ids:
                global_attr_dict["id"].append(so_id)
                so = segmentation.SegmentationObject(so_id, obj_type,
                                                     version, working_dir)

                so.attr_dict = this_attr_dc[so_id]

                if recompute:
                    so.load_voxels(voxel_dc=this_vx_dc)
                    so.calculate_rep_coord(voxel_dc=this_vx_dc)

                if recompute:
                    so.attr_dict["rep_coord"] = so.rep_coord
                if recompute:
                    so.attr_dict["bounding_box"] = so.bounding_box
                if recompute:
                    so.attr_dict["size"] = so.size

                for attribute in so.attr_dict.keys():
                    if attribute not in global_attr_dict:
                        global_attr_dict[attribute] = []

                    global_attr_dict[attribute].append(so.attr_dict[attribute])

                this_attr_dc[so_id] = so.attr_dict

            if recompute:
                this_attr_dc.save2pkl()

    return global_attr_dict


def map_objects_to_sv_multiple(sd, obj_types, kd_path, readonly=False, 
                               stride=50, qsub_pe=None, qsub_queue=None,
                               nb_cpus=1, n_max_co_processes=None):
    assert isinstance(obj_types, list)
    
    for obj_type in obj_types:
        map_objects_to_sv(sd, obj_type, kd_path, readonly=readonly, stride=stride,
                          qsub_pe=qsub_pe, qsub_queue=qsub_queue, nb_cpus=nb_cpus,
                          n_max_co_processes=n_max_co_processes)
        

def map_objects_to_sv(sd, obj_type, kd_path, readonly=False, stride=1000,
                      qsub_pe=None, qsub_queue=None, nb_cpus=1,
                      n_max_co_processes=None):
    """ Maps objects to SVs

    The segmentation needs to be written to a KnossosDataset before running this

    :param sd: SegmentationDataset
    :param obj_type: str
    :param kd_path: str
        path to knossos dataset containing the segmentation
    :param readonly: bool
        if True the mapping is only read from the segmentation objects and not
        computed. This requires the previous computation of the mapping for the
        mapped segmentation objects.
    :param stride: int
        number of voxel / attribute dicts per thread
    :param qsub_pe: str
        qsub parallel environment
    :param qsub_queue: str
        qsub queue
    :param nb_cpus: int
        number of cores used for multithreading
        number of cores per worker for qsub jobs
    :param n_max_co_processes: int
        max number of workers running at the same time when using qsub
    :return:
    """
    assert sd.type == "sv"
    assert obj_type in sd.version_dict

    seg_dataset = sd.get_segmentationdataset(obj_type)
    paths = seg_dataset.so_dir_paths

    # Partitioning the work

    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, obj_type,
                             sd.version_dict[obj_type], sd.working_dir,
                             kd_path, readonly])

    # Running workers - Extracting mapping

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_map_objects_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "map_objects",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file) as f:
                results.append(pkl.load(f))

    else:
        raise Exception("QSUB not available")

    sv_obj_map_dict = defaultdict(dict)
    for result in results:
        for sv_key, value in result.iteritems():
            sv_obj_map_dict[sv_key].update(value)

    mapping_dict_path = seg_dataset.path + "/sv_%s_mapping_dict.pkl" % sd.version
    with open(mapping_dict_path, "w") as f:
        pkl.dump(sv_obj_map_dict, f)

    paths = sd.so_dir_paths

    # Partitioning the work

    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, obj_type, mapping_dict_path])

    # Running workers - Writing mapping to SVs

    if qsub_pe is None and qsub_queue is None:
        sm.start_multiprocess(_write_mapping_to_sv_thread, multi_params,
                              nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        qu.QSUB_script(multi_params, "write_mapping_to_sv", pe=qsub_pe,
                       queue=qsub_queue, script_folder=script_folder,
                       n_cores=nb_cpus, n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _map_objects_thread(args):
    """ Worker of map_objects_to_sv """

    paths = args[0]
    obj_type = args[1]
    obj_version = args[2]
    working_dir = args[3]
    kd_path = args[4]
    readonly = args[5]
    if len(args) > 6:
        datatype = args[6]
    else:
        datatype = np.uint64

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)

    seg_dataset = segmentation.SegmentationDataset(obj_type,
                                                   version=obj_version,
                                                   working_dir=working_dir)

    sv_id_dict = {}

    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                     read_only=readonly, timeout=3600)
        this_vx_dc = VoxelDict(p + "/voxel.pkl", read_only=True,
                               timeout=3600)

        for so_id in this_vx_dc.keys():
            so = seg_dataset.get_segmentation_object(so_id)
            so.attr_dict = this_attr_dc[so_id]
            so.load_voxels(voxel_dc=this_vx_dc)

            if readonly:
                if "mapping_ids" in so.attr_dict:
                    ids = so.attr_dict["mapping_ids"]
                    id_ratios = so.attr_dict["mapping_ratios"]

                    for i_id in range(len(ids)):
                        if ids[i_id] in sv_id_dict:
                            sv_id_dict[ids[i_id]][so_id] = id_ratios[i_id]
                        else:
                            sv_id_dict[ids[i_id]] = {so_id: id_ratios[i_id]}
            else:
                if np.product(so.shape) > 1e12:
                    continue

                vx_list = np.argwhere(so.voxels) + so.bounding_box[0]
                try:
                    id_list = kd.from_overlaycubes_to_list(vx_list,
                                                           datatype=datatype)
                except:
                    continue

                ids, id_counts = np.unique(id_list, return_counts=True)
                id_ratios = id_counts / float(np.sum(id_counts))

                for i_id in range(len(ids)):
                    if ids[i_id] in sv_id_dict:
                        sv_id_dict[ids[i_id]][so_id] = id_ratios[i_id]
                    else:
                        sv_id_dict[ids[i_id]] = {so_id: id_ratios[i_id]}

                so.attr_dict["mapping_ids"] = ids
                so.attr_dict["mapping_ratios"] = id_ratios
                this_attr_dc[so_id] = so.attr_dict

        if not readonly:
            this_attr_dc.save2pkl()

    return sv_id_dict


def _write_mapping_to_sv_thread(args):
    """ Worker of map_objects_to_sv """

    paths = args[0]
    obj_type = args[1]
    mapping_dict_path = args[2]

    with open(mapping_dict_path, "r") as f:
        mapping_dict = pkl.load(f)

    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                     read_only=False, timeout=3600)

        for sv_id in this_attr_dc.keys():
            this_attr_dc[sv_id]["mapping_%s_ids" % obj_type] = \
                mapping_dict[sv_id].keys()
            this_attr_dc[sv_id]["mapping_%s_ratios" % obj_type] = \
                mapping_dict[sv_id].values()

        this_attr_dc.save2pkl()


# def split_cs(sd, new_obj_type, dist=None, stride=5000,
#             qsub_pe=None, qsub_queue=None, nb_cpus=None):
#     """ Splits contact sites
#
#     :param sd:
#     :param new_obj_type:
#     :param dist:
#     :param stride:
#     :param qsub_pe:
#     :param qsub_queue:
#     :param nb_cpus:
#     :return:
#     """
#     if dist is None:
#         dist = (7**2 * 2 + 3**2)**(1./2)
#
#     new_sd = segmentation.SegmentationDataset(obj_type=new_obj_type,
#                                               version="new",
#                                               working_dir=sd.working_dir,
#                                               version_dict=sd.version_dict)
#
#     # Partitioning the work
#
#     multi_params = []
#     start_id_indices = [i_id for i_id in range(0, len(sd.ids), stride)]
#     start_new_ids = [i * int(2e9 / len(start_id_indices)) for i in range(len(start_id_indices))]
#
#     for i_start_id in range(len(start_id_indices)):
#         multi_params.append([start_id_indices[i_start_id],
#                              stride, sd.type, new_obj_type,
#                              sd.version, new_sd.version,
#                              sd.version_dict, sd.working_dir,
#                              dist, start_new_ids[i_start_id]])
#
#     # Running workers
#
#     if qsub_pe is None and qsub_queue is None:
#         sm.start_multiprocess(_split_cs_thread, multi_params,
#                               nb_cpus=nb_cpus)
#
#     elif qu.__QSUB__:
#         path_to_out = qu.QSUB_script(multi_params,
#                                      "split_cs",
#                                      pe=qsub_pe, queue=qsub_queue,
#                                      script_folder=script_folder)
#
#         out_files = glob.glob(path_to_out + "/*")
#
#     else:
#         raise Exception("QSUB not available")
#
#
# def _split_cs_thread(args):
#     """ Worker of split_cs """
#     i_id = args[0]
#     stride = args[1]
#     obj_type = args[2]
#     new_obj_type = args[3]
#     version = args[4]
#     new_version = args[5]
#     version_dict = args[6]
#     working_dir = args[7]
#     dist = args[8]
#     new_id = args[9]
#
#     sd = segmentation.SegmentationDataset(obj_type=obj_type, version=version,
#                                           working_dir=working_dir,
#                                           version_dict=version_dict)
#     ids = sd.ids[i_id * stride: (i_id + 1) * stride]
#
#     new_sd = segmentation.SegmentationDataset(obj_type=new_obj_type,
#                                               version=new_version,
#                                               working_dir=working_dir,
#                                               version_dict=version_dict)
#
#     for i_id in range(len(ids)):
#         so_obj = sd.get_segmentationdataset(ids[i_id])
#         kdtree = scipy.spatial.cKDTree(so_obj.voxel_list)
#
#         graph = nx.from_edgelist(kdtree.query_pairs(dist))
#         ccs = list(nx.connected_components(graph))
#
#         if len(ccs) == 1:
#             new_so_obj = new_sd.get_segmentationdataset(new_id)
#             new_id += 1
#
#             new_so_obj.attr_dict["paths_to_voxels"] = so_obj.paths_to_voxels
#             new_so_obj.save_attr_dict()
#         else:
#             for cc in ccs:
#                 new_so_obj = new_sd.get_segmentationdataset(new_id)
#                 new_id += 1
#
#                 voxel_ids = np.array(list(cc), dtype=np.int32)
#                 this_voxel_list = so_obj.voxel_list[voxel_ids]
#
#                 bb = [np.min(this_voxel_list, axis=1),
#                       np.max(this_voxel_list, axis=1)]
#
#                 this_voxel_list -= bb[0]
#
#                 this_voxels = np.zeros(bb[1]-bb[0], dtype=np.bool)
#                 this_voxels[this_voxel_list[:, 0],
#                             this_voxel_list[:, 1],
#                             this_voxel_list[:, 2]] = True
#
#                 new_so_obj.save_voxels(this_voxels)


def init_sos(sos_dict):
    loc_dict = sos_dict.copy()
    svixs = loc_dict["svixs"]
    del loc_dict["svixs"]
    sos = [segmentation.SegmentationObject(ix, **loc_dict) for ix in svixs]
    return sos


def sos_dict_fact(svixs, version="0", scaling=(10, 10, 20), obj_type="sv",
                  working_dir="/wholebrain/scratch/areaxfs/", create=False):
    sos_dict = {"svixs": svixs, "version": version,
                "working_dir": working_dir, "scaling": scaling,
                "create": create, "obj_type": obj_type}
    return sos_dict


def predict_sos_views(model, sos, pred_key, nb_cpus=1, woglia=True,
                      verbose=False, raw_only=False):
    nb_chunks = np.max([1, len(sos) / 50])
    so_chs = basics.chunkify(sos, nb_chunks)
    for ch in so_chs:
        views = sm.start_multiprocess_obj("load_views", [[sv, {"woglia": woglia,
                                          "raw_only": raw_only}]
                                          for sv in ch], nb_cpus=nb_cpus)
        for kk in range(len(views)):
            data = views[kk]
            for i in range(len(data)):
                sing_cc = np.concatenate([
                                             single_conn_comp_img(data[i, 0, :1]),
                                             single_conn_comp_img(data[i, 0, 1:])])
                data[i, 0] = sing_cc
            views[kk] = data
        part_views = np.cumsum([0] + [len(v) for v in views])
        views = np.concatenate(views)
        probas = model.predict_proba(views, verbose=verbose)
        so_probas = []
        for ii, so in enumerate(ch):
            sv_probas = probas[part_views[ii]:part_views[ii + 1]]
            so_probas.append(sv_probas)
            # so.attr_dict[key] = sv_probas
        assert len(so_probas) == len(ch)
        params = [[so, prob, pred_key] for so, prob in zip(ch, so_probas)]
        sm.start_multiprocess(multi_probas_saver, params, nb_cpus=nb_cpus)


def multi_probas_saver(args):
    so, probas, key = args
    so.save_attributes([key], [probas])
