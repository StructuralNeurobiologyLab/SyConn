# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from itertools import combinations
from contactsites import write_summaries
from multi_proc.multi_proc_main import __QSUB__, start_multiprocess, QSUB_script
from processing.features import calc_prop_feat_dict
from processing.learning_rfc import write_feat2csv, load_rfcs
from processing.mapper import SkeletonMapper, prepare_syns_btw_annos, \
    similarity_check_star
from syconn.processing.cell_types import predict_celltype_label
from knossos_utils.skeleton import Skeleton
from utils.datahandler import *
from syconn.conmatrix import type_sorted_wiring
import time
import sys
import datetime


def analyze_dataset(wd):
    """Analyze dataset, i.e. predict barrier, cell objects, calculate hull,
    map cell objects to tracings and predict synapses together with ultrastructure
    of cells to create connectivity matrix

    Parameters
    ----------
    wd : str
        path to working directory
    """
    enrich_tracings_all(wd, overwrite=False)
    # remap_tracings_all(wd)
    detect_synapses(wd)
    predict_celltype_label(wd)
    type_sorted_wiring(wd)


def enrich_tracings_all(wd, overwrite=False, qsub_pe=None, qsub_queue=None):
    """Run :func: 'enrich_tracings' on available cluster nodes defined by
    somaqnodes or using multiprocessing.

    Parameters
    ----------
    wd : str
        Path to working directory
    overwrite : bool
        enforce overwriting of existing files
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    """
    use_qsub = qsub_pe is not None or qsub_queue is not None
    start = time.time()
    skel_dir = wd + '/tracings/'
    if not os.path.isdir(wd + '/neurons/'):
        os.makedirs(wd + '/neurons/')
    anno_list = [os.path.join(skel_dir, f) for f in
                 next(os.walk(skel_dir))[2] if 'k.zip' in f]
    print "------------------------------\n" \
          "Starting enrichment of %d cell tracings" % len(anno_list)
    np.random.shuffle(anno_list)
    if use_qsub and __QSUB__:
        nb_lists = 30
        list_of_lists = [[anno_list[i::nb_lists], wd, overwrite] for i
                         in xrange(nb_lists)]
        QSUB_script(list_of_lists, 'skeleton_mapping', queue=qsub_queue,
                    pe=qsub_pe, n_cores=np.max((int(cpu_count() / 2), 1)))
    elif use_qsub and not __QSUB__:
        raise RuntimeError("QSUB not available. Please make sure QSUB is"
                           "configured correctly.")
    else:
        nb_lists = np.max((int(cpu_count() / 2), 1))
        list_of_lists = [[anno_list[i::nb_lists], wd, overwrite] for i
                         in xrange(nb_lists)]
        start_multiprocess(enrich_tracings_star, list_of_lists,
                           nb_cpus=nb_lists)
    predict_celltype_label(wd)
    diff = time.time() - start
    print "Finished tracing enrichment (cell object mapping, prediction of " \
          "sub-cellular compartments and cell types) after %s." % \
          str(datetime.timedelta(seconds=diff))


def enrich_tracings_star(params):
    enrich_tracings(params[0], params[1], overwrite=params[2])


def enrich_tracings(anno_list, wd, map_objects=True, method='hull', radius=1200,
                    thresh=2.2, filter_size=(2786, 1594, 250),
                    create_hull=True, max_dist_mult=1.4, detect_outlier=True,
                    dh=None, overwrite=False, nb_neighbors=20,
                    nb_hull_vox=500, context_range=6000,
                    neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100,
                    write_obj_voxel=True):
    """Enriches a list of paths (to tracings) using dataset in working
    directory. Writes enriched tracings to 'neuron' folder in working directory,
    or, if specified, to DataHandler().data_path.

    Parameters
    ----------
    wd : str
        Path to working directory
    anno_list : list
        paths to tracing .k.zip's
    map_objects : bool
        Map cell components to skeleton
    method : str
        Either 'kd' for fix radius or 'hull'/'supervoxel' if
        membrane is available.
    radius : int
        Radius in nm. Single integer if integer radius is for
        all objects the same. If list of three integer stick to ordering
        [mitos, vc, sj].
    thresh : int
        Denotes the factor which is multiplied with the maximum
        membrane probability. The resulting value is used as threshold after
        which the membrane is assumed to be existant.
    nb_rays : int
        Number of rays send at each skeleton node
        (multiplied by a factor of 5). Defines the angle between two rays
        (=360 / nb_rays) in the orthogonal plane.
    neighbor_radius : int
        Radius of ball in which to look for supporting
        hull voxels. Used during outlier detection.
    nb_neighbors : int
        minimum number of neighbors needed during outlier detection
        for a single hull point to survive.
    filter_size : list, tuple of int
        minimum cell object sizes for each object (mitos, vesicle clouds,
        synaptic clefts)
    nb_hull_vox : int
        Number of object hull voxels which are used to
        estimate spatial proximity to skeleton.
    context_range : int
        Range of property features
    nb_voting_neighbors : int
        Number votes of skeleton hull voxels
        (membrane representation) for object-mapping.
    dh: DataHandler
        object containing SegmentationDataObjects mitos, vc, sj
    create_hull : bool
        create skeleton membrane estimation
    max_dist_mult : float
        Multiplier for radius to generate maximal
        distance of hull points to source node.
    detect_outlier : bool
        Use outlier-detection if True
    overwrite : bool
        Overwrite existing .k.zip's of mapped skeletons
    write_obj_voxel : bool
        write object voxel coordinates to .k.zip
    """
    rf_axoness_p = wd + '/models/rf_axoness/rf.pkl'
    rf_spiness_p = wd + '/models/rf_spiness/rf.pkl'
    rfc_axoness, rfc_spiness = None, None
    if dh is None:
        dh = DataHandler(wd)
    if not os.path.isdir(dh.myelin_ds_path):
        print "Could not find myelin dataset. Enriched tracings won't " \
              "contain myelin prediction."
        dh.myelin_ds_path = None
    output_dir = dh.data_path
    if not overwrite:
        existing_skel = [re.findall('[^/]+$', os.path.join(dp, f))[0] for
                         dp, dn, filenames in os.walk(output_dir)
                         for f in filenames if 'k.zip' in f]
    else:
        existing_skel = []
    if anno_list == [] and dh is not None:
        anno_list = [os.path.join(dp, f) for dp, dn, filenames in
                     os.walk(dh.skeleton_path) for f in filenames if
                     'consensus.nml' in f]
        anno_list += [os.path.join(dp, f) for dp, dn, filenames in
                      os.walk(dh.skeleton_path) for f in filenames if
                      'k.zip' in f]
    elif anno_list == [] and dh is None:
        raise RuntimeError("Aborting mapping procedure. Either DataHandler"
                           "with source path (dh.skeleton_path) or "
                           "nml_list must be given!")
    todo_skel = list(set([re.findall('[^/]+$', f)[0] for f in anno_list]) -
                     set(existing_skel))
    todo_skel = [f for f in anno_list if re.findall('[^/]+$', f)[0]
                 in todo_skel]
    if len(todo_skel) == 0:
        print "Nothing to do. Aborting."
        return
    # print "Found %d processed Skeletons. %d left. Writing result to %s. " \
    #       "Using %s barrier." % (len(existing_skel), len(todo_skel),
    #                              dh.data_path, dh.mem_path)
    cnt = 0
    if map_objects:
        rfc_axoness, rfc_spiness = load_rfcs(rf_axoness_p, rf_spiness_p)
    for ii, filepath in enumerate(list(todo_skel)[::-1]):
        dh.skeletons = {}
        cnt += 1
        _list = load_ordered_mapped_skeleton(filepath)
        annotation = _list[0]
        soma = connect_soma_tracing(_list[4])
        try:
            ix = int(re.findall('.*?([\d]+)', filepath)[-3])
        except IndexError:
            ix = cnt
        path = dh.data_path + re.findall('[^/]+$', filepath)[0]
        skel = SkeletonMapper(annotation, dh, ix=ix, soma=soma,
                              context_range=context_range)
        skel.write_obj_voxel = write_obj_voxel
        if create_hull:
            # try:
            skel.hull_sampling(detect_outlier=detect_outlier, thresh=thresh,
                               nb_neighbors=nb_neighbors,
                               neighbor_radius=neighbor_radius,
                               max_dist_mult=max_dist_mult)
            if map_objects:
                skel.annotate_objects(dh, radius, method, thresh,
                                      filter_size, nb_hull_vox=nb_hull_vox,
                                      nb_voting_neighbors=nb_voting_neighbors,
                                      nb_rays=nb_rays,
                                      nb_neighbors=nb_neighbors,
                                      neighbor_radius=neighbor_radius,
                                      max_dist_mult=max_dist_mult)
            # except Exception, e:
            #     warnings.warn(
            #         "%s. Problem with tracing %s. Skipping it." %
            #         (e, filepath), RuntimeWarning)
        if map_objects:
            if rfc_spiness is not None:
                skel.predict_property(rfc_spiness, 'spiness')
            skel._property_features = None
            if rfc_spiness is not None and rfc_axoness is not None:
                skel.predict_property(rfc_axoness, 'axoness')
        if not os.path.exists(dh.data_path):
            os.makedirs(dh.data_path)
        skel.write2kzip(path)
    print "Finished enrichment of %d cell tracings" % len(anno_list)


def remap_tracings_all(wd, dest_dir=None, recalc_prop_only=False,
                       method='hull', context_range=6000,
                       qsub_pe=None, qsub_queue=None):
    """Run remap_tracings on available cluster nodes defined by
    somaqnodes or using single node multiprocessing.

    Parameters
    ----------
    wd : str
        Path to working directory, which contains skeleton nml / kzip files in

    dest_dir : str
        Directory path to store mapped skeletons
    recalc_prop_only : bool
        Recalculate properties (spiness, axoness) only,
        without calculating hull
    method : str
        Method for object mapping procedure
    context_range : int
        Context range for property features
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    """
    use_qsub = qsub_pe is not None or qsub_queue is not None

    anno_list = get_filepaths_from_dir(wd + '/neurons/')
    np.random.shuffle(anno_list)
    if dest_dir is not None and not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    nb_processes = np.max((len(anno_list) / 3, 3))
    list_of_lists = [[wd, anno_list[i::nb_processes], dest_dir,
                      recalc_prop_only, method, context_range]
                     for i in xrange(nb_processes)]
    if use_qsub and __QSUB__:
        QSUB_script(list_of_lists, 'skeleton_remapping', queue=qsub_queue,
                    pe=qsub_pe, n_cores=np.max((int(cpu_count()), 1)))
    elif use_qsub and not __QSUB__:
        raise RuntimeError("QSUB not available. Please make sure QSUB is"
                           "configured correctly.")
    else:
        start_multiprocess(remap_tracings_star, list_of_lists, debug=False)


def remap_tracings_star(params):
    """Helper function for multiprocessed remap_tracings."""
    remap_tracings(params[0], params[1], output_dir=params[2],
                   recalc_prop_only=params[3], method=params[4],
                   context_range=params[5])


def remap_tracings(wd, mapped_skel_paths, dh=None, method='hull', radius=1200,
                   thresh=2.2, filter_size=(2786, 1594, 250),
                   max_dist_mult=1.4, nb_neighbors=20, nb_hull_vox=500,
                   neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100,
                   output_dir=None, write_obj_voxel=True,
                   mito_min_votes=235, vc_min_votes=191, sj_min_votes=346,
                   recalc_prop_only=True, context_range=6000):
    """Remaps objects to skeleton without recalculating the hull.

    Parameters
    ----------
    wd : str
        Path to working directory
    mapped_skel_paths : list of str
        Paths to tracings
    dh: DataHandler
        object containing SegmentationDataObjects mitos, vc, sj
    output_dir : str
        path to output dir. If none, use given path
    write_obj_voxel : bool
        write object voxel to k.zip
    sj_min_votes : int
        Best F2 score found by eval
    mito_min_votes : int
        Best F1 score found by eval
    vc_min_votes : int
        Best F2 score found by eval
    method : str
        Either 'kd' for fix radius or 'hull'/'supervoxel' if
        membrane is available.
    radius : int
        Radius in nm. Single integer if integer radius is for all
        objects the same. If list of three integer stick to ordering
        [mitos, vc, sj].
    thresh : float
        Denotes the factor which is multiplied with the maximum
        membrane probability. The resulting value is used as threshold after
        which the membrane is assumed to be existant.
    nb_rays : int
        Number of rays send at each skeleton node
        (multiplied by a factor of 5). Defines the angle between two rays
        (=360 / nb_rays) in the orthogonal plane.
    neighbor_radius : int
        Radius of ball in which to look for supporting
        hull voxels. Used during outlier detection.
    nb_neighbors : int
        minimum number of neighbors needed during
        outlier detection for a single hull point to survive.
    filter_size : List of integer
        for each object [mitos, vc, sj]
    nb_hull_vox : int
        Number of object hull voxels which are used to
        estimate spatial proximity to skeleton.
    context_range : int
        Range of property features
    nb_voting_neighbors : int
        Number votes of skeleton hull voxels (membrane
        representation) for object-mapping.
    max_dist_mult : float
        Multiplier for radius to generate maximal
        distance of hull points to source node.
    output_dir : str
        Path to output directory, if None dh._data_path
        will be used.
    recalc_prop_only : bool
        Recalculate only features and prediction of
        properties (axoness, spiness)
    write_obj_voxel : bool
        write object voxel coordinates to kzip
    """

    rf_axoness_p = wd + '/models/rf_axoness/rf.pkl'
    rf_spiness_p = wd + '/models/rf_spiness/rf.pkl'
    rfc_axoness, rfc_spiness = load_rfcs(rf_axoness_p, rf_spiness_p)
    cnt = 0
    if dh is None:
        dh = DataHandler(wd)
    for skel_path in mapped_skel_paths:
        cnt += 1
        # get first element in list and only skeletonAnnotation
        mapped_skel_old, mito, vc, sj, soma = \
            load_anno_list([skel_path], load_mitos=False)[0]
        if output_dir is not None:
            for fpath in mapped_skel_paths:
                shutil.copyfile(fpath, output_dir + os.path.split(fpath)[1])
            path = output_dir + re.findall('[^/]+$', skel_path)[0]
        else:
            path = skel_path
        try:
            ix = int(re.findall('.*?([\d]+)', skel_path)[-3])
        except IndexError:
            ix = cnt
        new_skel = SkeletonMapper(mapped_skel_old, dh,
                                  ix=ix, soma=soma)
        if recalc_prop_only:
            new_skel.old_anno.filename = skel_path
        else:
            new_skel.obj_min_votes['mitos'] = mito_min_votes
            new_skel.obj_min_votes['vc'] = vc_min_votes
            new_skel.obj_min_votes['sj'] = sj_min_votes
            new_skel.write_obj_voxel = write_obj_voxel
            new_skel.annotate_objects(
                dh, radius, method, thresh, filter_size,
                nb_hull_vox=nb_hull_vox,
                nb_voting_neighbors=nb_voting_neighbors, nb_rays=nb_rays,
                nb_neighbors=nb_neighbors, neighbor_radius=neighbor_radius,
                max_dist_mult=max_dist_mult)
        if rfc_spiness is not None:
            new_skel._property_features, new_skel.property_feat_names, \
            spiness_given = calc_prop_feat_dict(new_skel, context_range)
            new_skel.predict_property(rfc_spiness, 'spiness')
            new_skel._property_features = None
        if rfc_spiness is not None and rfc_axoness is not None:
            new_skel.predict_property(rfc_axoness, 'axoness')
        if not recalc_prop_only:
            new_skel.write2kzip(path)
        else:
            dummy_skel = Skeleton()
            dummy_skel.add_annotation(new_skel.old_anno)
            dummy_skel.add_annotation(mito)
            dummy_skel.add_annotation(vc)
            dummy_skel.add_annotation(sj)
            if soma is not None:
                dummy_skel.add_annotation(soma)
            dummy_skel.toNml(path[:-5] + 'nml')
            write_data2kzip(path, path[:-5] + 'nml')
            write_feat2csv(path + 'axoness_feat.csv',
                           new_skel.property_features['axoness'])
            write_feat2csv(path + 'spiness_feat.csv',
                           new_skel.property_features['spiness'])
            write_data2kzip(path, path + 'axoness_feat.csv')
            write_data2kzip(path, path + 'spiness_feat.csv')


def detect_synapses(wd, qsub_pe=None, qsub_queue=None):
    """Detects contact sites between enriched tracings and writes contact site
     summary and synapse summary to working directory.

    Parameters
    ----------
    wd : str Path to working directory
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    """
    print "------------------------------\n" \
          "Starting contact site detection with synapse classification."
    use_qsub = qsub_pe is not None or qsub_queue is not None
    start = time.time()
    nml_list = get_filepaths_from_dir(wd + '/neurons/')
    cs_path = wd + '/contactsites/'
    for ending in ['', 'cs', 'cs_vc', 'cs_sj', 'cs_vc_sj', 'pairwise',
                   'overlap_vx']:
        if not os.path.exists(cs_path+ending):
            os.makedirs(cs_path+ending)
    anno_permutations = list(combinations(nml_list, 2))

    if use_qsub and __QSUB__:
        list_of_lists = [[list(anno_permutations[i::300]), cs_path]
                         for i in xrange(300)]
        QSUB_script(list_of_lists, 'synapse_mapping', queue=qsub_queue,
                    pe=qsub_pe, n_cores=np.max((int(cpu_count()), 1)))
    elif use_qsub and not __QSUB__:
        raise RuntimeError("QSUB not available. Please make sure QSUB is"
                           "configured correctly.")
    else:
        prepare_syns_btw_annos(anno_permutations, cs_path)
    write_summaries(wd)
    diff = time.time() - start
    print "Finished connectivity analysis after %s.\n" \
          "---------------------------" % str(datetime.timedelta(seconds=diff))


def detect_similar_tracings(wd):
    """Print similar skeleton file paths

    Parameters
    ----------
    wd : str
        path to working directory
    """
    skel_paths = get_filepaths_from_dir(wd + '/tracings/')
    start_multiprocess(similarity_check_star, list(combinations(skel_paths, 2)))


def get_connectivity_list(wd):
    """Find synapses between tracing pairs

    Parameters
    ----------
    wd : str
        path to working directory

    Returns
    -------
    np.array (n, 1), np.array (n x 2)
        descending array of synapse (axo-dendritic, if count is zero, axo-axonic
        touches arepresent) number between corresponding tracing pairs
        (tuple of ids)
    """
    conn_dict_path = wd + '/contactsites/connectivity_dict.pkl'
    assert os.path.exists(conn_dict_path)
    conn_dict = load_pkl2obj(conn_dict_path)
    cell_ids = []
    synapse_touches = []
    for pair_name, pair in conn_dict.iteritems():
        if pair['total_size_area'] == 0:
            continue
        skel_id1, skel_id2 = re.findall('(\d+)_(\d+)', pair_name)[0]
        cell_ids.append(np.array([int(skel_id1), int(skel_id2)]))
        indiv_syn_sizes = np.array(pair['sizes_area'])
        indiv_syn_axoness = np.array(pair['partner_axoness']) == 1
        pair['sizes_area'] = indiv_syn_sizes[~indiv_syn_axoness]
        synapse_touches.append(len(pair['sizes_area']))
    synapse_touches = np.array(synapse_touches)
    cell_ids = np.array(cell_ids)
    sorted_ixs = np.argsort(synapse_touches)[::-1]
    synapse_touches = synapse_touches[sorted_ixs]
    cell_ids = cell_ids[sorted_ixs]
    return np.array(synapse_touches), np.array(cell_ids)
