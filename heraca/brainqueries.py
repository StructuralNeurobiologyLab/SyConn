__author__ = 'pschuber'
import os
import re
import shutil
from numpy import array as arr
from scipy import spatial
import h5py
import shutil
import zipfile
import NewSkeletonUtils as nsu
from itertools import combinations
import numpy as np
from NewSkeleton import NewSkeleton
try:
    from NewSkeleton import annotationUtils as au
except:
    import annotationUtils as au
try:
    from NewSkeleton.NewSkeletonUtils import annotation_from_nodes
except:
    from NewSkeletonUtils import annotation_from_nodes
from NewSkeleton import NewSkeleton
try:
    from Sven.QSUB import QSUB_MAIN as qm
except:
    from QSUB import QSUB_MAIN as qm
from heraca.utils.datahandler import *
from sklearn.externals import joblib
from conmatrix import write_summaries
from heraca.processing.learning_ut import write_feat2csv, load_rfcs,\
    load_csv2feat, start_multiprocess
from heraca.processing.mapper import prepare_syns_btw_annos, SkeletonMapper,\
    similarity_check, similarity_check_star, prepare_syns_btw_annos


def QSUB_mapping(skel_dir='/lustre/pschuber/consensi_fuer_joergen/'
                          'consensi_fuer_phil/',
                 output_dir='/lustre/pschuber/m_consensi_rr/'):#'/lustre/pschuber/consensi_fuer_joergen/'):#
    """
    Run annotate annos on available cluster nodes defined by somaqnodes.
    :param skel_dir: str Path to skeleton nml / kzip files
    :param output_dir: str Directory path to store mapped skeletons
    """
    def chunkify(lst,n):
        return [[lst[i::n], output_dir] for i in xrange(n)]
    anno_list = [os.path.join(skel_dir, f) for f in
                next(os.walk(skel_dir))[2] if 'k.zip' in f]
    np.random.shuffle(anno_list)
    print "Found %d mapped Skeletons." % len(anno_list)
    list_of_lists = chunkify(anno_list, 60)
    qm.QSUB_script(list_of_lists, 'skeleton_mapping', queue='somaqnodes',
                work_folder="/home/pschuber/QSUB/", username="pschuber",
                python_path="/home/pschuber/anaconda/bin/python",
                path_to_scripts="/home/pschuber/skeleton-analysis/Philipp/QSUB",
                job_name='skel_map')


def annotate_annos(anno_list, map_objects=True, method='hull', radius=1200,
                   thresh=2.2, filter_size=[2786, 1594, 250], create_hull=True,
                   max_dist_mult=1.4, save_files=True, detect_outlier=True,
                   dh=None, overwrite=False, nb_neighbors=20, nb_hull_vox=500,
                   neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100,
                   write2pkl=False, write_obj_voxel=True, output_dir=None,
                   gt_sampling=False, rf_axoness_p='/lustre/pschuber/gt_axoness/'
                                                 'rf/rf.pkl',
                   rf_spiness_p='/lustre/pschuber/gt_spines/rf/rf.pkl',
                   load_mapped_skeletons=True, az_min_votes=346):
    """
    Example function how to annotate skeletons. Mappes a list of paths to nml
    files to objects given in DataHandler (use dh-keyword for warmstart).
    :param anno_list: list of paths to nml or kzip files
    :param map_objects: bool, Map cell components to skeleton
    :param method: Either 'kd' for fix radius or 'hull'/'supervoxel' if
    membrane is available.
    :param radius: Radius in nm. Single integer if integer radius is for all
    objects the same. If list of three integer stick to ordering [mitos, p4, az].
    :param thresh: Denotes the factor which is multiplied with the maximum
    membrane probability. The resulting value is used as threshold after
     which the membrane is assumed to be existant.
    :param nb_rays: Integer, Number of rays send at each skeleton node
    (multiplied by a factor of 5). Defines the angle between two rays
    (=360 / nb_rays) in the orthogonal plane.
    :param neighbor_radius: Radius of ball in which to look for supporting
    hull voxels. Used during outlier detection.
    :param nb_neighbors: Integer, minimum number of neighbors needed during
    outlier detection for a single hull point to survive.
    :param filter_size: List of integer for each object [mitos, p4, az]
    :param nb_hull_vox: Number of object hull voxels which are used to
    estimate spatial proximity to skeleton.
    :param nb_voting_neighbors: Number votes of skeleton hull voxels (membrane
    representation) for object-mapping.
    :param dh: DataHandler object containing SegmentationDataObjects
     mitos, p4, az
    :param create_hull: Boolean, create skeleton membrane estimation
    :param max_dist_mult: float Multiplier for radius to generate maximal
    distance of hull points to source node.
    :param save_files: Save mapped skeletons as nml
    :param detect_outlier: Use outlier-detection if True
    :param overwrite: Overwrite existing nml's of mapped skeletons
    :param write_obj_voxel: bool if true write all object voxels to additional
    'obj_voxel' tree in nml.
    :param output_dir: str Path to output directory, if None dh._data_path will
    be used.
    :return:
    """
    if output_dir is None:
        if dh is None:
            output_dir = '/lustre/pschuber/m_consensi_rr/'
        else:
            output_dir = dh._data_path
    if not overwrite:
        existing_skel = [re.findall('[^/]+$', os.path.join(dp, f))[0] for
                         dp, dn, filenames in os.walk(output_dir+'nml_obj/')
                         for f in filenames if 'k.zip' in f]
    else:
        existing_skel = []
    if anno_list == [] and dh is not None:
        anno_list = [os.path.join(dp, f) for dp, dn, filenames in
                     os.walk(dh._skeleton_path) for f in filenames if
                     'consensus.nml' in f]
        anno_list += [os.path.join(dp, f) for dp, dn, filenames in
                     os.walk(dh._skeleton_path) for f in filenames if
                     'k.zip' in f]
    elif anno_list == [] and dh is None:
        print "Aborting mapping procedure. Either DataHandler with source path " \
              "(dh._skeleton_path) or nml_list must be given!"
        return
    todo_skel = list(set([re.findall('[^/]+$', f)[0] for f in anno_list])
                      - set(existing_skel))
    todo_skel = [f for f in anno_list if re.findall('[^/]+$', f)[0] in todo_skel]
    if len(todo_skel) == 0:
        print "Nothing to do. Aborting."
        return
    if dh is None:
        print 'Loading object data.'
        dh = DataHandlerObject()
        dh._data_path = output_dir
    print "Found %d processed Skeletons. %d left. Writing result to %s. Using" \
          " %s barrier." % (len(existing_skel), len(todo_skel), dh._data_path,
                            dh._mem_path)
    cnt = 0
    rfc_axoness, rfc_spiness = load_rfcs(rf_axoness_p, rf_spiness_p)
    for filepath in list(todo_skel)[::-1]:
        dh.skeletons = {}
        cnt += 1
        try:
            if load_mapped_skeletons:
                _list = load_ordered_mapped_skeleton(filepath)
                annotation = _list[0]
                soma = connect_soma_tracing(_list[4])
                if len(_list[4].getNodes()) != 0:
                    print "Loaded soma of skeleton."
            else:
                soma = None
                annotation = au.loadj0126NML(filepath)[0]
            try:
                id = int(re.findall('.*?([\d]+)', filepath)[-3])
            except IndexError:
                id = cnt
        except Exception, e:
            print e
            print "Couldn't load annotation file from", filepath
            continue
        path = dh._data_path + 'nml_obj/' + re.findall('[^/]+$', filepath)[0]
        skel = create_skel(dh, annotation, id=id, soma=soma)
        skel.az_min_votes = az_min_votes
        skel.write_obj_voxel = write_obj_voxel
        # uncomment to sample all to create voting value for obj.-mapping eval
        # skel.obj_min_votes = {'mitos': 0, 'az':0, 'p4':0}
        try:
            if create_hull:
                skel.hull_sampling(detect_outlier=detect_outlier, thresh=thresh,
                                   nb_neighbors=nb_neighbors,
                                   neighbor_radius=neighbor_radius,
                                   max_dist_mult=max_dist_mult)

            if map_objects:
                if not gt_sampling:
                    skel.annotate_objects(dh, radius, method, thresh,
                    filter_size, nb_hull_vox=nb_hull_vox,
                    nb_voting_neighbors=nb_voting_neighbors, nb_rays=nb_rays,
                    nb_neighbors=nb_neighbors, neighbor_radius=neighbor_radius,
                    max_dist_mult=max_dist_mult)
                else:
                    skel.annotate_objects(dh, method='kd', radius=1200,
                    thresh=2.2, filter_size=[2786, 1594, 498], max_dist_mult=1.4,
                    detect_outlier=True, nb_neighbors=20, nb_hull_vox=50,
                    neighbor_radius=220, nb_rays=20, nb_voting_neighbors=500)
                    dh_tmp = DataHandlerObject(p4_source='', az_source='', mito_source='',
                                          datapath=dh._data_path, mempath=dh._mem_path)
                    dh_tmp.mitos = skel.mitos
                    dh_tmp.p4 = skel.p4
                    dh_tmp.az = skel.az
                    skel.annotate_voting_neighobjects(dh_tmp, method='gt_sampling', radius=1200,
                               thresh=2.2, filter_size=[0, 0, 0], max_dist_mult=1.4,
                               detect_outlier=True, nb_neighbors=20, nb_hull_vox=50,
                               neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100)
        except IndexError:
            print "Skeleton contains only one or no node." \
                  " Interpolation failed. Continuing with next file."
            continue
        # save mapping result before prediction
        # if save_files:
        #     for folder in ['nml_obj/']:
        #         if not os.path.exists(dh._data_path+folder):
        #             os.makedirs(dh._data_path+folder)
        #     skel.write2kzip(path)
        # if write2pkl:
        #     skel.write2pkl(dh._data_path + 'pkl/' + re.findall('[^/]+$',
        #                                                        filepath)[0])
        # predict axoness and write to uninterpolated anno
        print "Starting cell compartment prediction."
        if rfc_spiness != None:
            skel.predict_property(rfc_spiness, 'spiness')
        skel._property_features = None
        # TODO: FIXME calc spiness feat, pred spiness, calc axoness feat, but
        # only feature dependent on spiness
        if rfc_spiness != None and rfc_axoness != None:
            skel.predict_property(rfc_axoness, 'axoness')
        if save_files:
            for folder in ['nml_obj/']:
                if not os.path.exists(dh._data_path+folder):
                    os.makedirs(dh._data_path+folder)
            skel.write2kzip(path)
        if write2pkl:
            skel.write2pkl(dh._data_path + 'pkl/' + re.findall('[^/]+$',
                                                               filepath)[0])

def QSUB_remapping(anno_list=[], dest_dir='/lustre/pschuber/m_consensi_rr/nml_obj/',
                   az_kd_set='/lustre/sdorkenw/j0126_cset/obj_az_1037_3d_3/',
                   az_size_threshold=250, min_votes_az=346,
                   recalc_prop_only=False, method='hull'):
    """
    Run annotate annos on available cluster nodes defined by somaqnodes.
    :param anno_list: str Paths to skeleton nml / kzip files
    :param dest_dir: str Directory path to store mapped skeletons
    """
    def chunkify(lst, n):
        return [[lst[i::n], dest_dir, az_kd_set, az_size_threshold,
                 min_votes_az, recalc_prop_only, method] for i in xrange(n)]
    if anno_list == []:
        anno_list = [os.path.join(dest_dir, f) for f in
                    next(os.walk(dest_dir))[2] if 'k.zip' in f]
    np.random.shuffle(anno_list)
    if dest_dir is not None and not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    print "Found %d mapped Skeletons." % len(anno_list)
    nb_processes = np.max((len(anno_list)/3, 3))
    list_of_lists = chunkify(anno_list, nb_processes)
    qm.QSUB_script(list_of_lists, 'skeleton_remapping', queue='somaqnodes',
                work_folder="/home/pschuber/QSUB/", username="pschuber",
                python_path="/home/pschuber/anaconda/bin/python",
                path_to_scripts="/home/pschuber/skeleton-analysis/Philipp/QSUB",
                job_name='skel_remap')


def remap_skeletons(mapped_skel_paths=[], dh=None, method='hull', radius=1200,
                    thresh=2.2, filter_size=[2786, 1594, 250], max_dist_mult=1.4,
                    save_files=True, nb_neighbors=20, nb_hull_vox=500,
                    neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100,
                    output_dir=None, rf_axoness_p='/lustre/pschuber/gt_axoness/'
                    'rf/rf.pkl', write_obj_voxel=True,
                    rf_spiness_p='/lustre/pschuber/gt_spines/rf/rf.pkl',
                    az_kd_set='/lustre/sdorkenw/j0126_cset/obj_az_1037_3d_3/',
                    mito_min_votes=235, p4_min_votes=191, az_min_votes=346,
                    max_neck2endpoint_dist=3000, max_head2endpoint_dist=600,
                    recalc_prop_only=False, nb_cpus=4):
    """
    Only remaps objects to skeleton without recalculating the hull.
    Min votes for cell objects are evaluated by f1-score using 500
    object hull voxels:
    # 235 0.997642899234
    # 191 0.98241358399
    # 346 0.986725663717
    :param output_dir: str path to output dir. If none, use given path
    :param rf_axoness_p: path to random forest for axoness
    :param write_obj_voxel: bool write object voxel to k.zip
    :param rf_spiness_p: path to random forest for spiness
    :param az_kd_set: path to az object dataset
    :param az_min_votes: Best F2 score found by eval
    :param mito_min_votes: Best F1 score found by eval
    :param p4_min_votes: Best F2 score found by eval
    :return:
    """
    rfc_axoness, rfc_spiness = load_rfcs(rf_axoness_p, rf_spiness_p)
    cnt = 0
    print
    for skel_path in mapped_skel_paths:
        cnt += 1
        # get first element in list and only skeletonAnnotation
        mapped_skel_old, mito, p4, az, soma = load_anno_list([skel_path],
                                                       load_mitos=False)[0]


        # if len(soma.getNodes()) == 0:
        #     print "skipping remapping of skel %s, because soma is empty." % \
        #         mapped_skel_old.filename
        #     continue


        if output_dir is not None:
            path = output_dir + re.findall('[^/]+$', skel_path)[0]
        else:
            path = skel_path
        try:
            id = int(re.findall('.*?([\d]+)', skel_path)[-3])
        except IndexError:
            id = cnt
        print "Remapping skeleton at %s and writing result to %s." % (skel_path,
        path)
        new_skel = SkeletonMapper(mapped_skel_old, mapped_skel_old.scaling,
                                  id=id, soma=soma)
        new_skel.nb_cpus = nb_cpus
        if recalc_prop_only:
            print "--- Recalculating properties only ---"
            new_skel.old_anno.filename = skel_path
        else:
            if dh is None:
                dh = DataHandlerObject(az_source=az_kd_set)
            new_skel.obj_min_votes['mitos'] = mito_min_votes
            new_skel.obj_min_votes['p4'] = p4_min_votes
            new_skel.obj_min_votes['az'] = az_min_votes
            # uncomment to sample all to create voting value for obj.-mapping eval
            # new_skel.obj_min_votes = {'mitos': 0, 'az':0, 'p4':0}
            new_skel.write_obj_voxel = write_obj_voxel
            new_skel.annotate_objects(
                dh, radius, method, thresh, filter_size, nb_hull_vox=nb_hull_vox,
                nb_voting_neighbors=nb_voting_neighbors, nb_rays=nb_rays,
                nb_neighbors=nb_neighbors, neighbor_radius=neighbor_radius,
                max_dist_mult=max_dist_mult)
        # predict axoness and write to uninterpolated anno
        # new_skel._property_features = {}
        # new_skel._property_features['axoness'] = load_csv2feat(path)[0]
        # new_skel._property_features['spiness'] = load_csv2feat(path,
        #                                                        property='spiness')[0]
        if rfc_spiness != None:
            new_skel.predict_property(rfc_spiness, 'spiness',
                        max_neck2endpoint_dist=max_neck2endpoint_dist,
                        max_head2endpoint_dist=max_head2endpoint_dist)
        if rfc_spiness != None and rfc_axoness != None:
            new_skel.predict_property(rfc_axoness, 'axoness')
        if save_files and not recalc_prop_only:
            new_skel.write2kzip(path)
        if save_files and recalc_prop_only:
            dummy_skel = NewSkeleton()
            dummy_skel.add_annotation(new_skel.old_anno)
            dummy_skel.add_annotation(mito)
            dummy_skel.add_annotation(p4)
            dummy_skel.add_annotation(az)
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
        print "Remapped skeleton %s successfully." % path


def create_skel(dh, skel_source, id=None, soma=None):
    """
    Creates MappedSkeleton object using DataHandler and number/id of nml file
    or annotation object directly.
    :param dh: DataHandlerObject
    :param skel_source: ID/number of nml file or annotation object
    :param id: extracted from path or should be given if skel_source is
    annotation object
    :return: MappedSkeleton object
    """
    if np.isscalar(skel_source):
        assert skel_source < len(dh._skeleton_files)
        id = int(re.findall('.*?([\d]+)', dh._skeleton_files[skel_source])[-3])
        print "--- Initializing skeleton %d with %d cores from path." %\
              (id, dh.nb_cpus)
        if id in dh.skeletons.keys():
            skel = dh.skeletons[id]
            return skel
        print "Skeleton %d does not exist yet. Building from scratch." % id
        skel = SkeletonMapper(dh._skeleton_files[skel_source], dh.scaling,
                              soma=soma)
    else:
        print "--- Initializing skeleton %d with %d cores from annotation" \
              " object." % (id, dh.nb_cpus)
        if id in dh.skeletons.keys():
             return dh.skeletons[id]
        skel = SkeletonMapper(skel_source, dh.scaling, id=id, soma=soma)
    skel._data_path = dh._data_path
    skel._mem_path = dh._mem_path
    skel.nb_cpus = dh.nb_cpus
    return skel


def QSUB_synapse_mapping(source_path='/lustre/pschuber/m_consensi_rr/nml_obj/',
                         max_hull_dist=60):
    """
    Finds synapses between skeletons and writes each contact site with at least
    p4 to a nml file. Afterwards these are collected and compressed to one
    contact_site.nml file with all necessary information about the synapses.
    This file can be used to create a wiring diagram.
    :param source_path: str Path to directory containing mapped annotation
    kzip's which are t be computed
    """
    nml_list = get_filepaths_from_dir(source_path)
    cs_path = os.path.dirname(nml_list[0]) + '/contact_sites_new/'
    for ending in ['', 'cs', 'cs_p4', 'cs_az', 'cs_p4_az', 'pairwise',
                   'overlap_vx']:
        if not os.path.exists(cs_path+ending):
            os.makedirs(cs_path+ending)
    anno_permutations = list(combinations(nml_list, 2))
    def chunkify(lst, n):
        return [[list(lst[i::n]), cs_path, max_hull_dist] for i in xrange(n)]
    list_of_lists = chunkify(anno_permutations, 300)
    qm.QSUB_script(list_of_lists, 'synapse_mapping', queue='somaqnodes',
                work_folder="/home/pschuber/QSUB/", username="pschuber",
                python_path="/home/pschuber/anaconda/bin/python",
                path_to_scripts="/home/pschuber/skeleton-analysis/Philipp/QSUB",
                job_name='syn_map')
    write_summaries(cs_path)


def annotate_dense_vol():
    # path='/lustre/pschuber/dense_vol_tracings/gt_skeletons.117.k.zip'
    # skels = au.loadj0126NML(path)
    # cnt = 0
    # for ii, skel in enumerate(skels):
    #     dest_path = '/lustre/pschuber/dense_vol_tracings/source/' \
    #                 'iter_0_%d.k.zip' % ii
    #     if os.path.exists(dest_path):
    #         continue
    #     print "Writing skel", skel
    #     dummy_skel = NewSkeleton()
    #     dummy_skel.add_annotation(skel)
    #     cnt += 1
    #     dummy_skel.to_kzip(dest_path)
    # print "Wrote %d files to source folder." % cnt
    # skel_paths = get_unique_skeletons('/lustre/pschuber/dense_vol_tracings/source/')
    # QSUB_mapping(skel_dir='/lustre/pschuber/dense_vol_tracings/source/',
    #              output_dir='/lustre/pschuber/dense_vol_tracings_shrinked/')
    dest_paths = get_filepaths_from_dir('/lustre/pschuber/dense_vol_tracings_shrinked/nml_obj/')
    # QSUB_synapse_mapping(source_path='/lustre/pschuber/dense_vol_tracings/nml_obj/')
    start_multiprocess(helper_write_dense_vol_skel_hull, dest_paths, nb_cpus=10,
                       debug=False)


def get_unique_skeletons(skel_dir):
    """
    Gets unique skeleton paths in skeleton folder.
    :param skel_dir: Path to folder containing skeleton files ending with k.zip
    :return: list of skeleton paths which are unique
    """
    skel_paths = get_filepaths_from_dir(skel_dir)
    unique_skel_paths = []
    start_multiprocess(similarity_check_star, list(combinations(skel_paths, 2)))
    # for skel_pair in list(combinations(skel_paths, 2)):
    #     skel1 = load_ordered_mapped_skeleton(skel_pair[0])[0]
    #     skel2 = load_ordered_mapped_skeleton(skel_pair[1])[0]
    #     similar = similarity_check(skel1, skel2)
    #         unique_skel_paths.append(skel_pair[0])
    #     else:
    #         unique_skel_paths += [skel_pair[0], skel_pair[0]]
    return 0


def helper_write_dense_vol_skel_hull(path):
    min, max = get_dense_bounding_box()
    min *= np.array([9, 9, 20])
    max *= np.array([9, 9, 20])
    final_coords = []
    final_normals = []
    dir, fname = os.path.split(path)
    skel = load_mapped_skeleton(path, True, True)[0]
    for ii, coord in enumerate(skel._hull_coords):
        if np.any(coord < min) or np.any(coord > max):
            continue
        final_coords.append(coord / arr([9, 9, 20]))
        final_normals.append(skel._hull_normals[ii])
    skel._hull_coords = arr(final_coords)
    skel._hull_normals = arr(final_normals)
    hull_coords = get_dense_hull(skel)
    np.save(dir + '/hull_' + fname[:-5] + 'npy', np.array(hull_coords))
    # zf = zipfile.ZipFile(path, 'r')
    # data = np.fromstring(zf.read('hull_points.xyz'), sep=' ')
    # hull_normals = data.reshape(data.shape[0]/6, 6)[:, 3:]
    # hull_coords = (data.reshape(data.shape[0]/6, 6)[:, :3]).astype(np.int)
    # zf.close()
    # hull2text(np.array(hull_coords), np.zeros((len(hull_coords), 3)),
    #           dir + '/hull_' + fname[:-5] + 'xyz')
    # hull_coords -= hull_coords.min(axis=0)
    # helper_array = np.zeros(hull_coords.max(axis=0)+1)
    # for coord in hull_coords.tolist():
    #     helper_array[tuple(coord)] = 1
    # h5f = h5py.File(dir + '/dense_hull_shrinked_' + fname[:-5] + 'h5', 'w')
    # h5f.create_dataset('hull_shrinked', data=helper_array)


def get_dense_hull(skel):
    skel_hull = skel._hull_coords
    min_bb = np.min(skel_hull, axis=0)
    max_bb = np.max(skel_hull, axis=0)
    print "Computing dense hull for bounding box with volume", np.prod(max_bb-min_bb)
    dense_bb = np.mgrid[min_bb[0]:max_bb[0], min_bb[1]:max_bb[1],
               min_bb[2]:max_bb[2]].reshape(3,-1).T

    nb_voting_neighbors = 20
    tree = spatial.cKDTree(skel_hull)

    def check_hull_normals(obj_coord, hull_coords, dir_vecs):
        obj_coord = obj_coord[None, :]
        left_side = np.inner(obj_coord, dir_vecs)
        right_side = np.sum(dir_vecs * hull_coords, axis=1)
        sign = np.sign(left_side - right_side)
        return np.sum(sign) < 0

    dists, obj_lookup_IDs = tree.query(dense_bb, k=nb_voting_neighbors)
    annotated_hull_ixs = []
    for i in range(len(obj_lookup_IDs)):
        current_ids = obj_lookup_IDs[i]
        is_in_hull = check_hull_normals(dense_bb[i], skel_hull[current_ids],
                                        skel._hull_normals[current_ids])
        if is_in_hull:
            annotated_hull_ixs.append(i)
    dense_hull = dense_bb[annotated_hull_ixs]
    print "removed %d voxels from skeleton %s" % (len(dense_bb)-len(dense_hull),
                                                  skel.filename)
    return dense_hull


def get_dense_bounding_box():
    node_coords = []
    skel_paths = get_filepaths_from_dir('/lustre/pschuber/dense_vol_tracings/source/')
    for path in skel_paths:
        try:
            node_coords += [node.getCoordinate() for node in
                      au.loadj0126NML(path)[0].getNodes()]
        except Exception, e:
            print e
            print "Couldnt load", path
    return np.min(node_coords, axis=0), np.max(node_coords, axis=0)


def find_boundary_cs():
    dic = load_pkl2obj('/lustre/pschuber/dense_vol_tracings/nml_obj/'
                       'contact_sites_new/cs_dict_all.pkl')
    print "Found %d cs in total." % len(dic.keys())
    min, max = get_dense_bounding_box()
    cnt = 0
    for k, val in dic.iteritems():
        cc = val['center_coord']
        if np.any(cc < (min + np.array([10, 10, 5]))) or \
                np.any(cc > (max - np.array([10, 10, 5]))):
            print cc, k
            cnt += 1
    print cnt


def rewrite_task_numbers():
    skel_paths = get_filepaths_from_dir('/lustre/pschuber/consensi_fuer_joergen/consensi_fuer_phil/')
    dest_dir = '/lustre/pschuber/soma_tracings_done/'
    old_paths = get_filepaths_from_dir(dest_dir)
    for ii, orig_path in enumerate(skel_paths):
        dir, fname = os.path.split(orig_path)
        for old_path in old_paths:
            if 'task%d.' % ii in old_path:
                shutil.move(old_path, dest_dir+fname)
                print "Found", ii, "in", old_path
                print "From %s to %s" %(old_path, dest_dir+fname)
                break


def map_myelin2gt():
    annos = au.loadj0126NML('/lustre/pschuber/myelin_seeds.034.k.zip')
    dummy_skel = NewSkeleton()
    for i, anno in enumerate(annos):
        print "Skeleton", i
        mapped_skel = SkeletonMapper(anno, [9., 9., 20.], id=i)
        mapped_skel.calc_myelinisation()
        dummy_skel.add_annotation(mapped_skel.old_anno)
    dummy_skel.to_kzip('/lustre/pschuber/myelin_seeds_mapped.k.zip')


def test_new_cs_annotation():
    nml_list = get_filepaths_from_dir('/home/pschuber/Documents/MPI/'
                                      'dense_vol_tracings/nml_obj/')
    cs_path = '/home/pschuber/Documents/MPI/cs_test/'
    for ending in ['', 'cs', 'cs_p4', 'cs_az', 'cs_p4_az', 'pairwise',
                   'overlap_vx']:
        if not os.path.exists(cs_path+ending):
            os.makedirs(cs_path+ending)
    anno_permutations = list(combinations(nml_list, 2))
    prepare_syns_btw_annos(anno_permutations, cs_path)


def write_soma_to_mapped_skels():
    dest_dir = '/lustre/pschuber/mapped_soma_tracings/nml_obj/'
    soma_dir = '/lustre/pschuber/soma_tracings_done/'
    for p in get_filepaths_from_dir(dest_dir):
        skel, mito, p4, az, soma_old = load_ordered_mapped_skeleton(p)
        print "Found soma with %d nodes in mapped skel %s." % \
              (len(soma_old.getNodes()), skel.filename)
        skel_id = get_skelID_from_path(p)
        soma_skel_path = get_paths_of_skelID([str(skel_id)], traced_skel_dir=soma_dir)[0]
        _, _, _, _, soma = load_ordered_mapped_skeleton(soma_skel_path)
        if len(soma.getNodes()) == 0:
            print "Skipping %s beacuse soma is empty." % skel.filename
            continue
        if len(soma.getNodes()) == len(soma_old.getNodes()):
            continue
        soma.setComment("soma")
        print "Found corresponding soma %s with %d nodes. Adding it to mapped" \
              "skeleton." % (soma.getComment(), len(soma.getNodes()))
        dummy_skel = NewSkeleton()
        dummy_skel.add_annotation(skel)
        dummy_skel.add_annotation(mito)
        dummy_skel.add_annotation(p4)
        dummy_skel.add_annotation(az)
        dummy_skel.add_annotation(soma)
        dummy_skel.to_kzip(p)
