from itertools import combinations
from syconn.utils.newskeleton import NewSkeleton
from multi_proc import QSUB_MAIN as qm
from utils.datahandler import *
from contactsites import write_summaries
from processing.features import calc_prop_feat_dict
from processing.learning_rfc import write_feat2csv, load_rfcs,\
    start_multiprocess
from processing.mapper import SkeletonMapper, prepare_syns_btw_annos
__author__ = 'pschuber'
__QSUB__ = True


def QSUB_mapping(wd):
    """
    Run annotate annos on available cluster nodes defined by somaqnodes.
    :param wd: str Path to working directory
    """
    skel_dir = wd + '/tracings/'
    output_dir = wd + '/neurons/'
    anno_list = [os.path.join(skel_dir, f) for f in
                next(os.walk(skel_dir))[2] if 'k.zip' in f]
    np.random.shuffle(anno_list)
    print "Found %d cell tracings." % len(anno_list)
    list_of_lists = [[anno_list[i::60], output_dir] for i in xrange(60)]
    if __QSUB__:
        qm.QSUB_script(list_of_lists, 'skeleton_mapping', queue='somaqnodes',
                    work_folder="/home/pschuber/QSUB/", username="pschuber",
                    python_path="/home/pschuber/anaconda/bin/python",
                    path_to_scripts="/home/pschuber/skeleton-analysis/Philipp/QSUB")
    else:
        start_multiprocess(annotate_annos, list_of_lists, nb_cpus=1)


def annotate_annos(wd, anno_list, map_objects=True, method='hull', radius=1200,
                   thresh=2.2, filter_size=[2786, 1594, 250],
                   create_hull=True,
                   max_dist_mult=1.4, save_files=True, detect_outlier=True,
                   dh=None, overwrite=False, nb_neighbors=20,
                   nb_hull_vox=500, context_range=6000,
                   neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100,
                   write2pkl=False, write_obj_voxel=True, output_dir=None,
                   load_mapped_skeletons=True):
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
        rf_axoness_p = wd + '/models/rf_axoness/rf.pkl'
        rf_spiness_p = wd + '/models/rf_spiness/rf.pkl'
        if output_dir is None:
            if dh is None:
                raise RuntimeError("No output directory could be found.")
            else:
                output_dir = dh.data_path
        if not overwrite:
            existing_skel = [re.findall('[^/]+$', os.path.join(dp, f))[0] for
                             dp, dn, filenames in os.walk(output_dir + 'nml_obj/')
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
        todo_skel = list(set([re.findall('[^/]+$', f)[0] for f in anno_list])
                         - set(existing_skel))
        todo_skel = [f for f in anno_list if re.findall('[^/]+$', f)[0] in todo_skel]
        if len(todo_skel) == 0:
            print "Nothing to do. Aborting."
            return
        if dh is None:
            dh = DataHandler(wd)
            dh.data_path = output_dir
        print "Found %d processed Skeletons. %d left. Writing result to %s. Using" \
              " %s barrier." % (len(existing_skel), len(todo_skel), dh.data_path,
                                dh.mem_path)
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
            path = dh.data_path + 'nml_obj/' + re.findall('[^/]+$', filepath)[0]
            skel = create_skel(dh, annotation, id=id, soma=soma,
                               context_range=context_range)
            skel.write_obj_voxel = write_obj_voxel
            if create_hull:
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
            print "Starting cell compartment prediction."
            if rfc_spiness is not None:
                skel.predict_property(rfc_spiness, 'spiness')
            skel._property_features = None
            if rfc_spiness is not None and rfc_axoness is not None:
                skel.predict_property(rfc_axoness, 'axoness')
            if save_files:
                for folder in ['nml_obj/']:
                    if not os.path.exists(dh.data_path + folder):
                        os.makedirs(dh.data_path + folder)
                skel.write2kzip(path)
            if write2pkl:
                skel.write2pkl(dh.data_path + 'pkl/' +
                               re.findall('[^/]+$', filepath)[0])


def QSUB_remapping(anno_list=[], dest_dir=None, recalc_prop_only=False,
                   method='hull', dist=6000, supp=''):
    """
    Run annotate annos on available cluster nodes defined by somaqnodes.
    :param anno_list: str Paths to skeleton nml / kzip files
    :param dest_dir: str Directory path to store mapped skeletons
    """
    np.random.shuffle(anno_list)
    if dest_dir is not None and not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    print "Found %d mapped Skeletons. Remapping with context range of %d" % \
          (len(anno_list), dist)

    if __QSUB__:
        nb_processes = np.max((len(anno_list) / 3, 3))
        list_of_lists = [[anno_list[i::nb_processes], dest_dir, recalc_prop_only,
                          method, dist] for i in xrange(nb_processes)]
        qm.QSUB_script(list_of_lists, 'skeleton_remapping', queue='somaqnodes',
                       work_folder="/home/pschuber/QSUB/" + supp + "/",
                       username="pschuber",
                       python_path="/home/pschuber/anaconda/bin/python",
                       path_to_scripts="/home/pschuber/skeleton-analysis/Philipp/QSUB")
    else:
        start_multiprocess(remap_skeletons, [anno_list, dest_dir, recalc_prop_only,
                          method, dist], nb_cpus=1)


def remap_skeletons(wd, mapped_skel_paths=[], dh=None, method='hull', radius=1200,
                    thresh=2.2, filter_size=[2786, 1594, 250],
                    max_dist_mult=1.4,
                    save_files=True, nb_neighbors=20, nb_hull_vox=500,
                    neighbor_radius=220, nb_rays=20, nb_voting_neighbors=100,
                    output_dir=None, write_obj_voxel=True,
                    mito_min_votes=235, p4_min_votes=191, az_min_votes=346,
                    max_neck2endpoint_dist=3000, max_head2endpoint_dist=600,
                    recalc_prop_only=True, nb_cpus=4, context_range=6000):
    """
    Only remaps objects to skeleton without recalculating the hull.
    Min votes for cell objects are evaluated by f1-score using 500
    object hull voxels:
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
    rf_axoness_p = wd + '/models/rf_axoness/rf.pkl'
    rf_spiness_p = wd + '/models/rf_spiness/rf.pkl'
    rfc_axoness, rfc_spiness = load_rfcs(rf_axoness_p, rf_spiness_p)
    cnt = 0
    for skel_path in mapped_skel_paths:
        cnt += 1
        # get first element in list and only skeletonAnnotation
        mapped_skel_old, mito, p4, az, soma = \
            load_anno_list([skel_path], load_mitos=False)[0]
        if output_dir is not None:
            if not os.path.exists(output_dir):
                print "Couldn't find output directory. Creating", output_dir
                os.makedirs(output_dir)
            for fpath in mapped_skel_paths:
                shutil.copyfile(fpath, output_dir + os.path.split(fpath)[1])
            path = output_dir + re.findall('[^/]+$', skel_path)[0]
        else:
            path = skel_path
        try:
            id = int(re.findall('.*?([\d]+)', skel_path)[-3])
        except IndexError:
            id = cnt
        print "Remapping skeleton at %s and writing result to %s.\n" \
              "Using context range of %d and method '%s'" % (skel_path, path,
                                                             context_range,
                                                             method)
        new_skel = SkeletonMapper(mapped_skel_old, mapped_skel_old.scaling,
                                  id=id, soma=soma)
        new_skel.nb_cpus = nb_cpus
        if recalc_prop_only:
            print "--- Recalculating properties only ---"
            new_skel.old_anno.filename = skel_path
        else:
            if dh is None:
                dh = DataHandler(wd)
            new_skel.obj_min_votes['mitos'] = mito_min_votes
            new_skel.obj_min_votes['p4'] = p4_min_votes
            new_skel.obj_min_votes['az'] = az_min_votes
            new_skel.write_obj_voxel = write_obj_voxel
            new_skel.annotate_objects(
                dh, radius, method, thresh, filter_size,
                nb_hull_vox=nb_hull_vox,
                nb_voting_neighbors=nb_voting_neighbors, nb_rays=nb_rays,
                nb_neighbors=nb_neighbors, neighbor_radius=neighbor_radius,
                max_dist_mult=max_dist_mult)
        if rfc_spiness is not None:
            new_skel._property_features, new_skel.property_feat_names = \
                calc_prop_feat_dict(new_skel, context_range)
            new_skel.predict_property(rfc_spiness, 'spiness',
                                max_neck2endpoint_dist=max_neck2endpoint_dist,
                                max_head2endpoint_dist=max_head2endpoint_dist)
            new_skel._property_features = None
        if rfc_spiness is not None and rfc_axoness is not None:
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


def create_skel(dh, skel_source, id=None, soma=None, context_range=6000):
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
        assert skel_source < len(dh.skeleton_files)
        id = int(re.findall('.*?([\d]+)', dh.skeleton_files[skel_source])[-3])
        print "--- Initializing skeleton %d with %d cores from path." %\
              (id, dh.nb_cpus)
        if id in dh.skeletons.keys():
            skel = dh.skeletons[id]
            return skel
        print "Skeleton %d does not exist yet. Building from scratch." % id
        skel = SkeletonMapper(dh.skeleton_files[skel_source], dh.scaling,
                              soma=soma)
    else:
        print "--- Initializing skeleton %d with %d cores from annotation" \
              " object." % (id, dh.nb_cpus)
        if id in dh.skeletons.keys():
             return dh.skeletons[id]
        skel = SkeletonMapper(skel_source, dh.scaling, id=id, soma=soma,
                              context_range=context_range)
    skel._data_path = dh.data_path
    skel._mem_path = dh.mem_path
    skel.nb_cpus = dh.nb_cpus
    return skel


def QSUB_synapse_mapping(source_path, max_hull_dist=60):
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
    for ending in ['', 'cs', 'cs_p4', 'cs_az', 'cs_p4_az', 'pairwise','overlap_vx']:
        if not os.path.exists(cs_path+ending):
            os.makedirs(cs_path+ending)
    anno_permutations = list(combinations(nml_list, 2))
    if __QSUB__:
        def chunkify(lst, n):
            return [[list(lst[i::n]), cs_path, max_hull_dist] for i in xrange(n)]
        list_of_lists = chunkify(anno_permutations, 300)
        qm.QSUB_script(list_of_lists, 'synapse_mapping', queue='somaqnodes',
                    work_folder="/home/pschuber/QSUB/", username="pschuber",
                    python_path="/home/pschuber/anaconda/bin/python",
                    path_to_scripts="/home/pschuber/skeleton-analysis/Philipp/QSUB")
    else:
        start_multiprocess(prepare_syns_btw_annos, [anno_permutations, cs_path,
                                                    max_hull_dist], nb_cpus=1)
    write_summaries(cs_path)
