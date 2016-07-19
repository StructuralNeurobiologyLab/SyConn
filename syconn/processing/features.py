import copy
import numpy as np
import os
import zipfile
from collections import Counter
from numpy import array as arr
from scipy import spatial
import networkx as nx
from learning_rfc import write_feat2csv, cell_classification
from syconn.utils import skeleton_utils as su
from syconn.utils.basics import euclidian_distance
from syconn.utils.datahandler import load_objpkl_from_kzip, \
    load_ordered_mapped_skeleton
from syconn.utils.skeleton import remove_from_zip


def update_property_feat_kzip_star(args):
    """Helper function for update_property_feat_kzip
    """
    update_property_feat_kzip(*args)


def update_property_feat_kzip(path2kzip, dist=6000):
    """Recompute axoness feature of skeleton at path2kzip and writes it to
    .k.zip

    Parameters
    ----------
    path2kzip : str
        Path to mapped skeleton
    dist : int
    """
    prop_dict, property_feat_names = calc_prop_feat_dict(path2kzip, dist)
    for prop, feat in prop_dict.iteritems():
        path2csv = path2kzip[:-6] + '_%s_feat.csv' % prop
        write_feat2csv(path2csv, feat, property_feat_names[prop])
        file_name = '%s_feat.csv' % prop
        try:
            remove_from_zip(path2kzip, file_name)
            with zipfile.ZipFile(path2kzip, "a", zipfile.ZIP_DEFLATED) as zf:
                zf.write(path2csv, file_name)
            print "Wrote new %s feature to %s." % (prop, path2kzip)
        except Exception, e:
            print 'Could not write %s to zip file.' % file_name
            print e
        os.remove(path2csv)


def calc_prop_feat_dict(source, dist=6000):
    """Calculates property feature

    Parameters
    ----------
    source : SkeletonAnnotation
    dist : int

    Returns
    -------
    dict, list of str
    Dictionary of property features, list of feature names
    """
    print "Calculating morphological features with context range %d." % dist
    property_features = {}
    property_feat_names = {}
    morph_feat, spinehead_feats, node_ids = morphology_feature(source, dist)
    morph_info = np.concatenate((node_ids[:, None],
                                 morph_feat.astype(np.float32)), axis=1)
    property_features["axoness"] = np.concatenate((morph_info,
                                                  spinehead_feats), axis=1)
    morph_feat_names = ['nodeID', 'rad_mean', 'rad_std'] + \
                       ['rad_hist'+str(i) for i in range(10)] +\
                       ['mito_nb', 'mito_size_mean', 'vc_nb', 'vc_size_mean',
                        'sj_nb', 'sj_size_mean', 'branch_dist', 'endpoint_dist']
    property_features["spiness"] = morph_info
    property_feat_names["axoness"] = morph_feat_names +\
                                    ['nb_spinehead', 'sh_rad_mean',
                                    'sh_rad_std', 'sh_proba_mean']
    property_feat_names["spiness"] = morph_feat_names
    return property_features, property_feat_names


def morphology_feature(source, max_nn_dist=6000):
    """Calculates features for discrimination tasks of neurite identities, such as
    axon vs. dendrite or cell types classification. Estimated on interpolated
    skeleton nodes. Features are calculated with a sliding window approach for
    each node. Window is 2*max_nn_dist (nm).

    Parameters
    ----------
    source : str
        Path to anno or MappedSkeletonObject
    max_nn_dist : float
        Radius in which neighboring nodes are found and
        used for calculating features in nm.

    Returns
    -------
    numpy.array, numpary.array, list of int
        two arrays of features for each node. number of nodes x 28 (22 radius
        feature and 6 object features)
    """
    if isinstance(source, basestring):
        anno = load_ordered_mapped_skeleton(source)[0]
        # build mito sample tree
        mitos, vc, sj = load_objpkl_from_kzip(source)
    else:
        anno = source.old_anno
        if source.mitos is None:
            mitos, vc, sj = load_objpkl_from_kzip(anno.filename)
        else:
            mitos = source.mitos
            vc = source.vc
            sj = source.sj
    m_dict, vc_dict, sj_dict = (mitos.object_dict, vc.object_dict,
                                sj.object_dict)
    nearby_node_list = nodes_in_pathlength(anno, max_nn_dist)
    node_coords = []
    node_radii = []
    node_ids = []
    for nodes in nearby_node_list:
        node_coords.append(nodes[0].getCoordinate_scaled())
        node_radii.append(nodes[0].getDataElem("radius"))
        node_ids.append(nodes[0].getID())
    m_feat = objfeat2skelnode(node_coords, node_radii, node_ids,
                              nearby_node_list, m_dict, anno.scaling)
    vc_feat = objfeat2skelnode(node_coords, node_radii, node_ids,
                               nearby_node_list, vc_dict, anno.scaling)
    sj_feat = objfeat2skelnode(node_coords, node_radii, node_ids,
                               nearby_node_list, sj_dict, anno.scaling)
    rad_feat, spinehead_feat = radfeat2skelnode(nearby_node_list)
    morph_feat = np.concatenate((rad_feat, m_feat, vc_feat, sj_feat),
                                axis=1)
    dist_feature, ids = node_branch_end_distance(anno, max_nn_dist)
    sort_ix = np.argsort(ids)
    ids = ids[sort_ix]
    dist_feature = dist_feature[sort_ix]
    node_ids = arr(node_ids)
    sort_ix2 = np.argsort(node_ids)
    node_ids = node_ids[sort_ix2]
    morph_feat = morph_feat[sort_ix2]
    assert np.all(node_ids == ids), 'Node IDs are different.'
    morph_feat = np.concatenate((morph_feat, dist_feature), axis=1)
    if np.any(np.isnan(morph_feat)):
        print "Found nans in morphological features of %s: %s" % \
              (source, np.where(np.isnan(morph_feat)))
        morph_feat = np.nan_to_num(morph_feat.astype(np.float32))
    spinehead_feat = spinehead_feat[sort_ix2]
    if np.any(np.isnan(spinehead_feat)):
        print "Found nans in spinhead features of %s: %s" % \
              (source, np.where(np.isnan(spinehead_feat)))
        spinehead_feat = np.nan_to_num(spinehead_feat.astype(np.float32))
    return morph_feat, spinehead_feat, ids


def radfeat2skelnode(nearby_node_list):
    """Calculate nodewise radius feature

    Parameters
    ----------
    nearby_node_list : list of list of SkeletonNodes
        grouped tracing nodes

    Returns
    -------
        array of number of nodes times 22 features, containing mean radius,
        sigma of radii, 20 hist features
    """
    radius_feat = np.zeros((len(nearby_node_list), 12))
    spiness_given = True
    try:
        _ = nearby_node_list[0][0].data["spiness_pred"]
    except KeyError:
        print "Spiness prediction not found in Annotation Object. Axoness" \
              "feature reduced to morphological features."
        spiness_given = False
    spinehead_feats = np.zeros((len(nearby_node_list), 4))
    for k, neighbor_nodes in enumerate(nearby_node_list):
        radius_feat[k] = radius_feats_from_nodes(neighbor_nodes)
        if spiness_given:
            spinehead_feats[k] = spiness_feats_from_nodes(neighbor_nodes)[:4]
    return radius_feat, spinehead_feats


def radius_feats_from_nodes(nodes, nb_bins=10, max_rad=5000):
    """Calculates mean, std and histogram features

    Parameters
    ----------
    nodes : list of SkeletonNodes
    nb_bins : int
        Number of bins for histogram features
    max_rad : int
        maximum radius to plot on histogram x-axis

    Returns
    -------
    np.array
        radius features with dim. nb_bins+2
    """
    radius_feat = np.zeros((12))
    nn_radius = arr([nn.getDataElem("radius") for nn in nodes])
    radius_feat[0] = np.mean(nn_radius)
    radius_feat[1] = np.std(nn_radius)
    hist, bin_edges = np.histogram(nn_radius*10, bins=nb_bins, range=(0, max_rad),
                                   normed=True)
    radius_feat[2:(nb_bins+2)] = hist
    return radius_feat


def spiness_feats_from_nodes(nodes):
    """
    Calculates spiness feats including abs. number of spineheads, mean and
    standard deviation (std) of spinehead size, mean spinehead probability and
    mean and std of spineneck lengths.
    :param nodes: list of SkeletonNodes
    :return: np.array of spiness features, dim. of 6
    """
    spinehead_feats = np.zeros((6))
    spinehead_radius = []
    spinehead_proba = []
    spine_neck_lengths = []
    for node in nodes:
        if node.degree() == 1 and int(node.data["spiness_pred"]) == 1:
            spinehead_proba.append(float(node.data["spiness_proba1"]))
            spinehead_radius.append(node.data["radius"])
            node1 = node
            neck_length = 0
            while (len(node1.getParents()) == 1) and (int(node1.data\
            ["spiness_pred"]) != 2):
                neck_length += euclidian_distance(node1.getCoordinate_scaled(),
                               list(node1.getParents())[0].getCoordinate_scaled())
                node1 = list(node1.getParents())[0]
            spine_neck_lengths.append(neck_length)
    if len(spinehead_radius) != 0:
        spinehead_mean = np.mean(spinehead_radius)
        spinhead_std = np.std(spinehead_radius)
        spinehead_proba_mean = np.mean(spinehead_proba)
        spinehead_feats[:4] = arr([len(spinehead_radius), spinehead_mean,
                               spinhead_std, spinehead_proba_mean])
    if len(spine_neck_lengths) != 0:
        spinehead_feats[4:6] = arr([np.mean(spine_neck_lengths),
                                   np.std(spine_neck_lengths)])
    return spinehead_feats


def sj_per_spinehead(anno):
    """
    Calculate number of sj per spinehead. Iterate over all mapped sj objects and
    find nearest skeleton node. If skeleton node has spiness prediction == 1
    (spinehead) then increment counter of this node by one.
    After the loop sum over all counter and divide by the number of nodes which
    have at least one sj assigned.
    :param anno: SkeletonAnnotation
    :return: Average number of sj per spinehead (assumes there is no spinehead
    without sj)
    """
    _, _, sj = load_objpkl_from_kzip(anno.filename)
    sj_dict = sj.object_dict
    nb_sj = len(sj_dict.keys())
    hull_samples = np.zeros((nb_sj, 100, 3))
    skel_nodes = [n for n in anno.getNodes()]
    node_coords = arr([n.getCoordinate_scaled() for n in skel_nodes])
    node_sj_counter = np.zeros((len(skel_nodes), ))
    skeleton_tree = spatial.cKDTree(node_coords)
    for i, sj_key in enumerate(sj_dict.keys()):
        sj = sj_dict[sj_key]
        m_hull = sj.hull_voxels * arr(anno.scaling)
        random_ixs = np.random.choice(np.arange(len(m_hull)), size=100)
        hull_samples[i] = m_hull[random_ixs]
    for i in range(nb_sj):
        dists, nearest_skel_ixs = skeleton_tree.query(hull_samples[i], 1)
        majority_ix = cell_classification(nearest_skel_ixs)
        if int(skel_nodes[majority_ix].data["spiness_pred"]) == 1:
            node_sj_counter[majority_ix] += 1
    sj_per_sh = np.sum(node_sj_counter) / float(np.sum(node_sj_counter != 0))
    return np.nan_to_num(sj_per_sh)


def propertyfeat2skelnode(node_list):
    """
    Calculate nodewise radius feature.
    :param node_list: list of grouped nodes
    :return: array of number of nodes times 22 features, containing mean radius,
    sigma of radii, 20 hist features
    """
    radius_feat = np.zeros((1, 12))
    n_radius = []
    type_feat = np.zeros((1, 6))
    n_axoness = []
    for node in node_list:
        n_radius.append(float(node.getDataElem("radius")))
        n_axoness.append(int(node.data["axoness_pred"]))
    n_radius = arr(n_radius)
    radius_feat[0, 0] = np.mean(n_radius)
    radius_feat[0, 1] = np.std(n_radius)
    hist, bin_edges = np.histogram(n_radius*10, bins=10, range=(0, 5000),
                                   normed=True)
    radius_feat[0, 2:12] = hist
    n_axoness = arr(n_axoness)
    nb_axonnodes = np.sum(n_axoness == 1)
    nb_dennodes = np.sum(n_axoness == 0)
    nb_somanodes = np.sum(n_axoness == 2)
    type_feat[0, :3] = [nb_axonnodes, nb_dennodes, nb_somanodes]
    type_feat[0, 3:] = type_feat[0, :3] / float(len(n_axoness))
    return radius_feat, type_feat


def celltype_axoness_feature(anno):
    """
    Calculates axones feature of mapped sekeleton for cell type prediction.
    These include proportion of axon, dendrite and soma pathlengths and maximum
    degree of soma nodes.

    Returns
    -------
    np.array (n x 4)
        axoness features
    """
    type_feats = np.zeros((1, 4))
    all_path_length = anno.physical_length() / 1000.
    for i in range(3):
        type_feats[0, i] = pathlength_of_property(anno, 'axoness_pred', i) / \
                        all_path_length
    for n in anno.getNodes():
        if int(n.data["axoness_pred"]) == 2:
            type_feats[0, 3] = np.max((n.degree(), type_feats[0, 3]))
    return type_feats


def pathlength_of_property(anno, property, value):
    """Calculate pathlength of nodes with certain property value

    Parameters
    ----------
    anno : SkeletonAnnotation
        mapped cell tracing
    property : str
        spiness / axoness
    value : int
        classification result, e.g. 0, 1, 2

    Returns
    -------
    int
        length (in um)
    """
    pathlength = 0
    for from_node, to_node in anno.iter_edges():
        if int(from_node.data[property]) == value and\
                        int(to_node.data[property]) == value:
            pathlength += euclidian_distance(from_node.getCoordinate_scaled(),
                                             to_node.getCoordinate_scaled())
    return pathlength / 1000.


def objfeat2skelnode(node_coords, node_radii, node_ids, nearby_node_list,
                     obj_dict, scaling):
    """
    Calculate features of SegmentationDatasetObjects along Skeleton.
    :return: array of dimension nb_skelnodes x 2. The two features are:
    absolute number of assigned objects and mean voxel size of the objects
    """
    skeleton_tree = spatial.cKDTree(node_coords)
    nb_skelnodes = len(node_coords)
    axoness_features = np.zeros((nb_skelnodes, 2))
    obj_assignment = [[] for i in range(nb_skelnodes)]
    nb_objs = len(obj_dict.keys())
    hull_samples = np.zeros((nb_objs, 100, 3))
    key_list = []
    for i, obj_key in enumerate(obj_dict.keys()):
        obj_object = obj_dict[obj_key]
        m_hull = obj_object.hull_voxels * scaling
        random_ixs = np.random.choice(np.arange(len(m_hull)), size=100)
        hull_samples[i] = m_hull[random_ixs]
        key_list.append(obj_key)
    for i in range(nb_objs):
        dists, nearest_skel_ixs = skeleton_tree.query(hull_samples[i], 1)
        for ix in list(set(nearest_skel_ixs)):
            if np.min(dists[nearest_skel_ixs == ix]) > node_radii[ix]*10:
                continue
            obj_assignment[ix] += [key_list[i]]
    for k in range(nb_skelnodes):
        nn_nodes = nearby_node_list[k]
        nn_ids = [nn.getID() for nn in nn_nodes]
        assigned_objs = []
        for nn_id in nn_ids:
            assigned_objs += obj_assignment[node_ids.index(nn_id)]
        axoness_features[k, 0] = len(assigned_objs)
        if len(assigned_objs) == 0:
            continue
        obj_sizes = [obj_dict[key].size for key in assigned_objs]
        axoness_features[k, 1] = np.mean(obj_sizes)
    return axoness_features


def nodes_in_pathlength(anno, max_path_len):
    """
    Find nodes reachable in max_path_len from source node, calculated for
    every node in anno.
    :param anno: AnnotationObject
    :param max_path_len: float Maximum distance from source node
    :return: list of lists containing reachable nodes in max_path_len where
    outer list has length len(anno.getNodes())
    """
    skel_graph = su.annotation_to_nx_graph(anno)
    list_reachable_nodes = []
    for source_node in anno.getNodes():
        source_node_coord = arr(source_node.getCoordinate_scaled())
        reachable_nodes = [source_node]
        for edge in nx.bfs_edges(skel_graph, source_node):
            next_node = edge[1]
            next_node_coord = arr(next_node.getCoordinate_scaled())
            if np.linalg.norm(next_node_coord - source_node_coord) > max_path_len:
                break
            reachable_nodes.append(next_node)
        list_reachable_nodes.append(reachable_nodes)
    return list_reachable_nodes


def assign_property2node(node, pred, property):
    """
    Assign prediction of property to node
    :param node: NewSkeletonNode
    :param pred: prediction appropriate to property
    :param property: property to change
    :return:
    """
    node.data["%s_pred" % property] = str(pred)
    node_comment = node.getComment()
    ax_ix = node_comment.find(property)
    if ax_ix == -1:
        node.appendComment(property+str(pred))
    else:
        help_list = list(node_comment)
        help_list[ax_ix+7] = str(pred)
        node.setComment("".join(help_list))


def majority_vote(anno, property='axoness', max_dist=5000):
    """
    Smoothes property prediction of annotation using running average with path
    length 2 * max_length (nm).
    :param anno:
    :param property:
    :param max_dist:
    :return:
    """
    print "Performing smoothing of %s using sliding window average of max " \
          "dist %d nm." % (property, max_dist)
    old_anno = copy.deepcopy(anno)
    nearest_nodes_list = nodes_in_pathlength(old_anno, max_dist)
    for nodes in nearest_nodes_list:
        curr_node_id = nodes[0].getID()
        new_node = anno.getNodeByID(curr_node_id)
        if int(new_node.data["axoness_pred"]) == 2:
            new_node.data["axoness_pred"] = 2
            continue
        # property_val = [int(re.findall(property+'(\d+)', n.getComment())[0])
        #            for n in nodes]
        property_val = [int(n.data[property+'_pred']) for n in nodes]
        # print "Using %d nodes for %s majority voting" % (len(property_val),
        #                                                  property)
        counter = Counter(property_val)
        new_ax = counter.most_common()[0][0]
        node_comment = new_node.getComment()
        ax_ix = node_comment.find(property)
        help_list = list(node_comment)
        help_list[ax_ix+len(property)] = str(new_ax)
        new_node.setComment("".join(help_list))
        new_node.setDataElem(property+'_pred', new_ax)


def get_obj_density(source, property='axoness_pred', value=1, obj='mito',
                    return_abs_density=True):
    """
    Calculate pathlength of nodes using edges.
    :param anno: list of SkeletonAnnotation
    :return: length in um
    """
    obj_dict = {'mito': 0, 'vc': 1, 'sj':2}
    if isinstance(source, basestring):
        anno = load_ordered_mapped_skeleton(source)[0]
        # build mito sample tree
        mitos, vc, sj = load_objpkl_from_kzip(source)
    else:
        anno = source.old_anno
        if source.mitos is None:
            mitos, vc, sj = load_objpkl_from_kzip(anno.filename)
        else:
            mitos = source.mitos
            vc = source.vc
            sj = source.sj
    m_dict, vc_dict, sj_dict = (mitos.object_dict, vc.object_dict,
                                sj.object_dict)
    obj_dict = [m_dict, vc_dict, sj_dict][obj_dict[obj]]
    node_coords = []
    node_radii = []
    node_ids = []
    for node in anno.getNodes():
        node_coords.append(node.getCoordinate_scaled())
        node_radii.append(node.getDataElem("radius"))
        node_ids.append(node.getID())
    pathlength = 0
    nodes_of_value = []
    for from_node, to_node in anno.iter_edges():
        if int(from_node.data[property]) == value:
            nodes_of_value.append(from_node.getID())
        if int(from_node.data[property]) == value and\
        int(to_node.data[property]) == value:
            pathlength += euclidian_distance(from_node.getCoordinate_scaled(),
                                             to_node.getCoordinate_scaled())
    skeleton_tree = spatial.cKDTree(node_coords)
    nb_skelnodes = len(node_coords)
    obj_assignment = [[] for i in range(nb_skelnodes)]
    nb_objs = len(obj_dict.keys())
    hull_samples = np.zeros((nb_objs, 500, 3))
    key_list = []
    for i, obj_key in enumerate(obj_dict.keys()):
        obj_object = obj_dict[obj_key]
        m_hull = obj_object.hull_voxels * anno.scaling
        random_ixs = np.random.choice(np.arange(len(m_hull)), size=500)
        hull_samples[i] = m_hull[random_ixs]
        key_list.append(obj_key)
    for i in range(nb_objs):
        dists, nearest_skel_ixs = skeleton_tree.query(hull_samples[i], 1)
        for ix in list(set(nearest_skel_ixs)):
            if np.min(dists[nearest_skel_ixs == ix]) > node_radii[ix]*10:
                continue
            obj_assignment[ix] += [key_list[i]]
    assigned_objs = []
    for k in range(nb_skelnodes):
        if not node_ids[k] in nodes_of_value:
            continue
        assigned_objs += obj_assignment[node_ids.index(node_ids[k])]
    assigned_objs = list(set(assigned_objs))
    if pathlength == 0:
        return 0
    if return_abs_density:
        return len(assigned_objs) / pathlength * 1000.
    obj_vols = []
    for key in assigned_objs:
        obj_vols.append(obj_dict[key].size * (9*9*20))
    obj_density = np.sum(obj_vols) / pathlength * 1000.
    return obj_density


def node_branch_end_distance(nml, dist):
    graph = su.annotation_to_nx_graph(nml)
    dic = su.nx.degree(graph)

    end = []
    for key, value in dic.items():
        if value == 1:
            end.append(key)

    bran = []
    for key, value in dic.items():
        if value >= 3:
            bran.append(key)

    features = []
    Y = []
    for node in graph.nodes():
        Y.append(node.getID())
        node_to_all_endnode = [dist]
        node_to_all_branchpoint = [dist]
        single_node_feature = []
        for endnode in end:
            node_to_all_endnode.append(node.distance_scaled(endnode))

        if len(node_to_all_endnode) != 0:
            distance2endpoint = min(node_to_all_endnode)
        else:
            distance2endpoint = np.float32(99999999)

        # distance2endpoint = min(node_to_all_endnode)
        node.data["endpointdistance"] = distance2endpoint
        single_node_feature.append(distance2endpoint)

        for branchpoint in bran:
            node_to_all_branchpoint.append(node.distance_scaled(branchpoint))
        if len(node_to_all_branchpoint) != 0:
            distance2branchpoint = min(node_to_all_branchpoint)
        else:
            distance2branchpoint = np.float32(99999999)
        single_node_feature.append(distance2branchpoint)
        features.append(single_node_feature)
        node.data["branchpointdistance"] = distance2branchpoint
    X = np.array(features)
    print "Max occuring distance:", np.max(features)
    Y = np.array(Y)
    return X, Y
