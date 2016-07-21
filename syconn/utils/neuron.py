# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import copy
import itertools
import numpy as np
import os
from scipy import spatial

import networkx as nx

import syconn.utils.skeleton_utils as su
from syconn.utils.skeleton import Skeleton
from syconn.processing.features import celltype_axoness_feature,\
    spiness_feats_from_nodes,  radius_feats_from_nodes, sj_per_spinehead,\
    pathlength_of_property
from syconn.processing.learning_rfc import cell_classification
from syconn.processing.synapticity import syn_sign_prediction
from syconn.utils.basics import convex_hull_area
from syconn.utils.datahandler import load_objpkl_from_kzip


class Neuron(object):
    """Class to calculate cell type features, based on results of
    SkeletonMapper

    Attributes
    ----------
    neuron_feature_names : list of str
        order names of features
    features : dict of np.array
        cell type features stored with their feature names
    """
    def __init__(self, annotations, unique_ID=None):

        # This identifier is always unique. If another neuron object is
        # discovered to be the same biological unit, both need to be merged
        # immediately.
        self._ID = unique_ID

        self.annotations = annotations
        # scan for consensus annotations

        if type(annotations) != list:
            self.annotations = [annotations]

        self.consensus_annotations = []
        self.features = dict()
        self._feature_name_dict = dict()
        self.neuron_feature_names = []

        for anno in self.annotations:
            if 'onsensu' in anno.filename \
                    and not 'training' in anno.filename \
                    and 'iter' in anno.filename:
                if 'skeleton' in anno.comment:
                    self.consensus_annotations.append(anno)

        self._set_annotation_neuron_IDs()

        self.nx_g = None

        # neurons that are postsynaptic to this neuron
        self.post_synapses = []

        # neurons that are presynaptic to this neuron
        self.pre_synapses = []

        # synapses (pre - and post synapses)
        self.synapses = []

        # not yet done - annotation objs (?) that represent the part of the
        # neuron
        self.axon = None
        self.dendrite = None
        self.soma = None

        self.tags = set()
        self.tag_src_coords = []
        self.comment = ''
        self._mapped_skel_dir = '/lustre/pschuber/m_consensi_rr/nml_obj/'

        # These values are for now calculated on the longest annotation of
        # all annotations that belong to the Neuron object. Ideally,
        # each neuron should be a single connected-component with only one
        # annotation object or even no annotation objects anymore if the
        # representation of the graph is finally saved as something much more
        # efficient (eg numpy matrix). For now, to get things done,
        # annotations stay first class citizens, even with making pickling
        # super slow
        self.all_path_length_um = 0.0
        self.num_all_branch_p = 0
        self.all_branch_density = 0.0
        self.num_pruned_branch_p = 0
        self.all_branch_density = 0.0
        self.pruned_branch_density = 0.0
        self.pruned_path_length_um = 0.0
        self._neuron_features = np.zeros((1, 0))

        # It is possible that tasks with different IDs belong to the same
        # biological neuron
        self.same_task_IDs = set()
        # list of all files used to generate this object
        self.source_files = []

    def add_synapse(self, syn):
        self.synapses.append(syn)
        if syn.pre_neuron == self:
            self.pre_synapses.append(syn)
        elif syn.post_neuron == self:
            self.post_synapses.append(syn)
        else:
            raise Exception('Synapse neuron does not belong to neuron.')
            # syn.neuron = self

    @property
    def skel_src_files(self):
        if self.annotations:
            return [a.filename for a in self.annotations]
        else:
            return []

    @property
    def num_synapses(self):
        return len(self.synapses)

    @property
    def avg_syn_sj_len(self):
        return np.mean(np.array([s.sj_len for s in self.synapses]))

    @property
    def syn_h_s_ratio(self):
        num_heads = float(len([syn for syn in self.synapses if syn.loc_vote ==
                               'spine-head']))
        num_shafts = float(len([syn for syn in self.synapses if syn.loc_vote ==
                                'shaft']))
        try:
            ratio = num_heads / (num_shafts + num_heads)
        except ZeroDivisionError:
            ratio = 1.
        return ratio

    @property
    def syn_type_voted(self):
        try:
            voted_types = [syn.type_vote for syn in self.synapses]
        except:
            return None

        return max(set(voted_types), key=voted_types.count)

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, value):
        self._ID = value
        self._set_annotation_neuron_IDs()

    @property
    def neuron_features(self):
        """Setter for cellt type features

        Returns
        -------
        np.array
            cell type features
        """
        if self._neuron_features.shape[1] == 0:
            self.neuron_feature_names = np.zeros((1, 0))
            self.set_statistics()
            feature_names = self.features.keys()
            feature_names.sort()
            for k in feature_names:
                feat = self.features[k]
                if (not isinstance(feat, list)) & \
                        (not isinstance(feat, np.ndarray)):
                    feat = np.array([feat]).reshape((1, 1))
                if feat.shape[1] != 1:
                    self.neuron_feature_names = np.concatenate((
                        self.neuron_feature_names, self._feature_name_dict[k]),
                        axis=1)
                else:
                    self.neuron_feature_names = np.concatenate((
                        self.neuron_feature_names, np.array([k])[None, :]),
                        axis=1)
                self._neuron_features = np.concatenate((self._neuron_features,
                                                        feat), 1)
            self.neuron_feature_names = self.neuron_feature_names[0]
        return self._neuron_features

    def to_skeleton_obj(self, only_cons=False):
        if self.annotations:
            skel_obj = Skeleton()
            if self.consensus_annotations and only_cons:
                for cons in self.consensus_annotations:
                    skel_obj.add_annotation(cons)
            else:
                for anno in self.annotations:
                    skel_obj.add_annotation(anno)
            return skel_obj
        else:
            return None

    def _set_annotation_neuron_IDs(self):
        if self.annotations:
            neuron_annos = []
            for anno in self.annotations:
                anno.neuron_ID = self._ID  # having neuron and the ID is
                # somewhat redundant, this should be thought of
                anno.neuron = self
                neuron_annos.append(anno)
            self.annotations = neuron_annos

    def set_statistics(self):
        """Setter for cell type features
        """
        if len(self.consensus_annotations) > 1:
            for a in self.consensus_annotations:
                if 'skeleton' in a.comment:
                    anno_to_use = a
        else:
            if self.annotations:
                anno_to_use = self.annotations[0]

        # for now, the first annotation is used if multiples are present
        self.nx_g = su.annotation_to_nx_graph(anno_to_use)
        self.features[
            'all_path_length_um'] = anno_to_use.physical_length() / 1000.
        self.features['num_all_branch_p'] = num_branch_points_of_nx_graph(
            self.nx_g)
        self.features['all_branch_density'] = 0.
        self.features['num_all_end_p'] = num_end_points_of_nx_graph(self.nx_g)
        self.features['tortuosity'] = calc_arc_choord(anno_to_use, self.nx_g)

        if os.path.isfile(self._mapped_skel_dir + anno_to_use.filename) or \
                os.path.isfile(anno_to_use.filename):
            object_feats, rad_feats, type_feats, spine_feats = \
                cell_morph_properties(anno_to_use)
            self.features['obj_morphology'] = object_feats  # 9
            self.features['rad_morphology'] = rad_feats  # 14
            self.features['type_morphology'] = type_feats  # 4
            self.features['spine_morphology'] = spine_feats  # 13
            self.features['synapse_type'] = calc_syn_type_feats(
                anno_to_use)  # 8
            self.features['mito density'] = 0.
            self.features['vc density'] = 0.
            self.features['sj density'] = 0.

            if self.features['all_path_length_um'] > 0.0:
                self.features['mito density'] = self.features['obj_morphology']\
                                                    [0, 2] / self.features[
                                                    'all_path_length_um']
                self.features['vc density'] = self.features['obj_morphology']\
                        [0, 5] / self.features['all_path_length_um']
                self.features['sj density'] = self.features['obj_morphology']\
                                                  [0, 8] / self.features[
                                                  'all_path_length_um']
            self._feature_name_dict['obj_morphology'] = \
                np.array(['mean size of mito', 'std of mito size',
                          'abs. number of mitos',
                          'mean size of vc', 'std of vc size',
                          'abs. number of vc',
                          'mean size of sj', 'std of sj size',
                          'abs. number if sj'])[None, :]
            self._feature_name_dict['rad_morphology'] = \
                np.array(['mean radius (dend.)', 'std of radius (dend.)'] +
                         ['bin %d of radius hist. (dend.)' % nb for nb in
                          range(4)] +
                         ['mean radius (axon)', 'std of radius (axon)'] +
                         ['bin %d of radius hist. (axon)' % nb for nb in
                          range(4)] +
                         ['mean radius (soma)', 'std of radius (soma)'])[None,
                :]
            self._feature_name_dict['type_morphology'] = \
                np.array(['prop. of dendrite pathlength',
                          'prop. of axon pathlength',
                          'prop. of soma pathlength',
                          'soma branches'])[None, :]
            self._feature_name_dict['spine_morphology'] = \
                np.array(['sh density (dend.)', 'mean radius (dend.)',
                          'std of radius (dend.)',
                          'mean of sh probability (dend.)',
                          'mean neck length (dend.)',
                          'std neck length (dend.)'] +
                         ['sh density (axon)', 'mean radius (axon)',
                          'std of radius (axon)',
                          'mean of sh probability (axon)',
                          'mean neck length (axon)', 'std neck length (axon)',
                          'number sj per sh'])[None, :]
            self._feature_name_dict['synapse_type'] = np.array(
                ['outgoing syn. type', 'proportion of inc. syn. type',
                 'median of outgoing syn. size', 'std of outgoing syn. size',
                 'median of inc. asym. size',
                 'std of inc. asym. size', 'median of inc. sym. size',
                 'std of inc. sym. size'])[None, :]

        if self.features['all_path_length_um'] > 0.0:
            self.features['all_branch_density'] = \
                self.features['num_all_branch_p'] / self.features[
                    'all_path_length_um']
            self.features['all_end_density'] = \
                self.features['num_all_end_p'] / self.features[
                    'all_path_length_um']


def num_end_points_of_nx_graph(nx_graph):
    try:
        num = len(list({k for k, v in nx_graph.degree().iteritems() if v == 1}))
    except Exception, e:
        print e
        print "Got exception during number end point calculation. Setting to 0."
    return num


def cell_morph_properties(mapped_annotation):
    """
    Get cell object properties based on mapped annotation object. Find
    total number and size (mean and std) of cell objects.

    Parameters
    ----------
    mapped_annotation : SkeletonAnnotation

    Returns
    -------
    np.array, np.array, np.array, np.array
        Cell object features (1 x 9), radius features (1 x 14), synapse type
        features (1 x 8), spiness features (1 x 13)
    """
    skel_nodes = np.array(list(mapped_annotation.getNodes()))
    ax_preds = np.zeros((len(skel_nodes)), dtype=np.uint16)
    type_rad_ranges = [1000, 500, 1500]
    for ii, node in enumerate(skel_nodes):
        ax_preds[ii] = int(node.data["axoness_pred"])
    rad_feats = np.zeros((1, 14))
    spiness_feats = np.zeros((1, 13))
    for i in range(3):
        ix_begin = i * 6
        ix_end = (i + 1) * 6
        ax_nodes = skel_nodes[ax_preds == i]
        if len(ax_nodes) == 0:
            continue
        if i == 2:
            rad_feats[0, ix_begin:] = radius_feats_from_nodes(
                ax_nodes, nb_bins=10, max_rad=type_rad_ranges[i])[:2]
        else:
            rad_feats[0, ix_begin:ix_end] = radius_feats_from_nodes(
                ax_nodes, nb_bins=10, max_rad=type_rad_ranges[i])[:6]
        if i != 2:
            spiness_feats[0, ix_begin:ix_end] = spiness_feats_from_nodes(
                ax_nodes)
    spiness_feats[0, 12] = sj_per_spinehead(mapped_annotation)
    dend_length = pathlength_of_property(mapped_annotation, 'axoness_pred', 0)
    if dend_length != 0:
        spiness_feats[0, 0] /= dend_length
    type_feats = celltype_axoness_feature(mapped_annotation)
    object_feats = calc_obj_feat(mapped_annotation)
    return object_feats, rad_feats, type_feats, spiness_feats


def calc_obj_feat(mapped_annotation):
    """
    Calculates object features. mapped_annotation needs its absolute
    path in its attribute 'filename'

    Parameters
    ----------
    mapped_annotation : SkeletonAnnotation

    Returns
    -------
    np.array
        object features of dimension (1, 9)
    """
    cell_objects = load_objpkl_from_kzip(mapped_annotation.filename)
    object_features = []
    for obj_type in cell_objects:
        obj_sizes = [obj_type.object_dict[key].size for key in
                     obj_type.object_dict.keys()]
        mean_size = np.mean(obj_sizes)
        std_size = np.std(obj_sizes)
        nb_obj = len(obj_sizes)
        object_features.append((mean_size, std_size, nb_obj))
    object_feats = np.nan_to_num(np.array(object_features)).reshape((1, 9))
    return object_feats


def calc_syn_type_feats(anno_to_use):
    """Calculate cell feature based on mapped synapses

    Parameters
    ----------
    anno_to_use : SkeletonAnnotation

    Returns
    -------
    np.array
        synapse type features of dimension (1, 8)
    """
    syn_type_feats = np.zeros((1, 8))
    syn_type_feats[0, 0] = -1
    syn_type_feats[0, 1] = -1
    syn_types_in = []
    syn_types_out = []
    outgoing_syn_size = []
    incoming_syn_size = []

    node_list = np.array([n for n in anno_to_use.getNodes()])
    coord_list = [n.getCoordinate_scaled() for n in node_list]
    skel_tree = spatial.cKDTree(coord_list)
    for ix in set(anno_to_use.sj_hull_ids):
        ix_bool_arr = anno_to_use.sj_hull_ids == ix
        obj_hull = anno_to_use.sj_hull_coords[ix_bool_arr]
        hull_com = np.mean(obj_hull, axis=0)
        dists, close_ixs = skel_tree.query([hull_com],
                                           k=np.min((3, len(node_list))))
        near_nodes = node_list[close_ixs]
        axoness = cell_classification([int(n.data["axoness_pred"]) for n in
                                       near_nodes[0]])
        syn_type_pred = syn_sign_prediction(obj_hull / np.array([9., 9., 20.]))
        sj_area = convex_hull_area(obj_hull) / 2.e6
        if axoness == 1:
            outgoing_syn_size.append(sj_area)
            syn_types_out.append(syn_type_pred)
        else:
            incoming_syn_size.append(sj_area)
            syn_types_in.append(syn_type_pred)
    if len(syn_types_out) != 0:
        syn_type_feats[0, 0] = cell_classification(syn_types_out)
    if len(syn_types_in) != 0:
        syn_type_feats[0, 1] = np.mean(syn_types_in)
    if len(outgoing_syn_size) != 0:
        syn_type_feats[0, 2] = np.sum(outgoing_syn_size)
        syn_type_feats[0, 3] = np.median(outgoing_syn_size)
    if len(incoming_syn_size) != 0:
        incoming_syn_size = np.array(incoming_syn_size)
        incoming_type = np.array(syn_types_in)
        if np.any(incoming_type == 0):
            syn_type_feats[0, 4] = np.median(
                incoming_syn_size[incoming_type == 0])
            syn_type_feats[0, 5] = np.std(incoming_syn_size[incoming_type == 0])
        if np.any(incoming_type == 1):
            syn_type_feats[0, 6] = np.median(
                incoming_syn_size[incoming_type == 1])
            syn_type_feats[0, 7] = np.std(incoming_syn_size[incoming_type == 1])
    return syn_type_feats


def num_branch_points_of_nx_graph(nx_graph):
    try:
        num = len(list({k for k, v in nx_graph.degree().iteritems() if v > 2}))
    except Exception, e:
        print e
        print "Got exception during number branch point calc. Setting to 0."
        num = 0
    return num


def get_annotation_branch_lengths(annotation):
    """
    Fragments an annotation into pieces at every branch point - remaining
    lonely nodes are deleted. Branch nodes are simply deleted. The lengths of
    the resulting fragments are then returned.

    WARNING: THIS FUNCTION IS REALLY SLOW FOR SOME REASONS I DO NOT FULLY
    UNDERSTAND AT THE MOMENT, PROBABLY OBJECT COPY RELATED

    :param annotation:
    :return: list of branch lengths in um
    """

    # this is necessary to avoid trouble because of in place modifications
    anno = copy.deepcopy(annotation)
    nx_graph = su.annotation_to_nx_graph(anno)

    branches = list({k for k, v in nx_graph.degree().iteritems() if v > 2})
    nx_graph.remove_nodes_from(branches)
    lonely = list({k for k, v in nx_graph.degree().iteritems() if v == 0})
    nx_graph.remove_nodes_from(lonely)

    ccs = list(nx.connected_component_subgraphs(nx_graph))

    lengths = [cc.size(weight='weight') / 1000. for cc in ccs]
    return lengths


def calc_arc_choord(anno, nxg=None):
    """
    Calculates the arc choord length of an annotation object as a measure of
    the tortuosity of an annotation. The return value is 1 for a completely
    straight skeleton. It uses the two most distant end nodes in an
    annotation and divides it by the graph path length between them.
    :param anno:
    :param nxg:
    :return: float
    """

    if not nxg:
        nxg = su.annoToNXGraph(anno)

    # find end nodes
    e_nodes = list({k for k, v in nxg.degree().iteritems() if v == 1})
    # get euclidian distance between end nodes max away
    # this can be done more efficiently by computing a convex hull first,
    # but the naive implementation should be fast enough for now
    dists = []
    for pair in itertools.permutations(e_nodes, 2):
        dists.append((pair, pair[0].distance_scaled(pair[1])))
    dists = sorted(dists, key=lambda x: x[1])
    try:
        path_len = nx.shortest_path_length(nxg, source=dists[-1][0][0],
                                           target=dists[-1][0][1],
                                           weight='weight')
    except:
        print('No path between nodes for tortuosity calculation for neuron %s' %
              anno.filename)
        return 0
    return dists[-1][1] / path_len