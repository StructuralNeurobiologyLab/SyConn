from processing.learning_rfc import cell_classification
from processing.spiness import collect_spineheads
from processing.features import propertyfeat2skelnode, \
    celltype_axoness_feature, spiness_feats_from_nodes, \
    radius_feats_from_nodes, az_per_spinehead, pathlength_of_property
from processing.datahandler import load_objpkl_from_kzip
from processing.synapticity import syn_sign_prediction
from utils.math import convex_hull_area


class Neuron(object):
    def __init__(self, annotations=[], unique_ID=None):

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
            if 'onsensu' in anno.filename\
                and not 'training' in anno.filename\
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
        #syn.neuron = self

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
    def avg_syn_az_len(self):
        return np.mean(np.array([s.az_len for s in self.synapses]))

    @property
    def syn_h_s_ratio(self):
        num_heads = float(len([syn for syn in self.synapses if syn.loc_vote ==
                         'spine-head']))
        num_shafts = float(len([syn for syn in self.synapses if syn.loc_vote ==
                     'shaft']))
        try:
            ratio = num_heads/(num_shafts+num_heads)
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
            skel_obj = NewSkeleton.NewSkeleton()
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
                anno.neuron_ID = self._ID # having neuron and the ID is
                # somewhat redundant, this should be thought of
                anno.neuron = self
                neuron_annos.append(anno)
            self.annotations = neuron_annos

    def set_statistics(self):
        if len(self.consensus_annotations) > 1:
            for a in self.consensus_annotations:
                if 'skeleton' in a.comment:
                    anno_to_use = a
        else:
            if self.annotations:
                anno_to_use = self.annotations[0]

        # for now, the first annotation is used if multiples are present
        self.nx_g = au.annotation_to_nx_graph(anno_to_use)
        self.features['all_path_length_um'] = anno_to_use.physical_length()/1000.
        self.features['num_all_branch_p'] = num_branch_points_of_nx_graph(self.nx_g)
        self.features['all_branch_density'] = 0.
        self.features['num_all_end_p'] = num_end_points_of_nx_graph(self.nx_g)
        self.features['all_branch_density'] = 0.
        self.features['tortuosity'] = calc_arc_choord(anno_to_use, self.nx_g)

        if os.path.isfile(self._mapped_skel_dir + anno_to_use.filename) or \
            os.path.isfile(anno_to_use.filename):
            object_feats, rad_feats, type_feats, spine_feats = \
                cell_morph_properties(anno_to_use)
            self.features['obj_morphology'] = object_feats
            self.features['rad_morphology'] = rad_feats
            self.features['type_morphology'] = type_feats
            self.features['spine_morphology'] = spine_feats
            self.features['synapse_type'] = calc_syn_type_feats(anno_to_use)
            self.features['mito density'] = 0.
            self.features['vc density'] = 0.
            self.features['sj density'] = 0.

            if self.features['all_path_length_um'] > 0.0:
                self.features['mito density'] = self.features['obj_morphology']\
                [0, 2] / self.features['all_path_length_um']
                self.features['vc density'] = self.features['obj_morphology']\
                                    [0, 5] / self.features['all_path_length_um']
                self.features['sj density'] = self.features['obj_morphology']\
                                    [0, 8] / self.features['all_path_length_um']
            self._feature_name_dict['obj_morphology'] = \
                np.array(['mean size of mito', 'std of mito size',
                          'abs. number of mitos',
                          'mean size of vc', 'std of vc size', 'abs. number of vc',
                          'mean size of sj', 'std of sj size', 'abs. number if sj'])[None, :]
            self._feature_name_dict['rad_morphology'] = \
            np.array(['mean radius (dend.)', 'std of radius (dend.)'] +
            ['bin %d of radius hist. (dend.)' % nb for nb in range(4)] +
            ['mean radius (axon)', 'std of radius (axon)'] +
            ['bin %d of radius hist. (axon)' % nb for nb in range(4)] +
            ['mean radius (soma)', 'std of radius (soma)'])[None, :]
            self._feature_name_dict['type_morphology'] = \
                np.array(['prop. of dendrite pathlength',
                          'prop. of axon pathlength',
                          'prop. of soma pathlength',
                          'soma branches'])[None, :]
            self._feature_name_dict['spine_morphology'] = \
                np.array(['sh density (dend.)', 'mean radius (dend.)',
                'std of radius (dend.)', 'mean of sh probability (dend.)',
                'mean neck length (dend.)', 'std neck length (dend.)'] +
                ['sh density (axon)', 'mean radius (axon)',
                'std of radius (axon)', 'mean of sh probability (axon)',
                'mean neck length (axon)', 'std neck length (axon)',
                'number az per sh'])[None, :]
            self._feature_name_dict['synapse_type'] = np.array(
            ['outgoing syn. type', 'proportion of inc. syn. type',
             'median of outgoing syn. size', 'std of outgoing syn. size',
             'median of inc. asym. size',
             'std of inc. asym. size', 'median of inc. sym. size',
             'std of inc. sym. size'])[None, :]

        if self.features['all_path_length_um'] > 0.0:
            self.features['all_branch_density'] = \
                self.features['num_all_branch_p'] / self.features['all_path_length_um']
            self.features['all_end_density'] = \
                self.features['num_all_end_p'] / self.features['all_path_length_um']
