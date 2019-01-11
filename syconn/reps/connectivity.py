# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import matplotlib
matplotlib.use("Agg", warn=False, force=True)
import matplotlib.colors as mcolors
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas
import scipy.ndimage
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
try:
    default_wd_available = True
    from ..config.global_params import wd
except:
    default_wd_available = False
from ..mp import qsub_utils as qu
from ..mp import mp_utils as sm
from ..config import parser
from . import connectivity_helper as ch
from . import super_segmentation as ss
from . import segmentation
from ..handler.basics import load_pkl2obj, write_obj2pkl


# TODO: unclear what and when this was used for, refactor and use in current project
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def diverge_map(low=(239/255., 65/255., 50/255.),
                high=(39/255., 184/255., 148/255.)):
# def diverge_map(low=(255/255., 100/255., 80/255.),
#                 high=(60/255., 200/255., 160/255.)):
    """Low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    """
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, str): low = c(low)
    if isinstance(high, str): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])


def get_cum_pos(den_ranges, ax_ranges, den_pos, ax_pos):
    """Calculates the position of synapse in average matrix, i.e. which sector
    it belongs to.
    """
    den_cum_pos = 0
    ax_cum_pos = 0
    for i in range(1, len(den_ranges)):
        if (den_pos >= den_ranges[i-1]) and (den_pos < den_ranges[i]):
            den_cum_pos = i-1
    for i in range(1,  len(ax_ranges)):
        if (ax_pos >= ax_ranges[i-1]) and (ax_pos < ax_ranges[i]):
            ax_cum_pos = i-1
    return den_cum_pos, ax_cum_pos


class ConnectivityMatrix(object):
    def __init__(self, working_dir=None, version=None, sj_version=None,
                 ssd_version=None, create=False, config=None):
        self._ssv_ids = None
        self._config = config
        self._sj_ids = None
        self._sj_version = sj_version
        self._ssd_version = ssd_version
        self._sj_dataset = None
        self._ss_dataset = None

        self._cons = None
        self._axoness = None
        self._shapes = None
        self._blacklist = None
        self._cell_types = None

        if working_dir is None:
            if default_wd_available:
                self._working_dir = wd
            else:
                raise Exception(
                    "No working directory (wd) specified in config")
        else:
            self._working_dir = working_dir

        if version is None:
            try:
                self._version = self.config.entries["Versions"][self.type]
            except:
                raise Exception("unclear value for version")
        elif version == "new":
            other_datasets = glob.glob(
                self.working_dir + "/%s_*" % self.type)
            max_version = -1
            for other_dataset in other_datasets:
                other_version = \
                    int(re.findall("[\d]+",
                                   os.path.basename(other_dataset))[-1])
                if max_version < other_version:
                    max_version = other_version

            self._version = max_version + 1
        else:
            self._version = version

        if self._ssd_version is None:
            try:
                self._ssd_version = self.config.entries["Versions"]["ssv"]
            except:
                raise Exception("No version dict specified in config")

        if self._sj_version is None:
            try:
                self._sj_version = self.config.entries["Versions"]["sj"]
            except:
                raise Exception("No version dict specified in config")

        if create and not os.path.exists(self.path):
            os.makedirs(self.path)

    @property
    def type(self):
        return "con"

    @property
    def version(self):
        return str(self._version)

    @property
    def path(self):
        return "%s/con_%s/" % (self._working_dir, self.version)

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def config(self):
        if self._config is None:
            self._config = parser.Config(self.working_dir)
        return self._config

    @property
    def sj_dataset(self):
        if self._sj_dataset is None:
            self._sj_dataset = segmentation.SegmentationDataset(
                obj_type="sj", version=self._sj_version, working_dir=self.working_dir,
                create=False)
        return self._sj_dataset

    @property
    def sj_ids(self):
        if self._sj_ids is None:
            t = self.config.entries["Sizethresholds"]["sj"]
            self._sj_ids = np.load(self.sj_dataset.path + "/ids.npy")
            sizes = np.load(self.sj_dataset.path + "/sizes.npy")
            self._sj_ids = self._sj_ids[sizes > t]
        return self._sj_ids

    @property
    def ss_dataset(self):
        if self._ss_dataset is None:
            self._ss_dataset = ss.SuperSegmentationDataset(
                version=self._ssd_version, working_dir=self.working_dir)
        return self._ss_dataset

    @property
    def connectivity(self):
        if self._cons is None:
            self.load_cons()
        return self._cons

    @property
    def axoness(self):
        if self._axoness is None:
            self.load_axoness()
        return self._axoness

    @property
    def shapes(self):
        if self._shapes is None:
            self.load_shapes()
        return self._shapes

    @property
    def cell_types(self):
        if self._cell_types is None:
            self.load_cell_types()
        return self._cell_types

    @property
    def blacklist(self):
        if self._blacklist is None:
            self.load_blacklist()
        return self._blacklist

    def extent_sorted_sso_ids(self):
        sso_ids = np.array(list(self.shapes.keys())).astype(np.int)
        mask = np.in1d(sso_ids, self.blacklist, invert=True)
        sso_ids = sso_ids[mask]
        shapes = np.array(self.shapes.values())[mask]
        man_norm = np.linalg.norm(shapes, ord=1, axis=1)
        return sso_ids[np.argsort(man_norm)]

    def extract_connectivity(self, stride=1000, qsub_pe=None, qsub_queue=None,
                             nb_cpus=1):
        multi_params = []
        for id_block in [self.sj_ids[i:i + stride]
                         for i in range(0, len(self.sj_ids), stride)]:
            multi_params.append([id_block, self._sj_version, self._ssd_version,
                                 self.working_dir])

        if qsub_pe is None and qsub_queue is None:
            results = sm.start_multiprocess(ch.extract_connectivity_thread,
                                            multi_params, nb_cpus=nb_cpus)

        elif qu.__BATCHJOB__:
            path_to_out = qu.QSUB_script(multi_params,
                                         "extract_connectivity",
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=None)
            out_files = glob.glob(path_to_out + "/*")
            results = []
            for out_file in out_files:
                with open(out_file, 'rb') as f:
                    results.append(pkl.load(f))
        else:
            raise Exception("QSUB not available")

        cons = results[0]
        for result in results[1:]:
            cons = np.concatenate((cons, result))

        self._cons = cons
        self.save_cons()

    def get_sso_specific_info(self, stride=50, qsub_pe=None,
                              qsub_queue=None, nb_cpus=1):
        present_sso_ids = np.unique(self.connectivity[:, :2])

        multi_params = []
        for id_block in [present_sso_ids[i:i + stride]
                         for i in range(0, len(present_sso_ids), stride)]:
            multi_params.append([id_block, self._sj_version, self._ssd_version,
                                 self.working_dir, self.version])

        if qsub_pe is None and qsub_queue is None:
            results = sm.start_multiprocess(ch.get_sso_specific_info_thread,
                                            multi_params, nb_cpus=nb_cpus)

        elif qu.__BATCHJOB__:
            path_to_out = qu.QSUB_script(multi_params,
                                         "get_sso_specific_info",
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=None)
            out_files = glob.glob(path_to_out + "/*")
            results = []
            for out_file in out_files:
                with open(out_file, 'rb') as f:
                    results.append(pkl.load(f))
        else:
            raise Exception("QSUB not available")

        axoness_entries = []
        cell_types = {}
        blacklist = []
        shapes = {}

        for result in results:
            ae, ct, sh, bl = result
            if len(axoness_entries) == 0:
                axoness_entries = ae
            else:
                axoness_entries = np.concatenate((axoness_entries, ae))

            cell_types.update(ct)
            blacklist += bl
            shapes.update(sh)

        self._axoness = np.zeros([len(self.connectivity), 2], dtype=np.int)
        self._axoness[axoness_entries[:, 0], axoness_entries[:, 1]] = \
            axoness_entries[:, 2]

        self._shapes = shapes
        self._blacklist = blacklist
        self._cell_types = cell_types

        self.save_axoness()
        self.save_shapes()
        self.save_blacklist()
        self.save_cell_types()

    def build_matrix(self, n_elements=20000, type_threshold=.30):
        sso_ids = self.extent_sorted_sso_ids()[-n_elements:]

        relevant_mask = np.in1d(self.connectivity[:, 0], sso_ids)
        relevant_mask = np.logical_and(relevant_mask,
                                       np.in1d(self.connectivity[:, 1], sso_ids))

        relevant_cons = self.connectivity[relevant_mask]
        relevant_axoness = self.axoness[relevant_mask]

        problematic_axon_mask = ~np.any(relevant_axoness < 0, axis=1)

        relevant_cons = relevant_cons[problematic_axon_mask]
        relevant_axoness = relevant_axoness[problematic_axon_mask]

        axon_den_filter = np.sum(relevant_axoness, axis=1) == 1
        relevant_cons = relevant_cons[axon_den_filter]
        relevant_axoness = relevant_axoness[axon_den_filter]

        ax_partners = relevant_cons[:, :2][relevant_axoness == 0].astype(np.int)
        de_partners = relevant_cons[:, :2][relevant_axoness != 0].astype(np.int)

        syn_types = np.array(relevant_cons[:, -4] < type_threshold,
                             dtype=np.int) * 2 - 1

        unique_ax_partner = np.unique(ax_partners)
        unique_de_partner = np.unique(de_partners)

        ax_syn_dict = dict.fromkeys(unique_ax_partner, 0)
        for i_ax_p in range(len(ax_partners)):
            ax_syn_dict[ax_partners[i_ax_p]] += syn_types[i_ax_p]

        ax_syn_dict = dict(zip(unique_ax_partner,
                               np.array(syn_types < 0, dtype=np.int) * 2 - 1))

        print(np.unique(ax_syn_dict.values(), return_counts=True))
        print("N ax partner", len(unique_ax_partner))
        print("N de partner", len(unique_de_partner))

        syn_sizes = relevant_cons[:, -5]

        sso_ids = np.unique(relevant_cons[:, :2])

        cts = [[] for _ in range(4)]
        ct_cnt = np.zeros(4)
        for sso_id in sso_ids:
            cts[self.cell_types[sso_id]].append(sso_id)
            ct_cnt[self.cell_types[sso_id]] += 1

        for i in range(4):
            np.random.shuffle(cts[i])

        id_list = np.concatenate(cts)
        sso_id_mapping = {}
        for i_pos in range(len(id_list)):
            sso_id_mapping[id_list[i_pos]] = i_pos

        # cts = np.array(cts)
        # cts_sorting = np.argsort(cts)
        # sso_id_mapping = dict(zip(sso_ids, cts_sorting))

        matrix = np.zeros([len(sso_ids), len(sso_ids)])

        dendrites = np.zeros(4)
        axons = np.zeros(4)

        con_dict = {}

        cell_types = []
        for i_ax_p in range(len(ax_partners)):
            cell_types.append(np.array([self.cell_types[ax_partners[i_ax_p]], self.cell_types[de_partners[i_ax_p]]]))
            matrix[sso_id_mapping[de_partners[i_ax_p]],
                   sso_id_mapping[ax_partners[i_ax_p]]] += \
                syn_sizes[i_ax_p] * syn_types[i_ax_p] # * ax_syn_dict[ax_partners[i_ax_p]]

            key = (ax_partners[i_ax_p], de_partners[i_ax_p])
            if key in con_dict:
                con_dict[key].append(relevant_cons[i_ax_p, 2])
            else:
                con_dict[key] = [relevant_cons[i_ax_p, 2]]

            axons[self.cell_types[ax_partners[i_ax_p]]] += 1
            dendrites[self.cell_types[de_partners[i_ax_p]]] += 1

        print("Dendrites")
        print(dendrites)

        print("Axons")
        print(axons)

        cell_types = np.array(cell_types)
        sum_array = np.concatenate((ax_partners[:, None], de_partners[:, None], cell_types, relevant_cons[:, 2:]), axis=1)

        loc_dict = dict(zip(sso_id_mapping.values(), sso_id_mapping.keys()))

        np.save(self.path + "/con_sum_%d.npy" % matrix.shape[0], sum_array)
        write_obj2pkl(self.path + "/loc_to_sso_id_%d.pkl" % matrix.shape[0], loc_dict)
        write_obj2pkl(self.path + "/sso_id_to_loc_%d.pkl" % matrix.shape[0], sso_id_mapping)
        write_obj2pkl(self.path + "/sso_id_pair_to_sj_ids_%d.pkl" % matrix.shape[0], con_dict)

        #     print i_class, len(de_cell_types[i_class])
        #     df_de_ids += de_cell_types[i_class]

        return matrix, np.cumsum(ct_cnt).astype(np.int)[:-1]

        # con_dict = dict(zip(unique_ax_partner, [{} for _ in range(len(unique_ax_partner))]))
        #
        # for i_ax_p in range(len(ax_partners)):
        #     if de_partners[i_ax_p] in con_dict[ax_partners[i_ax_p]]:
        #         con_dict[ax_partners[i_ax_p]][de_partners[i_ax_p]] += \
        #             syn_sizes[i_ax_p] * ax_syn_dict[ax_partners[i_ax_p]]
        #     else:
        #         con_dict[ax_partners[i_ax_p]][de_partners[i_ax_p]] = \
        #             syn_sizes[i_ax_p] * ax_syn_dict[ax_partners[i_ax_p]]
        #
        # df = pandas.DataFrame(con_dict)
        #
        # df_ax_ids = df.columns.values
        # ax_cell_types = dict(zip(range(4), [[] for _ in range(4)]))
        # for df_ax_id in df_ax_ids:
        #     ax_cell_types[self.cell_types[df_ax_id]].append(df_ax_id)
        #
        # df_ax_ids = []
        # print "Axons"
        # for i_class in range(4):
        #     print i_class, len(ax_cell_types[i_class])
        #     df_ax_ids += ax_cell_types[i_class]
        #
        # df_de_ids = list(df.index)
        # de_cell_types = dict(zip(range(4), [[] for _ in range(4)]))
        # for df_de_id in df_de_ids:
        #     de_cell_types[self.cell_types[df_de_id]].append(df_de_id)
        #
        # df_de_ids = []
        # print "Dendrites"
        # for i_class in range(4):
        #     print i_class, len(de_cell_types[i_class])
        #     df_de_ids += de_cell_types[i_class]
        #
        # # df.reindex_axis(df_ax_ids, 1)
        # df.reindex_axis(df_de_ids, 0)
        #
        # self._con_matrix = df
        # self.save_matrix()

    def save_matrix(self):
        self._con_matrix.to_pickle(self.path + "/con_matrix.pkl")

    def load_matrix(self):
        self._con_matrix = pandas.read_pickle(self.path + "/con_matrix.pkl")

    def save_cons(self):
        np.save(self.path + "/raw_cons.npy", self._cons)

    def load_cons(self):
        if os.path.exists(self.path + "/raw_cons.npy"):
            self._cons = np.load(self.path + "/raw_cons.npy")
        else:
            return None

    def save_axoness(self):
        np.save(self.path + "/axoness_cons.npy", self._axoness)

    def load_axoness(self):
        if os.path.exists(self.path + "/axoness_cons.npy"):
            self._axoness = np.load(self.path + "/axoness_cons.npy")
        else:
            return None

    def save_blacklist(self):
        np.save(self.path + "/blacklist.npy", self._blacklist)

    def load_blacklist(self):
        if os.path.exists(self.path + "/blacklist.npy"):
            self._blacklist = np.load(self.path + "/blacklist.npy")
        else:
            return None

    def save_shapes(self):
        write_obj2pkl(self.path + "shapes.pkl", self._shapes)

    def load_shapes(self):
        try:
            self._shapes = load_pkl2obj(self.path + "shapes.pkl")
        except (IOError, EOFError):
            return None

    def save_cell_types(self):
        write_obj2pkl(self.path + "cell_types.pkl", self._cell_types)

    def load_cell_types(self):
        try:
            self._cell_types = load_pkl2obj(self.path + "cell_types.pkl")
        except (IOError, EOFError):
            return None

    def plot_wiring(self, intensity_plot, den_borders, ax_borders, entry_width=7, cum=False, cum_size=0):
        """Plot type sorted connectivity matrix and save to figures folder in
        working directory

        Parameters
        ----------
        wiring : np.array
            symmetric 2D array of size #cells x #cells
        den_borders:
        cell type boarders on post synaptic site
        ax_borders:
            cell type boarders on pre synaptic site
        max_val : float
            maximum cumulated contact area shown in plot
        confidence_lvl : float
            minimum probability of cell type prediction to keep cell
        binary : bool
            if True existence of synapse is weighted by 1, else 0
        add_fname : str
            supplement of image file
        maj_vote : tuple

        big_entries : bool
            artificially increase pixel size from 1 to 3 for better visualization
        """
        if cum:
            entry_width = 1

        intensity_plot_neg = intensity_plot < 0
        intensity_plot_pos = intensity_plot > 0

        borders = [0] + list(ax_borders) + [intensity_plot.shape[1]]
        for i_border in range(1, len(borders)):
            start = borders[i_border - 1]
            end = borders[i_border]
            sign = np.sum(intensity_plot_pos[:, start: end]) - \
                   np.sum(intensity_plot_neg[:, start: end]) > 0

            sign = [0, False, True, True, True][i_border]

            if sign:
                intensity_plot[:, start: end][intensity_plot[:, start: end] < 0] *= -1
            else:
                intensity_plot[:, start: end][intensity_plot[:, start: end] > 0] *= -1

        intensity_plot_neg = intensity_plot < 0
        intensity_plot_pos = intensity_plot > 0

        int_cut_pos = np.mean(intensity_plot[intensity_plot_pos]) + np.std(intensity_plot[intensity_plot_neg])
        int_cut_neg = np.abs(np.mean(intensity_plot[intensity_plot_neg])) + np.std(intensity_plot[intensity_plot_neg])

        print(int_cut_pos)
        print(int_cut_neg)

        intensity_plot *= -1
        if not cum:
            print(intensity_plot.shape)
            for k, b in enumerate(den_borders):
                b += k * 1
                intensity_plot = np.concatenate(
                    (intensity_plot[:b, :], np.zeros((1, intensity_plot.shape[1])),
                     intensity_plot[b:, :]), axis=0)

            print(intensity_plot.shape)
            for k, b in enumerate(ax_borders):
                b += k * 1
                intensity_plot = np.concatenate(
                    (intensity_plot[:, :b], np.zeros((intensity_plot.shape[0], entry_width)),
                     intensity_plot[:, b:]), axis=1)
            ax_borders_h = np.array([0, ax_borders[0], ax_borders[1], ax_borders[2],
                                     intensity_plot.shape[1]]) + np.array([0, 1, 2, 3, 4]) * entry_width

            bin_intensity_plot = intensity_plot != 0
            bin_intensity_plot = bin_intensity_plot.astype(np.float)
            intensity_plot = scipy.ndimage.convolve(intensity_plot, np.ones((entry_width, entry_width)))
            bin_intensity_plot = scipy.ndimage.convolve(bin_intensity_plot, np.ones((entry_width, entry_width)))
            intensity_plot /= bin_intensity_plot

            print(ax_borders_h, ax_borders)
            print(intensity_plot.shape)
            for b in ax_borders_h[-2:0:-1]:
                # intensity_plot = np.concatenate((intensity_plot[:, :b-entry_width+1], intensity_plot[:, b:]), axis=1)
                intensity_plot = np.concatenate((intensity_plot[:, :b], intensity_plot[:, b+entry_width-1:]), axis=1)

        print(intensity_plot.shape)

        ax_borders_h = np.array([0, ax_borders[0], ax_borders[1], ax_borders[2],
                                 intensity_plot.shape[1]]) + np.array([0, 1, 2, 3, 4])

        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure()
        # Create scatter plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
        gs.update(wspace=0.05, hspace=0.08)
        ax = plt.subplot(gs[0, 0], frameon=False)

        cax = ax.matshow(-intensity_plot.transpose(1, 0),
                         cmap=diverge_map(),
                         extent=[0, intensity_plot.shape[0], intensity_plot.shape[1], 0],
                         interpolation="none", vmin=-int_cut_neg,
                         vmax=int_cut_neg)
        ax.set_xlabel('Post', fontsize=18)
        ax.set_ylabel('Pre', fontsize=18)
        ax.set_xlim(0, intensity_plot.shape[0])
        ax.set_ylim(0, intensity_plot.shape[1])
        plt.grid(False)
        plt.axis('off')

        if cum:
            for k, b in enumerate(den_borders):
                plt.axvline(b, color='k', lw=0.5, snap=True,
                            antialiased=True)
            for k, b in enumerate(ax_borders):
                plt.axhline(b, color='k', lw=0.5, snap=True,
                            antialiased=True)
        else:
            for k, b in enumerate(den_borders):
                b += k * 1
                plt.axvline(b + 0.5, color='k', lw=0.5, snap=True,
                            antialiased=True)
            for k, b in enumerate(ax_borders):
                b += k * 1
                plt.axhline(b + 0.5, color='k', lw=0.5, snap=True,
                            antialiased=True)

        cbar_ax = plt.subplot(gs[0, 1])
        cbar_ax.yaxis.set_ticks_position('none')
        cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])
        plt.close()

        if cum:
            fig.savefig(self.path + "/matrix_cum_%d_%d_%d.png" % (cum_size, int(int_cut_neg*100000), int(int_cut_pos*100000)), dpi=600)
        else:
            fig.savefig(self.path + "/matrix_%d_%d_%d.png" % (intensity_plot.shape[0], int(int_cut_neg*100000), int(int_cut_pos*100000)), dpi=600)

    def plot_cum_wiring(self, intensity_plot, den_borders, ax_borders):
        cum_matrix = np.zeros([len(ax_borders) + 1, len(ax_borders) + 1])

        borders = [0] + list(ax_borders) + [intensity_plot.shape[1]]

        for i_ax_border in range(1, len(borders)):
            for i_de_border in range(1, len(borders)):
                ax_start = borders[i_ax_border - 1]
                ax_end = borders[i_ax_border]
                de_start = borders[i_de_border - 1]
                de_end = borders[i_de_border]
                cum = np.sum(np.abs(intensity_plot[de_start: de_end,
                                                   ax_start: ax_end]))

                cum_matrix[i_de_border-1, i_ax_border-1] = cum / (ax_end - ax_start) / (de_end - de_start)

        print(range(1, len(ax_borders)+1))
        self.plot_wiring(cum_matrix, range(1, len(ax_borders)+1), range(1, len(ax_borders)+1), cum=True, cum_size=intensity_plot.shape[0])


def get_sso_specific_info_thread(args):
    sso_ids = args[0]
    sj_version = args[1]
    ssd_version = args[2]
    working_dir = args[3]
    version = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir,
                                      version=ssd_version)

    cm = ConnectivityMatrix(working_dir, version=version,
                            sj_version=sj_version, create=False)
    axoness_entries = []
    cell_types = {}
    blacklist = []
    shapes = {}
    for sso_id in sso_ids:
        print(sso_id)
        sso = ssd.get_super_segmentation_object(sso_id)

        if not sso.load_skeleton():
            blacklist.append(sso_id)
            continue

        if "axoness" not in sso.skeleton:
            blacklist.append(sso_id)
            continue

        if sso.cell_type is None:
            blacklist.append(sso_id)
            continue

        con_mask, pos = np.where(cm.connectivity[:, :2] == sso_id)

        sj_coords = cm.connectivity[con_mask, -3:]
        sj_axoness = sso.axoness_for_coords(sj_coords)

        con_ax = np.concatenate([con_mask[:, None], pos[:, None],
                                 sj_axoness[:, None]], axis=1)

        if len(axoness_entries) == 0:
            axoness_entries = con_ax
        else:
            axoness_entries = np.concatenate((axoness_entries, con_ax))

        cell_types[sso_id] = sso.cell_type
        shapes[sso_id] = sso.shape

    axoness_entries = np.array(axoness_entries, dtype=np.int)
    return axoness_entries, cell_types, shapes, blacklist

