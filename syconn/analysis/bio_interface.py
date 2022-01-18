# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from mayavi import mlab
import functools
#import sys
#sys.setrecursionlimit(500)
#import numba
#print('test0')
import matplotlib as mpl
mpl.use('Qt5Agg')
mpl.interactive(True)
#from mayavi import mlab
from syconn.reps import super_segmentation
#print('test1')
from syconn.reps import segmentation
#print('test2')
from syconn import global_params

import numpy as np
import pickle
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial import distance
import scipy
from typing import Optional, List, Dict, Union, Iterable
import time
import os
import pandas as pd
import copy
import random
import itertools
#import matplotlib as mpl
#mpl.use('TkAgg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import numba
#print('test2')

sys.setrecursionlimit(50000000) # required for pickling 50000000

sys.path.append('/wholebrain/u/jkor/repos/')

# This variable is required for neuron wiring distance calculation with numba
# See function get_norm_mds_fv() and numba_pairwise_neuron_dist()
# for an usage example.
glob_conn_mat = np.zeros((2,2))

rev_celltype = {0:'STN', 1:'DA', 2:'MSN', 3:'LMAN', 4:'HVC', 5:'TAN', 6:'GPe', 7:'GPi', 8:'FS', 9:'LTS', 10:'NGF'}

class Timer:
    """
    Simple timer, optionally named, for use as context manager.
    Time is given in seconds.
    E.g.:
    with Timer('print-test'):
        print(f'{[i for i in range(100000)]}')
    """
    def __init__(self, name: Optional[str] = None):
        """
        Init function.
        Args:
            name: Plots the given name additionally to the passed time.
        """
        self.start = 0
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        if self.name:
            print(f'Timer {self.name} finished after {time.time()-self.start}s.')
        else:
            print(f'Timer finished after {time.time() - self.start}s.')

class Dataset:
    """
    Lightweight in memory wrapper of a SyConn dataset for analyses purposes.
    Analyses often iterate and access the data either in a synapse-centric or
    neuron-centric paradigm (or focused on another organelle), which is why
    these are directly exposed.
    todo: benchmark max. num synapses for e.g. xGB of RAM.
    """
    def __init__(self):
        """
        Datasets are currently populated using the module level
        function init_in_mem_dataset() mainly historical reasons.
        """
        self.neurons: dict = dict()
        self.synapses: dict = dict()
        self.mitos: dict = dict()
        self.vesicles: dict = dict()

class Neuron:
    """
    Represents a single neuron in the dataset.
    """
    def __init__(self, ID, celltype: Optional[str] = None) -> None:
        self.ID: int = ID
        self.SV_ids: Optional[List[int]] = None
        self.celltype: str = celltype
        self.axon: Compartment = Compartment()
        self.dendrite: Compartment = Compartment()
        self.soma: Compartment = Compartment()
        self.feature_cache: Optional[dict] = None
        self.in_analysis_set: bool = False

    #def feature_vector(self):
    #    if not self.feature_cache:
            # keep this in a separate function, makes development easier,
            # because changing features doesn't require a complete object-reinit
    #        update_neuron_features(self)
    #    return self.feature_cache

class Compartment:
    """
    A neuron is structured into compartments which allow access to the contained
    structures. It is not enforced that a compartment is anatomically continuous.
    The main compartments in use at the moment are axon, dendrite and soma,
    but any other type could be defined. Compartments support querying basic
    statistics about their structure.
    """
    def __init__(self) -> None:
        self.synapses: Dict = dict()
        self.mitos: Dict = dict()
        self.vesicles: Dict = dict()
        self.neuron: Neuron = None
        self.skeleton = None # too heavy for larger datasets for sure.
        self.skel_length = 0.

    @property
    def mito_sizes(self):
        if self.mitos:
            return np.array([v.size for v in self.mitos.values()])
        else:
            return [0.0]

    @property
    def vesicle_sizes(self):
        if self.vesicles:
            return np.array([v.size for v in self.vesicles.values()])
        else:
            return [0.0]

    @property
    def synapse_sizes(self):
        if self.synapses:
            return np.array([v.size for v in self.synapses.values()])
        else:
            return [0.0]

    @property
    def synapse_path_density(self):
        # caching decorators can be problematic with pickling
        try:
            return self._synapse_path_density
        except AttributeError:
            if self.skel_length > 0.:
                self._synapse_path_density = len(self.synapses)/self.skel_length
            else:
                self._synapse_path_density = 0.
            return self._synapse_path_density

    @property
    def mitos_path_density(self):
        try:
            return self._mitos_path_density
        except AttributeError:
            if self.skel_length > 0.:
                self._mitos_path_density = len(self.mitos)/self.skel_length
            else:
                self._mitos_path_density = 0.
            return self._mitos_path_density

    @property
    def vesicles_path_density(self):
        try:
            return self._vesicles_path_density
        except AttributeError:
            if self.skel_length > 0.:
                self._vesicles_path_density =  len(self.vesicles) / self.skel_length
            else:
                self._vesicles_path_density = 0.
            return self._vesicles_path_density

    @property
    def synapse_vol_path_density(self):
        try:
            return self._synapse_vol_path_density
        except AttributeError:
            if self.skel_length > 0.:
                self._synapse_vol_path_density = np.sum(self.abs_synapse_sizes) / self.skel_length
            else:
                self._synapse_vol_path_density = 0.
            return self._synapse_vol_path_density

    @property
    def mitos_vol_path_density(self):
        try:
            return self._mitos_vol_path_density
        except AttributeError:
            if self.skel_length > 0.:
                self._mitos_vol_path_density = np.sum(self.mito_sizes) / self.skel_length
            else:
                self._mitos_vol_path_density = 0.
            return self._mitos_vol_path_density

    @property
    def vesicles_vol_path_density(self):
        try:
            return self._vesicles_vol_path_density
        except AttributeError:
            if self.skel_length > 0.:
                self._vesicles_vol_path_density = np.sum(self.vesicle_sizes) / self.skel_length
            else:
                self._vesicles_vol_path_density = 0.
            return self._vesicles_vol_path_density

    @property
    def abs_synapse_sizes(self):
        if self.synapses:
            return np.array([np.abs(v.size) for v in self.synapses.values()])
        else:
            return [0.0]

    @property
    def fraction_asym(self):
        try:
            return self._fraction_asym
        except AttributeError:
            asym_syns_size = self.synapse_sizes[(self.synapse_sizes > 0.)]
            sym_syns_size = self.abs_synapse_sizes[(self.synapse_sizes < 0.)]
            self._fraction_asym = (np.sum(asym_syns_size)/(np.sum(asym_syns_size)+np.sum(sym_syns_size)))
            return self._fraction_asym

    @property
    def post_neurons(self):
        return [v.post for v in self.synapses.values()]

    @property
    def pre_neurons(self):
        return [v.pre for v in self.synapses.values()]

    @property
    def head_other_ratio(self):
        # We currently care only about the dendritic morphology,
        # also in the case of axon compartments! Might be interesting
        # to add the presynaptic morphology as well at some point
        try:
            return self._head_other_ratio
        except AttributeError:
            head_syns = np.array(
                [np.abs(s.size) for s in self.synapses.values() if
                 s.post_morph == 1])
            other_syns = np.array(
                [np.abs(s.size) for s in self.synapses.values() if
                 s.post_morph != 1])
            self._head_other_ratio = (np.sum(head_syns) / (np.sum(other_syns) + np.sum(head_syns)))
            return self._head_other_ratio

class Synapse:
    """
    Class which represents a synapse, including various properties that
    were inferred from syconn.
    """

    def __init__(self, ID: int, pre, post, post_spine_vol, post_morph, pre_morph, size: float,
                 coordinate, pre_latent_morph, post_latent_morph, prob: float) -> None:

        self.id: int = ID
        self.type: Optional[int] = None
        self.pre = pre
        self.post = post
        self.size = size
        self.post_morph = post_morph # spine, shaft, neck
        self.post_spine_vol = post_spine_vol
        self.pre_morph = pre_morph
        self.pre_latent_morph = pre_latent_morph
        self.post_latent_morph = post_latent_morph
        self.coordinate = coordinate
        self.prob = prob

class Organelle():
    """
    Class which represents e.g., mitochondria or synaptic vesicles clouds.
    """
    def __init__(self, ID: int, parent: Compartment, o_type: str, size: float, coordinate):
        self.id = ID
        self.type = o_type
        self.parent = parent
        self.size = size
        self.coordinate = coordinate

class Spine:
    def __init__(self, ID, dendrite, volume, coordinate):
        self.id = ID
        self.dendrite = dendrite
        self.synapses = dict()
        self.volume = volume
        self.coordinate = coordinate


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def scale_coords(coords, sc=[0.009, 0.009, 0.02], ds='j0126'):
    if ds == 'j0126':
        scaled = np.multiply(coords, np.array([sc]*len(coords)))
    elif ds == 'j0251':
        scaled = np.multiply(coords, np.array([[0.01, 0.01, 0.025]]*len(coords)))
    return scaled

def scale_coord(coord, sc=[0.009, 0.009, 0.02], ds='j0126'):
    if ds == 'j0126':
        scaled = np.multiply(coord, sc)
    elif ds == 'j0251':
        scaled = np.multiply(coord, [0.01, 0.01, 0.025])
    return scaled

def descale_coord(coord, roundint=False):
    scaled = np.divide(coord, [0.009, 0.009, 0.02])
    if roundint:
        scaled = scaled.astype(np.int)
    return scaled

def benchmark_dataset_creation(num_syns: int, num_neurons:int ) -> None:
    """
    Very simple benchmark to test how many neurons and synapses can be stored
    in this way.
    Args:
        num_syns: Number of synapses to simulated
        num_neurons: Number of neurons to simulate

    """

    # generate random pre and post partners
    pre = [random.randint(1, num_neurons) for _ in range(num_syns)]
    post = [random.randint(1, num_neurons) for _ in range(num_syns)]
    syn_cnt = 1

    mds = Dataset()
    print(f'Creating in memory dataset with {num_syns} synapses'
          f' and {num_neurons} neurons')

    with Timer('in memory dataset creation'):
        for pre, post in zip(pre, post):
            if not pre in mds.neurons:
                mds.neurons[pre] = Neuron(pre, celltype='a')
            if not post in mds.neurons:
                mds.neurons[post] = Neuron(post, celltype='a')

            this_syn = Synapse(ID=syn_cnt, pre=mds.neurons[pre],
                               post=mds.neurons[post], size=0.5,
                               post_spine_vol=0.2,
                               post_morph='spine',
                               pre_morph='',
                               coordinate=[2,
                                           2,
                                           0],
                               pre_latent_morph=[1,1,1,1,1,1,1,1,1,1,1],
                               post_latent_morph=[1,1,1,1,1,1,1,1,1,1,1],
                               prob=0.5)
            syn_cnt += 1
            mds.synapses[syn_cnt] = this_syn
            mds.neurons[pre].axon.synapses[syn_cnt] = this_syn
            mds.neurons[post].dendrite.synapses[syn_cnt] = this_syn

    return


def update_neuron_features(n: Neuron) -> None:
    """
    Recalculates the feature vector for a neuron. This happens outside of the
    neuron class to allow for rapid development cycles, since recreation of the
    neuron objects is relatively time consuming.
    The feature vector of each neurite is split up into three feature classes:

        axon features, dendrite features and soma features.

    If a neurite lacks a compartment, these features do not contribute to
    the distance metric between two neurites.

    todo: Replace hard coded parameters

    Args:
        n: neuron instance

    """
    fv: Dict = dict()

    skel_nx = create_nx_skel_of_neuron(n, ds='j0251')
    #skel_nx = prune_skel_stub_branches(skel_nx, n, 5.)
    high_deg_nodes = set({k for k, v in skel_nx.degree if v >= 2})
    if (n.axon.skel_length + n.dendrite.skel_length + n.soma.skel_length) > 0.:
        global_branch_density = len(high_deg_nodes) / (n.axon.skel_length +
                                                       n.dendrite.skel_length
                                                       + n.soma.skel_length)
    else:
        global_branch_density = 0.

    #branch_density = 0.
    #n.pruned_skel_nx = skel_g

    if len(n.axon.synapses) > 15:
        # there are two different postsynaptic compartments, dendrites and
        # somata, and some might be empty
        weighted_dendrite_fract_asym = np.mean(
            [s.post.dendrite.fraction_asym * np.abs(s.size) for s in
            n.axon.synapses.values() if s.post.dendrite.synapses])
        #weighted_soma_fract_asym = [
        #    s.post.soma.fraction_asym * np.abs(s.size) for s in
        #    n.axon.synapses.values() if s.post.soma.synapses]
        #comb_post_fract_asym = np.mean(weighted_dendrite_fract_asym + weighted_soma_fract_asym)
        terminal_syns = np.sum([np.abs(s.size) for s in
                                n.axon.synapses.values() if s.pre_morph == 'terminal'])
        en_passant_syns = np.sum([np.abs(s.size) for s in
                                n.axon.synapses.values() if s.pre_morph == 'bouton'])
        if terminal_syns + en_passant_syns > 0.:
            fract_terminal_syns = terminal_syns / (terminal_syns + en_passant_syns)
        else:
            fract_terminal_syns = 0.

        ax_idx = n.skeleton['axoness_avg10000'] == 1 # find ax nodes
        ax_rad = n.skeleton['diameters'][ax_idx] # find diameters
        tot = len(ax_rad)
        if tot:
            rad_bin0 = len([r for r in ax_rad if r <= 10.])/tot
            rad_bin1 = len([r for r in ax_rad if r > 10. and r < 35.])/tot
            rad_bin2 = len([r for r in ax_rad if r >= 35.])/tot
        else:
            rad_bin0 = rad_bin1 = rad_bin2 = 0

        weighted_dendrite_head_other = np.mean(
            [s.post.dendrite.head_other_ratio * np.abs(s.size) for s in
            n.axon.synapses.values() if s.post.dendrite.synapses])
        if not np.isfinite(weighted_dendrite_head_other):
            weighted_dendrite_head_other = 0.

        if not np.isfinite(weighted_dendrite_fract_asym):
            weighted_dendrite_fract_asym = 0.

        latent_features_post = np.array(
            [s.post_latent_morph for s in n.axon.synapses.values()])

        tot_sum_syn = np.sum(n.axon.abs_synapse_sizes)
        tot_sum_vesicles = np.sum(n.axon.vesicle_sizes)
        syn_vesicle_ratio = tot_sum_vesicles / (tot_sum_syn+tot_sum_vesicles)

        fv['axon'] = [
                      global_branch_density,
                      n.axon.synapse_path_density,
                      n.axon.synapse_vol_path_density,
                      n.axon.mitos_path_density,
                      n.axon.mitos_vol_path_density,
                      n.axon.vesicles_path_density,
                      n.axon.vesicles_vol_path_density,
                      n.axon.fraction_asym,
                      fract_terminal_syns,
                      rad_bin0,
                      rad_bin1,
                      rad_bin2,
                      n.axon.head_other_ratio,
                      np.median(n.axon.mito_sizes),
                      np.std(n.axon.mito_sizes),
                      np.median(n.axon.vesicle_sizes),
                      np.std(n.axon.vesicle_sizes),
                      np.median(n.axon.abs_synapse_sizes),
                      np.std(n.axon.abs_synapse_sizes),
                      weighted_dendrite_fract_asym,
                      weighted_dendrite_head_other,
                      syn_vesicle_ratio]

        latent_features = np.array(
            [s.pre_latent_morph for s in n.axon.synapses.values()])
        fv['axon'].extend(np.mean(latent_features, axis=0))
        fv['axon'].extend(np.mean(latent_features_post, axis=0))

    else:
        fv['axon'] = np.zeros(42)

    if len(n.dendrite.synapses) > 15:
        weighted_axon_fract_asym = [
            s.pre.axon.fraction_asym * np.abs(s.size) for s in
            n.dendrite.synapses.values() if s.pre.axon.synapses]
        comb_pre_fract_asym = np.mean(weighted_axon_fract_asym)

        weighted_axon_head_other = [
            s.pre.axon.head_other_ratio * np.abs(s.size) for s in
            n.dendrite.synapses.values() if s.pre.axon.synapses]
        comb_pre_head_other = np.mean(weighted_axon_head_other)

        terminal_syns = np.sum([np.abs(s.size) for s in
                                n.dendrite.synapses.values() if s.pre_morph == 'terminal'])
        en_passant_syns = np.sum([np.abs(s.size) for s in
                                n.dendrite.synapses.values() if s.pre_morph == 'bouton'])
        if terminal_syns + en_passant_syns > 0.:
            fract_terminal_syns = terminal_syns / (terminal_syns + en_passant_syns)
        else:
            fract_terminal_syns = 0.

        idx = n.skeleton['axoness_avg10000'] == 0 # find dendrite nodes
        rad = n.skeleton['diameters'][idx] # find diameters
        tot = len(rad)
        if tot:
            rad_bin0 = len([r for r in rad if r <= 20.])/tot
            rad_bin1 = len([r for r in rad if r > 20. and r < 45.])/tot
            rad_bin2 = len([r for r in rad if r >= 45.])/tot
        else:
            rad_bin0 = rad_bin1 = rad_bin2 = 0

        head_syns = len([np.abs(s.size) for s in n.dendrite.synapses.values() if s.post_morph == 1 and s.size > 0.])
        head_path_density_asym = head_syns / n.dendrite.skel_length

        head_syns = len([np.abs(s.size) for s in n.dendrite.synapses.values() if s.post_morph == 1 and s.size < 0.])
        head_path_density_sym = head_syns / n.dendrite.skel_length

        fv['dendrite'] = [n.dendrite.synapse_path_density,
                          n.dendrite.synapse_vol_path_density,
                          n.dendrite.mitos_path_density,
                          n.dendrite.mitos_vol_path_density,
                          n.dendrite.fraction_asym,
                          fract_terminal_syns,
                          head_path_density_asym,
                          head_path_density_sym,
                          rad_bin0,
                          rad_bin1,
                          rad_bin2,
                          n.dendrite.head_other_ratio,
                          np.median(n.dendrite.mito_sizes),
                          np.std(n.dendrite.mito_sizes),
                          comb_pre_fract_asym,
                          comb_pre_head_other,
                          np.median(np.abs(n.dendrite.synapse_sizes)), # value_if_true if condition else value_if_false
                          np.std(np.abs(n.dendrite.synapse_sizes))]
        latent_features = np.array(
            [s.post_latent_morph for s in n.dendrite.synapses.values()])

        latent_features_pre = np.array(
            [s.pre_latent_morph for s in n.dendrite.synapses.values()])
        #fv['dendrite'].extend(np.mean(latent_features, axis=0))
        #fv['dendrite'].extend(np.mean(latent_features_pre, axis=0))
    else:
        fv['dendrite'] = np.zeros(18)

    if len(n.soma.synapses) > 5:
        fv['soma'] = [n.soma.fraction_asym,
                      np.median(n.soma.mito_sizes),
                      np.std(n.soma.mito_sizes),
                      np.sum(n.soma.mito_sizes), # no vol. normal. ok for soma
                      n.soma.mitos_path_density,
                      n.soma.synapse_path_density,
                      np.median(np.abs((n.soma.synapse_sizes))),
                      np.std(np.abs(n.soma.synapse_sizes))]
    else:
        fv['soma'] = np.zeros(8)

    n.feature_cache = fv
    return


def get_feature_labels(dendrite=True, axon=True, soma=True):
    labels_axon = [
              # axon below
              'global_branch_density',
              'axon.synapse_path_density',
              'axon.synapse_vol_path_density',
              'axon.mitos_path_density',
              'axon.mitos_vol_path_density',
              'axon.vesicles_path_density',
              'axon.vesicles_vol_path_density',
              'axon.fraction_asym',
              'axon.fract_terminal_syns',
              'axon.rad_bin0',
              'axon.rad_bin1',
              'axon.rad_bin2',
              'axon.head_other_ratio',
              'axon.median(mito_sizes)',
              'axon.std(mito_sizes)',
              'axon.median(vesicle_sizes)',
              'axon.std(vesicle_sizes)',
              'axon.median(abs_synapse_sizes)',
              'axon.std(abs_synapse_sizes)',
              'axon.weighted_dendrite_fract_asym',
              'axon.weighted_dendrite_head_other',
              'axon.syn_vesicle_ratio',
              #'axon.soma_post_density',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',
              'axon.mean.latent_pre',

              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post',
              'axon.mean.latent_post']
    labels_dendrite = [
              # dendrite below
              'dendrite.synapse_path_density',
              'dendrite.synapse_vol_path_density',
              'dendrite.mitos_path_density',
              'dendrite.mitos_vol_path_density',
              'dendrite.fraction_asym',
              'dendrite.fract_terminal_syns',
              'head_path_density_asym',
              'head_path_density_sym',
              'dendrite.rad_bin0',
              'dendrite.rad_bin1',
              'dendrite.rad_bin2',
              'dendrite.head_other_ratio',
              'dendrite.median(mito_sizes)',
              'dendrite.std(mito_sizes)',
              'dendrite.comb_pre_fract_asym',
              'dendrite.comb_pre_head_other',
              'dendrite.median(abs_synapse_sizes)',
              'dendrite.std(abs_synapse_sizes)']
    #                           np.median(np.abs(n.dendrite.synapse_sizes)), # value_if_true if condition else value_if_false
    #                           np.std(np.abs(n.dendrite.synapse_sizes))]

              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',
              #'dendrite.mean.latent_post',

              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
             # 'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
              #'dendrite.mean.latent_pre',
    labels_soma = [
              # soma below
              'soma.fraction_asym',
              'soma.median(mito_sizes)',
              'soma.std(mito_sizes)',
              'soma.sum(mito_sizes)',
              'soma.mitos_path_density',
              'soma.synapse_path_density',
              'soma.median(abs_synapse_sizes)',
              'soma.std(abs_synapse_sizes)'
              ]

    labels = (labels_axon if axon else []) +\
             (labels_dendrite if dendrite else []) +\
             (labels_soma if soma else [])

    return labels

syconn_rec = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019/connectivity_matrix/conn_mat.csv'

# with reconnects and soma filter:
syconn_rec = '/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed/'
'assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed'

# /ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger/
syconn_no_rec = '/ssdscratch/pschuber/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/'
celltype_gt_csv = '/wholebrain/songbird/j0126/GT/celltype_gt/j0126_cell_type_gt_areax_fs6_v3.csv'
mds_pkl_path='/wholebrain/songbird/j0126/mds_reconnects_May20.pkl'
#celltype_gt_csv = '' should be read out from the syconn working dir!

#syconn_wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019/'
syconn_wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/'
syconn_wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/'


syconn_wd = '/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/'

#mds_pkl_path='/wholebrain/songbird/j0251/mds_May_25_2020.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_June_26_2020.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_July_16_2020.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_August_4_2020_v2.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Sept_10_2020_v2.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Sept_20_2020_v2.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Dec_12_2020_v3.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Dec_15_2020_v3.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Dec_18_2020_v3_low_prob_min_50ax.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Dec_30_2020_v3_low_prob_min_50ax.pkl'
mds_pkl_path='/wholebrain/songbird/j0251/mds_Dec_10_2021_v4.pkl'

def init_in_mem_dataset(from_scratch = False,
                        syconn_working_dir = syconn_wd,
                        mds_pkl_path = mds_pkl_path):

    # check if mds exists already:
    if not from_scratch:
        if os.path.exists(mds_pkl_path):
            with Timer('mds from pickle'):
                #import gc
                #gc.disable()
                with open(mds_pkl_path, 'rb') as fh:
                    mds = pickle.load(fh)
                    #update_mds_neuron_features(mds)
                #gc.enable()
                return mds

    # Extracting the information stored in the csv file is expensive,
    # we therefore reuse these data instead of parsing it from scratch for now.
    # The goal is however to skip the csv file and populate everything directly
    # from syconn cache arrays, or any other backend syconn may provide in the
    # future.
    csv_conn = syconn_working_dir + '/connectivity_matrix/conn_mat.csv'
    #csv_conn = '/ssdscratch/jkor/to_del_mat.csv'

    mds = Dataset()

    with Timer('pandas csv read'):
        cm = pd.read_csv(csv_conn, sep='\t')

    with Timer('dataset construction csv'):
        unassigned_syns = 0
        unassigned_syns_post_morph = 0
        unassigned_syns_finite= 0
        unassigned_syns_axo_dend = 0
        syn_cnt = 1
        num_zero_syns = 0
        for row in cm.itertuples(index=False):
            #print(row)

            synprob = row.synprob

            # for testing
            #if synprob < 0.9:
            #    continue

            # 0. in case there is no data
            post_spine_vol = 0.0
            # check whether the synapse information is complete

            comp1 = round(row.comp1)
            comp2 = round(row.comp2)

            if comp1 == -1 or comp2 == -1:
                unassigned_syns += 1
                continue

            latent_morph1 = [row.latentmorph1_0, row.latentmorph1_1,
                            row.latentmorph1_2, row.latentmorph1_3,
                            row.latentmorph1_4, row.latentmorph1_5,
                            row.latentmorph1_6, row.latentmorph1_7,
                            row.latentmorph1_8, row.latentmorph1_9]

            latent_morph2 = [row.latentmorph2_0, row.latentmorph2_1,
                            row.latentmorph2_2, row.latentmorph2_3,
                            row.latentmorph2_4, row.latentmorph2_5,
                            row.latentmorph2_6, row.latentmorph2_7,
                            row.latentmorph2_8, row.latentmorph2_9]



            # replace hard coded labels with global syconn provided labels
            if ((comp1 == 1 or comp1 == 3 or comp1 == 4) and \
                    (comp2 == 0 or comp2 == 2)):
                pre = round(row.ssv1)
                post = round(row.ssv2)
                post_spine_vol = row.spinehead_vol2
                post_morph = round(row.spiness2)
                post_comp = comp2
                pre_type = round(row.celltype1)
                post_type = round(row.celltype2)
                pre_morph_latent = latent_morph1
                post_morph_latent = latent_morph2
                pre_morph = 'axon'
                if comp1 == 3:
                    pre_morph = 'bouton'
                if comp1 == 4:
                    pre_morph = 'terminal'
            elif ((comp2 == 1 or comp2 == 3 or comp2 == 4) and (comp1 == 0 or comp1 == 2)):
                pre = round(row.ssv2)
                post = round(row.ssv1)
                post_spine_vol = row.spinehead_vol1
                post_morph = round(row.spiness1)
                post_comp = comp1
                pre_type = round(row.celltype2)
                post_type = round(row.celltype1)
                pre_morph_latent = latent_morph2
                post_morph_latent = latent_morph1
                pre_morph = 'axon'
                if comp2 == 3:
                    pre_morph = 'bouton'
                if comp2 == 4:
                    pre_morph = 'terminal'
            else:
                # skip axo-axonic, dendro-dendritic etc synapses for now
                unassigned_syns_axo_dend += 1
                unassigned_syns += 1
                continue

            if not np.isfinite(row.size):
                unassigned_syns_finite += 1
                unassigned_syns += 1
                continue

            if row.size == 0.:
                num_zero_syns += 1
                continue

            if post_morph == -1:
                # skip synapses with unclassified post-synaptic morphology
                unassigned_syns += 1
                unassigned_syns_post_morph += 1
                continue

            # check if associated neurons exist already in dataset and add if not
            if not pre in mds.neurons:
                mds.neurons[pre] = Neuron(pre, celltype = rev_celltype[pre_type])
            if not post in mds.neurons:
                mds.neurons[post] = Neuron(post, celltype = rev_celltype[post_type])

            # create synapse object; todo: replace id with global syconn synapse id
            this_syn = Synapse(ID=syn_cnt, pre=mds.neurons[pre],
                               post=mds.neurons[post], size=row.size,
                               post_spine_vol=post_spine_vol,
                               post_morph=post_morph,
                               pre_morph=pre_morph,
                               coordinate=[int(row._0), int(row.y), int(row.z)],
                               pre_latent_morph = pre_morph_latent,
                               post_latent_morph = post_morph_latent,
                               prob=synprob)
            syn_cnt += 1
            mds.synapses[syn_cnt] = this_syn
            mds.neurons[pre].axon.synapses[syn_cnt] = this_syn

            if post_comp == 0:
                mds.neurons[post].dendrite.synapses[syn_cnt] = this_syn
            elif post_comp == 2:
                mds.neurons[post].soma.synapses[syn_cnt] = this_syn

    # print basic parsing statistics
    print(f'num 0-size syns: {num_zero_syns};'
          f'num unassigned syns {unassigned_syns};'
          f'unassigned_syns_axo_dend {unassigned_syns_axo_dend};'
          f'unassigned_syns_post_morph {unassigned_syns_post_morph};'
          f'unassigned_syns_finite {unassigned_syns_finite}')

    print(f'Parsed {len(mds.neurons)} neurons and {len(mds.synapses)} synapses')

    # Since the csv cannot contain all data, we start loading some data directly
    # from syconn.
    cnt = 0
    for n in mds.neurons.values():
        #
        if len(n.dendrite.synapses) > 50 or len(n.axon.synapses) >= 50:
        #if len(n.dendrite.synapses) > 2 or len(n.axon.synapses) > 2:
            cnt+=1
            n.in_analysis_set = True
        else:
            n.in_analysis_set = False
    print(f'Neurons in analysis set: {cnt}')

    print('Start loading properties directly from syconn')
    with Timer('dataset construction syconn'):
        with Timer('syconn cache read'):
            global_params.wd = syconn_working_dir
            ssd = super_segmentation.SuperSegmentationDataset(global_params.wd,
                                                              sso_locking=False)
            #ssd.load_cached_data("celltype_cnn_e3")
            # load segmentation datasets
            mi_sd = segmentation.SegmentationDataset('mi', create=False)
            # all cache entries are in the same order, ie belong to the same
            # organelle object
            mi_id_cache = mi_sd.load_numpy_data('id')
            # for each organelle we have to build a fast dict index to get from the id
            # to the array idx
            mi_id2cache_idx = {mi_id:idx for idx, mi_id in enumerate(mi_id_cache)}
            mi_size_cache = mi_sd.load_numpy_data('size')
            mi_rep_coords = mi_sd.load_numpy_data('rep_coord')
            mi_rep_coords_scaled = scale_coords(mi_rep_coords)

            vc_sd = segmentation.SegmentationDataset('vc', create=False)
            # all cache entries are in the same order, ie belong to the same
            # organelle object
            vc_id_cache = vc_sd.load_numpy_data('id')
            # for each organelle we have to build a fast dict index to get from the id
            # to the array idx
            vc_id2cache_idx = {vc_id:idx for idx, vc_id in enumerate(vc_id_cache)}
            vc_size_cache = vc_sd.load_numpy_data('size')
            vc_rep_coords = vc_sd.load_numpy_data('rep_coord')
            vc_rep_coords_scaled = scale_coords(vc_rep_coords)

        mi_cnt = 0
        vc_cnt = 0
        n_cnt = 0
        # necessary for progress bar
        n_in_analysis_set = [n for n in mds.neurons.values() if n.in_analysis_set]
        for n in tqdm(n_in_analysis_set):
            #if not n.in_analysis_set:
            #    continue
            n_cnt += 1
            #if n_cnt > 100:
            #    return

            ssv = ssd.get_super_segmentation_object(n.ID)
            ssv.load_attr_dict()
            try:
                ssv.load_skeleton()
            except:
                print(f'ssv ID: {n.ID}')
            this_skel = ssv.skeleton

            # construct kd tree of skel nodes, which gives a spatial index into the
            # property arrays
            # normalize coords
            skel_coords = scale_coords(this_skel['nodes'])
            skel_tree = cKDTree(skel_coords)
            this_mis = ssv.attr_dict['mi']
            this_vcs = ssv.attr_dict['vc']

            # add organelles to neuron, to their respective compartment
            # for this, iterate over all organelles, query the closest skeleton
            # node(s?) for each for the mapping and assign it then to the compartment

            for this_mi in this_mis:
                mi_cnt += 1
                # make compartment lookup
                min_dist, lookup_idx = skel_tree.query(
                    mi_rep_coords_scaled[mi_id2cache_idx[this_mi]], k=1)
                # add max dist or not?
                #print(this_skel.keys())
                this_org_comp = int(this_skel['axoness_avg10000'][lookup_idx])

                this_org = Organelle(mi_cnt, n, 'mi',
                                     mi_size_cache[mi_id2cache_idx[this_mi]],
                                     mi_rep_coords[mi_id2cache_idx[this_mi]])
                if this_org_comp == 1 or this_org_comp == 3 or this_org_comp == 4:  # axon
                    # check whether compartment exists, if not add it, then add organelle
                    n.axon.mitos[mi_cnt] = this_org
                elif this_org_comp == 0:  # dendrite
                    n.dendrite.mitos[mi_cnt] = this_org
                elif this_org_comp == 2:  # soma
                    n.soma.mitos[mi_cnt] = this_org

                # only add to mds if it could be mapped to a valid compartment?
                mds.mitos[mi_cnt] = this_org

            for this_vc in this_vcs:

                vc_cnt += 1
                # make compartment lookup
                #try:
                min_dist, lookup_idx = skel_tree.query(
                    vc_rep_coords_scaled[vc_id2cache_idx[this_vc]], k=1)
                #except KeyError:
                #    print('Keyerror this_vc {0}'.format(this_vc))
                #    print('Keyerror neuron id {0}'.format(n.ID))
                #    print('Keyerror this_vcs {0}'.format(this_vcs))
                # add max dist or not?

                this_org_comp = int(this_skel['axoness_avg10000'][lookup_idx])

                this_org = Organelle(vc_cnt, n, 'vc',
                                     vc_size_cache[vc_id2cache_idx[this_vc]],
                                     vc_rep_coords[vc_id2cache_idx[this_vc]])
                if this_org_comp == 1 or this_org_comp == 3 or this_org_comp == 4:  #
                    # check whether compartment exists, if not add it, then add organelle
                    n.axon.vesicles[vc_cnt] = this_org
                elif this_org_comp == 0:  # dendrite
                    n.dendrite.vesicles[vc_cnt] = this_org
                elif this_org_comp == 2:  # soma
                    n.soma.vesicles[vc_cnt] = this_org
                # only add to mds if it could be mapped to a valid compartment?
                mds.vesicles[vc_cnt] = this_org

            # add skeleton to neuron
            n.skeleton = ssv.skeleton
            #print('type {0}'.format(type(ssv.skeleton)))

            skel_g = create_nx_skel_of_neuron(n)

           # n.skel_nx = skel_g
            # prune away everything but the largest branches (> 5 um)
            # everything else is just "noise" from a backbone perspective anyway
            # this includes spines on purpose, we care mostly about backbone
            # path length for most analyses!
            #skel_g = prune_skel_stub_branches(skel_g, n, 5.)
            #n.pruned_skel_nx = skel_g

            # calculate skeleton length for each compartment

            ax_edges = []
            ax_len = 0.
            dend_edges = []
            dend_len = 0.
            soma_edges = []
            soma_len = 0.
            # find for each edge to which compartment it belongs, a bit sloppy here,
            # the compartment of the first node is used for the classification
            # this is ok, because it would be anyway unclear what to do with the
            # edges that are in between compartments. These cases should be very
            # infrequent anyway, otherwise the whole length calculation makes no
            # sense.

            # scale the nodes
            skel_coords = scale_coords(this_skel['nodes'])

            for e1, e2 in ssv.skeleton['edges']:

                # check compartment for e1 node only
                #print('comp class {}'.format(this_skel['axoness_avg10000_comp_maj'][e1]))
                e_len = np.linalg.norm(skel_coords[e1] - skel_coords[e2])
                if int(this_skel['axoness_avg10000'][e1]) == 1: # axon
                    if e1 in skel_g: # take backbone length only
                        ax_edges.append((e1, e2, e_len))
                        ax_len += e_len
                elif int(this_skel['axoness_avg10000'][e1]) == 0: # dendrite
                    if e1 in skel_g: # take backbone length only
                        dend_edges.append((e1, e2, e_len))
                        dend_len += e_len
                elif int(this_skel['axoness_avg10000'][e1]) == 2: # soma
                    soma_edges.append((e1, e2, e_len))
                    soma_len += e_len

            n.axon.skel_length = ax_len
            n.dendrite.skel_length = dend_len
            n.soma.skel_length = soma_len

            n.celltype = rev_celltype[ssv.attr_dict['celltype_cnn_e3']]  # celltype_cnn_e3 old attr dict key
            n.celltype_certainty = ssv.certainty_celltype()

    #update_mds_neuron_celltype_gt(mds, celltype_gt_csv)
    with Timer('neuron feature calculation'):
        update_mds_neuron_features(mds)

    with Timer('dataset pickling & writing'):
        with open(mds_pkl_path, 'wb') as fh:
            print('Deleting expensive skeletons')
            for n in mds.neurons.values():
                n.skel_nx = None
                n.pruned_skel_nx = None
            print('Writing pickle')
            pickle.dump(mds, fh)

    return mds

def syn_dist(s1, s2, ds='j0126'):
    return np.linalg.norm(scale_coord(s1.coordinate, ds=ds) - scale_coord(s2.coordinate, ds=ds))

#from functools import lru_cache
#skel_dist_cache = dict()
#@lru_cache(maxsize=500)
def get_skeleton_path_distance(c1, c2, n,
                               nx_skel = None,
                               max_query_coord_dist=1.5,
                               dendritic_shaft_dist_only=False,
                               return_path_nodes=False,
                               scale=False,
                               ds = 'j0251',
                               cutoff = 50.):
    #print(f'c1: {c1} c2: {c2}')
    # skeleton path distance of two closest skeleton nodes to coords c on
    # neuron n; max dist for query coords to actual skeleton nodes to
    # prevent bogus results, in um; c1 and c2 must be in um, unless scale is True
    if scale:
        c1 = scale_coord(c1, ds=ds)
        c2 = scale_coord(c2, ds=ds)
    #print(f'skel path dist c1: {c1}')
    #print(f'skel path dist c2: {c2}')
    # create helper datastructures
    skel_coords = scale_coords(n.skeleton['nodes'], ds=ds)
    #print(f'skel path dist skel_coords[0,:]: {skel_coords[0,:]}')
    skel_tree = cKDTree(skel_coords)
    if nx_skel:
        skel_g = nx_skel
    else:
        try:
            skel_g = n.nx_skel
        except:
            # create MSN nx skeletons if they do not exist in mds cache
            #with bf.Timer('nx skel'):
            print('creating nx skel')
            create_nx_skel_of_neuron(n, ds='j0251', write_to_object=True)
            skel_g = n.nx_skel

    #print(f'skel_g: {type(skel_g)}')
    _, idx_c1 = skel_tree.query(c1, k=1, distance_upper_bound=max_query_coord_dist)
    _, idx_c2 = skel_tree.query(c2, k=1, distance_upper_bound=max_query_coord_dist)
    #print(f'query idx_c1: {idx_c1}')
    #print(f'query idx_c1: {idx_c2}')
    path = []
    #dist, path = nx.single_source_dijkstra(skel_g, idx_c1, target=idx_c2, weight='weight', cutoff=cutoff)
    #print(f'dist: {dist}')
    if dendritic_shaft_dist_only:
        try:
            #dist, path = nx.single_source_dijkstra(skel_g, idx_c1, target=idx_c2, weight='weight', cutoff=50.)
            #print(f'full dist {dist}')
            dend_shaft_node1 = 'no1'
            dend_shaft_node2 = 'no2'
            #print('start n 1')
            for node in nx.traversal.dfs_preorder_nodes(skel_g, idx_c1):
            #for node in path:
                #print(f'node coordinate {n.skeleton["nodes"][node]}')
                if node in n.pruned_skel_nx:
                    #print(f'hit node coordinate {n.skeleton["nodes"][node]}')
                    #print(f'hit node {node}')
                #if skel_g.degree[node] > 2:
                    dend_shaft_node1 = node
                    #print(f'Src node {descale_coord(skel_coords[idx_c1], roundint=True)}; deg2 Node1 {descale_coord(skel_coords[node], roundint=True)}')
                    break

            #print('start n 2')
            for node in nx.traversal.dfs_preorder_nodes(skel_g, idx_c2):
            #for node in path[::-1]:
                #print(f'node coordinate {n.skeleton["nodes"][node]}')
                if node in n.pruned_skel_nx:
                    #print(f'hit node coordinate {n.skeleton["nodes"][node]}')
                #if skel_g.degree[node] > 2:
                    dend_shaft_node2 = node
                    #print(f'Node2 {descale_coord(skel_coords[n])}')
                    #print(f'Trg node {descale_coord(skel_coords[idx_c2], roundint=True)}; deg2 Node2 {descale_coord(skel_coords[node], roundint=True)}')
                    break

            if dend_shaft_node1 != 'no1' and dend_shaft_node2 != 'no2' and dend_shaft_node1 != dend_shaft_node2:
                #print('calc prune dist')
                dist, path = nx.single_source_dijkstra(n.pruned_skel_nx, dend_shaft_node1,
                                                       target=dend_shaft_node2,
                                                       weight='weight', cutoff=cutoff) # , .
                #print(f'pruned dist {dist}')
            else:
                dist = -5.#print(f'Dist {dist}\n')
            #elif dend_shaft_node1 == dend_shaft_node2:

            #    dist, path = nx.single_source_dijkstra(skel_g, dend_shaft_node1,
            #                                           target=idx_c2,
            #                                           weight='weight', cutoff=50.)
                #print(f'Dist {dist}\n')
                #dist = 0.
            #elif dend_shaft_node1 == dend_shaft_node2:
            #    dist, path = nx.single_source_dijkstra(skel_g, dend_shaft_node1,
            #                                           target=dend_shaft_node2,
            #                                           weight='weight')


        except:
            #print('No path found for c1 {} and c2 {} in neuron ID {}'.format(descale_coord(c1), descale_coord(c2), n.ID))
            dist = -5.
    else:
        try:
            dist, path = nx.single_source_dijkstra(skel_g, idx_c1, target=idx_c2, weight='weight', cutoff=cutoff)
        except:
            #print('No path found for c1 {} and c2 {} in neuron ID {}'.format(descale_coord(c1), descale_coord(c2), n.ID))
            dist = -5.
    if return_path_nodes:
        return dist, path
    else:
        return dist


def update_mds_neuron_features(mds):
    n_in_analysis_set = [n for n in mds.neurons.values() if n.in_analysis_set]
    for n in tqdm(n_in_analysis_set):
        update_neuron_features(n)
    try:
        len(mds.syn_dist_cache_backb)
    except:
        mds.syn_dist_cache_backb = dict()

    return

@numba.njit(parallel=False)
def numba_pairwise_neuron_euclidean_dist_with_conn_mat(n1_features, n2_features):
    """
    The last two columns of the n1 and n2 features contains an index into the
    global connectivity matrix conn_mat. This is used to compare their incoming
    synaptic connections and the outgoing connections.
    The calculated distance by this function is euclidean for all but the
    axonic and dendritic wiring distance features, which are combined with the
    euclidean distance of the other features.
    Args:
        n1_features:
        n2_features:

    Returns:
        dist: combined euclidean feature distance and connectivity matrix distance
    """

    # row overlap
    n1_conn_idx = int(n1_features[-1])
    n2_conn_idx = int(n2_features[-1])
    intersection = 0.
    union = 0.
    for row in range(glob_conn_mat.shape[0]):
        if glob_conn_mat[row, n1_conn_idx] > 0. or \
                glob_conn_mat[row, n2_conn_idx] > 0.:
            union += 1.

        if glob_conn_mat[row, n1_conn_idx] > 0. and \
                glob_conn_mat[row, n2_conn_idx] > 0.:
            intersection += 1.

    if union > 0.:
        row_IoU = 1. - intersection / union
    else:
        row_IoU = 1.

    for col in range(glob_conn_mat.shape[0]):
        if glob_conn_mat[n1_conn_idx, col] > 0. or \
            glob_conn_mat[n2_conn_idx, col] > 0.:
            union += 1.

        if glob_conn_mat[n1_conn_idx, col] > 0. and \
                glob_conn_mat[n2_conn_idx, col] > 0.:
            intersection += 1.

    if union > 0.:
        col_IoU = 1. - intersection / union
    else:
        col_IoU = 1.

    #dist = np.linalg.norm(n1_features[0:-1] - n2_features[0:-1])
    dist = numba_pairwise_neuron_dist(n1_features[0:-1], n2_features[0:-1])
    dist += 1*(row_IoU + col_IoU)

    return dist


@numba.njit(parallel=False)
def numba_pairwise_neuron_dist(n1_features, n2_features):
    """
    Fast pairwise neuron feature distance using a custom metric that calculates
    distances between axons, dendrites and soma compartments separately.

    Args:
        n1_features: feature vector of neuron 1
        n2_features: feature vector of neuron 2

    Returns: distance

    """

    # todo: do not hard code feature numbers, make parameter.
    # fixed order in feature vector: axon comes first, then dendrite and soma last
    a_feat = 42
    d_feat = 18

    # compartments that do not exist will have all 0s and be excluded from
    # comparison
    a1_f = n1_features[:a_feat]
    a2_f = n2_features[:a_feat]
    a_d = 0
    #a1_f[17:37] = 0.
    #a2_f[17:37] = 0.
    #a1_f[7] = 0.
    #a2_f[7] = 0.

    if np.sum(a1_f!=0) and np.sum(a2_f!=0):
        a_d = np.linalg.norm(a1_f - a2_f)

    d1_f = n1_features[a_feat:a_feat+d_feat]
    d2_f = n2_features[a_feat:a_feat+d_feat]
    d_d = 0
    #d1_f[12:32] = 0.
    #d2_f[12:32] = 0.

    if np.sum(d1_f != 0) and np.sum(d2_f != 0):
        d_d = np.linalg.norm(d1_f - d2_f)

    s1_f = n1_features[a_feat+d_feat:]
    s2_f = n2_features[a_feat+d_feat:]
    s_d = 0
    if np.sum(s1_f != 0) and np.sum(s2_f != 0):
        s_d = np.linalg.norm(s1_f - s2_f)

    #if d_d > 0:
    #    a_d = 0
    #    s_d = 0

    if a_d == 0 and d_d == 0 and s_d == 0:
        dist = 1000. # arbitrary, but has worked so far
    else:
        dists = np.array([a_d, d_d, s_d]) # weight to your taste
        dist = np.mean(dists[dists > 0]) # empty compartments do not contribute

    return dist

typemap = {'LMAN': 'red', #'#FF7F7F', # red # #FF7F7F # FF0101
           'HVC': '#98B2CB', #98B2CB', #  # 98B2CB #336699
           'STN': 'green',#'#ccb974',
           'GP': '#cccc66',
           'NGF': '#FC8403',
           'GPi': '#cccc66',
           'GPe': 'blue',
           'DA': '#dd8452',#'#dd8452',
           'TAN': '#afeeee',#'#afeeee',
           'FS': '#66cccc',
           'LTS': 'blueviolet',
           #'exc': 'green',
           'INT': '#66cccc',
           'mod': 'blueviolet', # cc6666
           #'MSN_dend_removed': 'blueviolet',
           #'MSN-exc': 'gray',
           #'MSN-LMAN': 'red',
           #'MSN-HVC': '#336699',
           'MSN': '#9966cc'}

def get_norm_mds_fv(mds: Dataset = None,
                    neurons: Iterable = None,
                    return_CMN_celltype: bool = False,
                    return_nID_index: bool = False,
                    add_conn_mat_idx: bool = False,
                    selection_mask: Optional[np.ndarray] = None) -> Union[list, Optional[list], Optional[list], Optional[list]]:
    """
    Helper function for feature calculation, which can return an equally indexed
    array of cell types classifications for plotting convenience.
    Args:
        mds: in memory dataset
        return_CMN_celltype: flag whether celltypes should be returned

    Returns: list of features with optional lists of celltypes and neuron IDs
             in same order.

    """

    if not neurons:
        neurons = mds.neurons.values()

    X = []
    CMN = []
    nID = []

    for n in neurons:
        if not n.in_analysis_set:
            continue
        if return_CMN_celltype:
            CMN.append(n.celltype)
        if return_nID_index:
            nID.append(n.ID)

        fv = n.feature_cache
        X.append(np.concatenate((fv['axon'], fv['dendrite'], fv['soma'])))



    # check if a subset of features should be returned
    if selection_mask:
        feature_labels = []
        feature_selection_mask_idx = []
        for fname in get_feature_labels():
            print(f'Checking feature {fname}')
            if fname in selection_mask:
                print(f'Adding feature {fname}')
                feature_selection_mask_idx.append(True)
                feature_labels.append(fname)
            else:
                feature_selection_mask_idx.append(False)

        feature_selection_mask_idx = np.array(feature_selection_mask_idx)

    X = norm_fv(X)
    if selection_mask:
        X = X[:, feature_selection_mask_idx]

    if add_conn_mat_idx:
        #print(f'X.shape {X.shape}')
        conn_mat_idx = np.array(range(X.shape[0]))
        #print(f'conn_mat_idx.shape {conn_mat_idx.shape}')
        X = np.column_stack((X, conn_mat_idx))
        #print(f'X.shape {X.shape}')
        # this updates the global connectivity matrix for use by numba functions
        # to calculate pairwise neuron distances.
        global glob_conn_mat
        glob_conn_mat = build_conn_mat(mds, neurons, min_syn_size=0.05)

    _ = (X,)
    if return_CMN_celltype:
        _ = *_, CMN
    if return_nID_index:
        _ = *_, nID
    if selection_mask:
        _ = *_, feature_labels

    return _


def norm_fv(X: list) -> np.ndarray:
    """
    Standardize features by subtracting mean and dividing by standard deviation.
    Args:
        X: Matrix containing features.

    Returns: Normalized matrix.

    """
    X = np.array(X)
    X_norm = np.empty_like(X)
    idx = np.empty(X.shape[1], dtype=np.bool)
    for col in range(0, X.shape[1]):
        #print(f'col: {col}')
        idx[col] = True
        if np.sum(np.isnan(X[:, col])):
            print(f'Warning: feature col {col} contains nan.')
            #X[:, col] = np.empty_like(X[:, col])
            idx[col] = False
            continue
        if np.sum(X[:, col]) == 0.:
            print(f'Warning: feature col {col} empty.')
            #X[:, col] = np.empty_like(X[:, col])
            idx[col] = False
            continue
        if np.sum(np.isinf(X[:, col])):

            print(f'Warning: feature col {col} contains {np.sum(np.isinf(X[:, col]))} inf.')
            X[:, col] = np.nan_to_num(X[:, col])
            #idx[col] = False
            continue

        col_mean = np.mean((X[:, col])[X[:,col]!=0])
        col_std = np.std((X[:, col])[X[:,col]!=0])
        if col_std == 0.:
            col_std = 1.
        X_norm_col = (X[:, col] - col_mean) / col_std
        X_norm[:, col] = X_norm_col
        X_norm_col[X[:,col] == 0] = 0.
        X_norm[:, col] = X_norm_col
    return X_norm[:, idx]

def prune_skel_stub_branches(nx_g, n, len_thres=5.):
    start = time.time()
    pruned = True
    num_ends = 0
    #print(f'Length before pruning: {nx_g.size(weight="weight")}')

    skel_coords = scale_coords(n.skeleton['nodes'])

    deg_2_nodes = set({k for k, v in nx_g.degree if v == 2})
    # sparsen skeleton
    while True:
        if len(deg_2_nodes):
            node = deg_2_nodes.pop()
        else:
            break
        # check neighbors
        n1, n2 = nx_g.neighbors(node)
        c1 = skel_coords[n1]
        c2 = skel_coords[n2]
        dist = np.linalg.norm(skel_coords[n1] - skel_coords[n2])
        if dist < 0.5:
            # prune edge
            nx_g.remove_node(node)
            if not nx_g.has_edge(n1, n2):
                nx_g.add_edge(n1, n2, weight=dist)
            else:
                # in case of cycles, do not add an edge, but remove the
                # nodes from the set
                if n1 in deg_2_nodes: deg_2_nodes.remove(n1)
                if n2 in deg_2_nodes: deg_2_nodes.remove(n2)

    #print(f'sparsening took {time.time() - start}')
    ############

    # remove cycles using minimum spanning tree
    T = nx.minimum_spanning_tree(nx_g)

    # remove those nodes that are not part of the MST
    e_to_remove = []
    for e1, e2 in nx_g.edges:
        if not T.has_edge(e1,e2):
            e_to_remove.append((e1, e2))

    for e1, e2 in e_to_remove:
        nx_g.remove_edge(e1, e2)
    #print(f'mst took {time.time()-start}')

    ######################
    do_not_inspect = dict()


    end_nodes = set({k for k, v in nx_g.degree if v == 1})

    # this algorithm below might not be the best implementation possible...
    while pruned:
        pruned = False
        one_end_node_pruned = False
        # find all tip nodes in an anno, ie degree 1 nodes

        # DFS from end node to first branch node, and collected candidate nodes
        # for pruning on the way
        for end_node in end_nodes:
            if end_node in do_not_inspect:
                continue
            if one_end_node_pruned:
                break
            #remove_from_set = None
            maybe_prune_nodes = []
            maybe_prune_nodes.append(end_node)
            for curr_node in nx.traversal.dfs_preorder_nodes(nx_g, end_node):
                if nx_g.degree[curr_node] > 2:
                    b_len = nx.shortest_path_length(nx_g, end_node,
                                                    curr_node,
                                                    weight='weight')
                    if b_len < len_thres:
                        nx_g.remove_nodes_from(maybe_prune_nodes)
                        #remove_from_set = end_node
                        do_not_inspect[end_node] = True

                        one_end_node_pruned = True
                        num_ends += 1
                        pruned = True
                        break
                    else:
                        #remove_from_set = end_node
                        do_not_inspect[end_node] = True
                        # nothing to prune here, this stub is too long
                        one_end_node_pruned = False
                        break
                maybe_prune_nodes.append(curr_node)
        #end_nodes.remove(remove_from_set)

    #print(f'Length after pruning: {nx_g.size(weight="weight")}')
    #print(f'took {time.time()-start} ')
    return nx_g

def create_nx_skel_of_neuron(n, ds='j0126', write_to_object=False):

    skel_coords = scale_coords(n.skeleton['nodes'], ds=ds)
    skel_nx = nx.Graph()
    e_weights = []
    # calculate edge weights (euclidian distances) first - optimize by removing loop
    for e1, e2 in n.skeleton['edges']:
        e_weights.append(np.linalg.norm(skel_coords[e1] - skel_coords[e2]))

    e_bunch = [(e[0], e[1], e_w) for e, e_w in zip(n.skeleton['edges'], e_weights)]

    skel_nx.add_weighted_edges_from(e_bunch)
    no_of_seg = nx.number_connected_components(skel_nx)

    if no_of_seg > 1:
        positions = {n_id: {'position': coord} for n_id, coord in enumerate(n.skeleton['nodes'])}
        nx.set_node_attributes(skel_nx, positions)

        skel_nx_nodes = np.array([skel_nx.nodes[ix]['position'] for ix in skel_nx.nodes()], dtype=np.int)
        new_nodes = skel_nx_nodes.copy()
        while no_of_seg != 1:
            rest_nodes = []
            current_set_of_nodes = []
            list_of_comp = np.array([c for c in sorted(nx.connected_components(skel_nx), key=len, reverse=True)])
            for single_rest_graph in list_of_comp[1:]:
                rest_nodes = rest_nodes + [skel_nx_nodes[int(ix)] for ix in single_rest_graph]
            for single_rest_graph in list_of_comp[:1]:
                current_set_of_nodes = current_set_of_nodes + [skel_nx_nodes[int(ix)] for ix in single_rest_graph]
            tree = scipy.spatial.cKDTree(rest_nodes, 1)
            thread_lengths, indices = tree.query(current_set_of_nodes)
            start_thread_index = np.argmin(thread_lengths)
            stop_thread_index = indices[start_thread_index]
            start_thread_node = \
                np.where(np.sum(np.subtract(new_nodes, current_set_of_nodes[start_thread_index]), axis=1) == 0)[0][0]
            stop_thread_node = np.where(np.sum(np.subtract(new_nodes, rest_nodes[stop_thread_index]), axis=1) == 0)[0][
                0]
            skel_nx.add_edge(start_thread_node, stop_thread_node)
            no_of_seg -= 1

    if write_to_object:
        n.nx_skel = skel_nx

    return skel_nx


def build_conn_mat(mds, neurons, min_syn_size=0.01):
    """
    Construct a simple dense directed connectivity matrix, containing only the
    neurons in neurons, which must be part of the mds.
    Args:
        mds: in memory dataset
        neurons: iterable of Neurons

    Returns:

    """
    conn_mat = np.zeros((len(list(neurons)), len(list(neurons))))
    n_ID_contained = {int(nID): True for nID in [n.ID for n in neurons]}
    ID_to_idx = {int(nID): idx for idx, nID in
                 enumerate([n.ID for n in neurons])}

    with Timer('constructing matrix'):
        hits = 0
        for s in mds.synapses.values():
            if (s.pre.ID in n_ID_contained) and (s.post.ID in n_ID_contained) and np.abs(s.size) > min_syn_size:
                conn_mat[ID_to_idx[s.pre.ID], ID_to_idx[s.post.ID]] += np.abs(s.size)
                hits += 1

    print(f'Constructed matrix based on {hits} synapses.')
    return conn_mat