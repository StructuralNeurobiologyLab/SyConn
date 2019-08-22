# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
import os

from ..handler.compression import load_lz4_compressed, save_lz4_compressed


class ViewContainer(object):
    def __init__(self, view_dir, views=None, nb_views=2, clahe=False):
        """

        Parameters
        ----------
        view_dir : str
        views : dict
        nb_views : int
            Number of views per location, i.e. how many perspectives are stored
            for each multi-view
        clahe: bool
        """
        self.clahe = clahe
        self.view_dir = view_dir
        base = "views_cc"
        if self.clahe:
            base += "_clahe"
        self._view_path = view_dir + "/" + base + ".lz4"
        self.nb_views = nb_views
        self.views = views

    @property
    def view_path(self):
        return self._view_path

    def __str__(self):
        return self.view_path

    def __eq__(self, other):
        return self.view_path == other.view_path

    def __ne__(self, other):
        return not self.__eq__(other)

    def delete_files(self):
        if os.path.isfile(self.view_path):
            os.remove(self.view_path)
            print("Removed view %s." % self.view_path)

    def view_is_existent(self):
        return os.path.isfile(self.view_path)

    def save(self):
        if not os.path.isdir(self.view_dir):
            os.makedirs(self.view_dir)
        assert self.views is not None
        save_lz4_compressed(self.view_path, self.views)

    def load(self):
        views = load_lz4_compressed(self.view_path, shape=(-1, 1, self.nb_views,
                                                           128, 256))
        return views

    def plot(self, fig=None, view_nb=0, perspective_nb=0):
        import matplotlib
        matplotlib.use("Agg", warn=False, force=True)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        tick_spacing = 40
        if self.views is None:
            views = self.load()
        else:
            views = self.views
        if fig is None:
            fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        # plt.suptitle("%s" % str(self))
        colors = ['Greys_r', 'Blues_r', 'Greens_r', 'Reds_r']
        for k in range(4):
            if len(np.unique(views[view_nb, k, perspective_nb])) == 1:
                continue
            cm = plt.cm.get_cmap(colors[k], lut=256)
            cm._init()
            cm._lut[-20:, -1] = 0
            cm._lut[:-20, -1] = 0.7
            plt.imshow(views[view_nb, k, perspective_nb], cmap=cm, interpolation='none')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.tick_params(axis='x', which='major', labelsize=0, direction='out',
                       length=4, width=3, right=False, top=False, pad=10,
                       left=False, bottom=False)
        ax.tick_params(axis='y', which='major', labelsize=0, direction='out',
                       length=4, width=3, right=False, top=False, pad=10,
                       left=False, bottom=False)

        ax.tick_params(axis='x', which='minor', labelsize=0, direction='out',
                       length=4, width=3, right=False, top=False, pad=10,
                       left=False, bottom=False)
        ax.tick_params(axis='y', which='minor', labelsize=0, direction='out',
                       length=4, width=3, right=False, top=False, pad=10,
                       left=False, bottom=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    def write_single_plot(self, dest_path, view_nb, perspective_nb=0, dpi=400):
        import matplotlib
        matplotlib.use("Agg", warn=False, force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        fig = plt.figure()
        self.plot(fig=fig, view_nb=view_nb, perspective_nb=perspective_nb)
        plt.tight_layout()
        plt.savefig(dest_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def empty_view(self, strict=True):
        views = self.load()
        if strict:
            center = np.array([64, 128])
            if np.all(views[0, :, center[0]-2:center[0]+2,
               center[1]-2:center[1]+2] == 1.) or \
               np.all(np.all(views[0, :, center[0]-2:center[0]+2,
               center[1]-2:center[1]+2] == 0.)):
                return True
        if np.sum(views[0]) == np.prod(views[0].shape) or \
           np.sum(views[0]) == 0:
            return True
        else:
            return False


def plot_n_views(view_array):
    """
    So far for a grid of 20 views. TODO: make it adaptable to input length.

    Parameters
    ----------
    view_array : numpy.array
    """
    import matplotlib
    matplotlib.use("Agg", warn=False, force=True)
    import matplotlib.pyplot as plt
    nb_views = len(view_array)
    fig, ax = plt.subplots(5, 4)
    for ii, v in enumerate(view_array):
        plt.subplot(5, 4, ii + 1)
        plt.imshow(v, cmap="Greys_r", interpolation=None)