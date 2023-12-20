# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os

import numpy as np

from ..handler.compression import load_lz4_compressed, save_lz4_compressed


class ViewContainer(object):
    """
    Container for managing and processing multiple views of an object.
    
    This class handles the storage, retrieval, and processing of multi-view images
    for objects in a connectomics dataset. It supports operations such as saving and
    loading views, checking their existence, and plotting.
    
    Attributes:
        clahe (bool): Indicates whether Contrast Limited Adaptive Histogram Equalization
                      (CLAHE) is applied to the views.
        view_dir (str): Directory path where the views are stored.
        nb_views (int): Number of views per location, indicating how many perspectives
                        are stored for each multi-view.
        views (dict or None): A dictionary containing the multi-view images if they are
                              pre-loaded, otherwise None.
        _view_path (str): The file path to the compressed views file.
    """
    def __init__(self, view_dir, views=None, nb_views=2, clahe=False):
        """
        Initializes the ViewContainer with the given parameters.
        
        Args:
            view_dir (str): The directory where the views are stored.
            views (dict or None): A pre-loaded dictionary of views, or None to indicate
                                  that views will be loaded from disk.
            nb_views (int): Number of views per location, i.e. how many perspectives
                            are stored for each multi-view
            clahe (bool): Whether to apply CLAHE to the views.
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
        """
        Returns the file path to the compressed views file.
        
        Returns:
            str: The file path to the compressed views file.
        """
        return self._view_path

    def __str__(self):
        """
        Returns the file path to the compressed views file as a string.
        
        Returns:
            str: The file path to the compressed views file.
        """
        return self.view_path

    def __eq__(self, other):
        """
        Checks equality with another ViewContainer based on the view path.
        
        Args:
            other (ViewContainer): Another ViewContainer instance to compare with.
        
        Returns:
            bool: True if the view paths are equal, False otherwise.
        """
        return self.view_path == other.view_path

    def __ne__(self, other):
        """
        Checks inequality with another ViewContainer based on the view path.
        
        Args:
            other (ViewContainer): Another ViewContainer instance to compare with.
        
        Returns:
            bool: True if the view paths are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def delete_files(self):
        """
        Deletes the view files from the file system.
        """
        if os.path.isfile(self.view_path):
            os.remove(self.view_path)
            print("Removed view %s." % self.view_path)

    def view_is_existent(self):
        """
        Checks if the view file exists on the file system.
        
        Returns:
            bool: True if the view file exists, False otherwise.
        """
        return os.path.isfile(self.view_path)

    def save(self):
        """
        Saves the views to the file system in a compressed format.
        """
        if not os.path.isdir(self.view_dir):
            os.makedirs(self.view_dir)
        assert self.views is not None
        save_lz4_compressed(self.view_path, self.views)

    def load(self):
        """
        Loads the views from the file system.
        
        Returns:
            np.array: The loaded views as a numpy array.
        """
        views = load_lz4_compressed(self.view_path, shape=(-1, 1, self.nb_views,
                                                           128, 256))
        return views

    def plot(self, fig=None, view_nb=0, perspective_nb=0):
        """
        Plots a single view from the view container.
        
        Args:
            fig (matplotlib.figure.Figure or None): The figure to plot on, or None to create
                                                    a new figure.
            view_nb (int): The index of the view to plot.
            perspective_nb (int): The index of the perspective to plot.
        """
        import matplotlib.pyplot as plt
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

    def write_single_plot(self, dest_path, view_nb, perspective_nb=0, dpi=300):
        """
        Writes a single plot of a view to the specified destination path.
        
        Args:
            dest_path (str): The file path where the plot will be saved.
            view_nb (int): The index of the view to plot.
            perspective_nb (int): The index of the perspective to plot.
            dpi (int): The resolution in dots per inch for the saved plot.
        """
        import matplotlib.pyplot as plt
        plt.ioff()
        fig = plt.figure()
        self.plot(fig=fig, view_nb=view_nb, perspective_nb=perspective_nb)
        plt.tight_layout()
        plt.savefig(dest_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def empty_view(self, strict=True):
        """
        Checks if the view is empty, meaning it contains only a single color or no data.
        
        Args:
            strict (bool): If True, performs a stricter check by looking at the center of
                           the view. If False, checks if the entire view is empty.
        
        Returns:
            bool: True if the view is considered empty, False otherwise.
        """
        views = self.load()
        if strict:
            center = np.array([64, 128])
            if np.all(views[0, :, center[0] - 2:center[0] + 2,
                      center[1] - 2:center[1] + 2] == 1.) or \
                    np.all(np.all(views[0, :, center[0] - 2:center[0] + 2,
                                  center[1] - 2:center[1] + 2] == 0.)):
                return True
        if np.sum(views[0]) == np.prod(views[0].shape) or \
                np.sum(views[0]) == 0:
            return True
        else:
            return False


def plot_n_views(view_array):
    """
    Plots a grid of multiple views.
    
    This function plots a specified number of views in a grid layout. While it is set
    up for a grid of 20 views, future adaptations may allow for different numbers of
    views.
    
    Args:
        view_array (np.array): An array of views to be plotted. The current
                               implementation requires exactly 20 views, but this may
                               be adjusted in the future to accept a variable number of
                               views.
    """
    import matplotlib.pyplot as plt
    nb_views = len(view_array)
    fig, ax = plt.subplots(5, 4)
    for ii, v in enumerate(view_array):
        plt.subplot(5, 4, ii + 1)
        plt.imshow(v, cmap="Greys_r", interpolation=None)
