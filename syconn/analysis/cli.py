from __future__ import print_function
import argparse
import os
import numpy as np
from knossos_utils import KnossosDataset

from syconn.handler.logger import log_main as logger
from syconn import global_params
from syconn.analysis.backend import SyConnBackend

import neuroglancer.cli
from neuroglancer import config


def configure_backend():
    """Setups SyConnBackend object and logger
    
    :return backend:
    :rtype backend: SyConnBackend
    """

    logger.info('SyConn gate server starting up on working directory '
                '"{}".'.format(global_params.wd))

    if not np.any(['syn_ssv' in name for name in os.listdir(global_params.config.working_dir)]):
        msg = f'Could not find synapse results in working directory {global_params.config.working_dir}.'
        logger.error(msg)
        raise RuntimeError(msg)

    backend = SyConnBackend(global_params.config.working_dir, logger, 0.9)

    return backend


class SyConnClient(object):
    """Client used to visualize the segmented and image data with cell
    skeletons, cell meshes and organelle meshes. Contains embedded 
    :class:`neuroglancer.viewer.Viewer` 

    :param backend: to retrieve the skeleton and mesh data in 
        neuroglancer.LocalVolume
    :type backend: SyConnBackend
    :param seg_path: path of the segmentation data 
        (global_params.config.kd_seg_path)
    :type seg_path: str
    :param organelles: command line arg; supported organelles
    :type organelles: list
    :param clargs: command line args; used for retrieving <HOST> and 
        <PORT> 
    :type clargs: dict

    .. note:: One client instance corresponds to one neuroglancer 
        viewer.
    """

    def __init__(self, params, organelles, token=None):
        
        self.acquisition = params.acquisition
        self.version = params.version
        # warn the user for no organelles
        if organelles == []:
            logger.info('No organelles selected')
        
        viewer = self.viewer = neuroglancer.Viewer(token=token, allow_credentials=True)
        logger.info('Neuroglancer viewer object initialized')
        
        # configure viewer
        with viewer.txn() as s:
            if self.acquisition == "example_cube":
                self.configure_viewer(params["backend"], s, raw_dataset=params["segmentation"], seg_dataset=params["segmentation"],
                                organelles=organelles)

            else:
                self.configure_viewer(params["backend"], s, raw_dataset=params["image"], seg_dataset=params["segmentation"],
                                organelles=organelles)

    @property
    def seg_name(self):
        """Segmentation name"""
        
        return self.acquisition + '_' + self.version

    @property
    def raw_name(self):
        """Image name"""
        if "example_cube" in self.acquisition:
            return self.acquisition + '_' + self.version

        return "j0251_72_clahe2"

    def configure_viewer(self, backend: SyConnBackend, state, raw_dataset=None, seg_dataset=None, organelles=[]):
        """
        Configures the Syconn client so it parses the desired data to 
        Neuroglancer Viewer. Layer visibility depends on ordering. Last
        layer overrides the side panel visibility of all layers.
        
        :param backend: SyConnBackend
        :param state: neuroglancer.viewer_state.ViewerState
        :param data: numpy.ndarray (e.g KnossosDataset)
        :param dimensions: neuroglancer.CoordinateSpace 
            (viewer/layer dimensions)

        .. note::

           The precomputed source uses http://syconn.esc.mpcdf.mpg.de/ 
           in the production environment. For a local tornado 
           development environment, we use http://localhost:[PORT]/.
           Additionally, if flask is used in the development 
           environment (cross-origin source), configure flask-CORS in 
           neuroglancer/python/neuroglancer/flask_server.py
        """

        if config.dev_environ:
            host = config.global_server_args['host']
            port = config.global_server_args['port']
            source = f'http://{host}:{port}'
        
        else:
            source = f'http://syconn.esc.mpcdf.mpg.de'

        def append_organelle_layer(state, organelle):
            """
            Organelle mesh colors referenced from 
            SyConn/syconn/analysis/syconn_knossos_viewer.py#L878
            """

            # handle names
            if organelle == 'mi':
                name = 'mitochondria'
                color = "#0099ff" # rgba(0, 153, 255, 255)
            elif organelle == 'sj':
                name = 'synaptic junctions'
                color = "#f03232" # rgba(240, 50, 50, 255)
            elif organelle == 'vc':
                name = 'vesicle clouds'
                color = "#2d954d" # rgba(45, 149, 77, 255)
            else:
                logger.error('Unsupported organelle layer requested')
                return

            state.layers.append(
                name=name,
                layer=neuroglancer.SegmentationLayer(
                    source=f'precomputed://' + source + f'/{organelle}', # organelle mesh
                    segment_default_color=color,
                    linked_segmentation_group=self.seg_name,
                    linked_segmentation_color_group=False,
                )
            )

            state.selected_layer.layer = name
            state.selected_layer.visible = False

        # set local volume dimensions 
        scales = raw_dataset.scale

        # raw image
        state.layers.append(
            name=self.raw_name,
            layer=neuroglancer.ImageLayer(
                source=[
                    f'precomputed://' + source + '/volume/image' # raw volume
                ]
            )
        )

        state.selected_layer.layer = self.raw_name
        state.selected_layer.visible = False

        scales = seg_dataset.scale
        
        # segmentation 
        state.layers.append(
            name=self.seg_name,
            layer=neuroglancer.SegmentationLayer(
                source=[
                    f'precomputed://' + source + '/volume/segmentation', # segmentation volume
                    f'precomputed://' + source + '/sv', # ssv mesh
                    f'precomputed://' + source + '/skeletons', # ssv skeleton
                    # f'precomputed://' + source + '/properties',
                ],
                mesh_silhouette_rendering=2,
            )
        )

        state.selected_layer.layer = self.seg_name
        state.selected_layer.visible = True

        # Configure skeleton layer
        # Adjust the skeleton rendering options
        # state.layers[1].skeleton_rendering.mode2d = 'lines'
        # state.layers[1].skeleton_rendering.line_width2d = 3
        # state.layers[1].skeleton_rendering.mode3d = 'lines'
        # state.layers[1].skeleton_rendering.line_width3d = 1

        for organelle in organelles:
            append_organelle_layer(state, organelle)

##############################################################
# Retrieve and transform SyConn data to support Neuroglancer #
##############################################################

if __name__ == '__main__':
    """
    Start the SyConn client with the desired Viewer and provide
    segmentation and raw data to the Neuroglancer Tornado server
    Supported organelles = ('mi', 'sj', 'vc')
    
    To run client: 
    python -i cli.py --wd=<WORKING_DIRECTORY> --host=<HOST> 
        --port=<PORT> --organelles <list of organelles>
    """

    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    if args.wd == '':
        logger.error('No working directory selected... Aborting')

    global_params.wd = os.path.expanduser(args.wd)

    global backend
    backend = configure_backend()

    if "example_cube" in args.wd:
        params = dict(backend=backend, segmentation=KnossosDataset(global_params.config.working_dir+"/knossosdatasets/seg"), image=KnossosDataset(global_params.config.working_dir+"/knossosdatasets/seg"))
    else:
        params = dict(backend=backend, segmentation=KnossosDataset(global_params.config.kd_seg_path), image=KnossosDataset("/wholebrain/songbird/j0251/j0251_72_clahe2"))

    client = SyConnClient(params, args.organelles)
    logger.info('Neuroglancer server running at {}'.format(client.viewer))
