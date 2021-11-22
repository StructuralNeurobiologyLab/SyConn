from __future__ import print_function
import argparse
import os
from typing import Optional
import numpy as np
import threading
import time
from knossos_utils import KnossosDataset

from syconn.handler.logger import log_main as logger
from syconn import global_params
from syconn.analysis.backend import SyConnBackend

import neuroglancer.cli
from neuroglancer.config import NeuroConfig
from neuroglancer import config


def configure_backend() -> SyConnBackend:
    """Setups SyConnBackend object and logger

    Raises:
        RuntimeError: If synapse segmentations are not available

    Returns:
        SyConnBackend object initialized with logger and working directory
    """    

    logger.info('SyConn gate server starting up in working directory '
                '"{}".'.format(global_params.wd))

    if not np.any(['syn_ssv' in name for name in os.listdir(global_params.config.working_dir)]):
        msg = f'Could not find synapse results in working directory {global_params.config.working_dir}.'
        logger.error(msg)
        raise RuntimeError(msg)

    backend = SyConnBackend(global_params.config.working_dir, logger, 0.9)

    return backend


class SyConnClient(object):
    """Base class for the web client with a configurable viewer state.
    Each instance of this class represents a single viewer instance.
    Inherit this class to create a custom neuroglancer viewer instance
    (see :class:`~syconn.analysis.property_filter.PropertyFilter`).

    The viewer state can be configured by overriding the 
    ``configure_viewer`` method.

    Attributes:
        acquisition: name of the dataset
        version: version of the dataset
    """

    def __init__(self, params: NeuroConfig, token: str, organelles: Optional[list] = None):
        """Initialize the client with the given parameters.

        Args:
            params: neuroglancer server configuration
            token: 40-character hex token for the client
            organelles: list of organelle meshes to be displayed
        """        

        self.acquisition = params.acquisition
        self.version = params.version

        # warn the user if no organelles are provided
        if organelles == None:
            logger.warning('No organelles selected')
        
        viewer = self.viewer = neuroglancer.Viewer(token=token)
        
        # configure viewer
        with viewer.txn() as s:
            self.configure_viewer(s, organelles)

    @property
    def token(self):
        return self.viewer.token

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

    def configure_viewer(self, state: neuroglancer.viewer_state.ViewerState, organelles: Optional[list] = None):
        """Configures the viewer state with the desired image and
        segmentation volumes, skeletons, cell and organelle meshes.
        These objects are datasources and can be provided by the 
        supported datasource protocols (see ``neuroglancer/src/\
            neuroglancer/datasource``).

        Args:
            state: neuroglancer viewer state
            organelles: list of organelle meshes to be displayed

        Note:
            1. The default implementation of this method adds all the \
                datasources. Override this method to configure the \
                viewer state.

            2. Layer visibility is set for the SegmentationLayer with
            the segmentation, mesh and skeleton data. Changing it will 
            affect the properties behavior in the side panel of the 
            viewer. Change accordingly.
        """        

        if config.dev_environ:
            host = config.global_server_args['host']
            port = config.global_server_args['port']
            source = f'http://{host}:{port}'
        
        else:
            source = f'https://syconn.esc.mpcdf.mpg.de'

        def append_organelle_layer(state: neuroglancer.ViewerState, organelle: str):
            """Creates a mesh datasource layer for the given organelle
            
            Organelle mesh colors referenced from 
            `~syconn/analysis/syconn_knossos_viewer.py#L878`

            Args:
                state: neuroglancer viewer state
                organelle: organelle name

            Returns:
                None: if organelle is invalid 
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

            # state.selected_layer.layer = name
            # state.selected_layer.visible = False

        # raw image
        state.layers.append(
            name=self.raw_name,
            layer=neuroglancer.ImageLayer(
                source=[
                    f'precomputed://' + source + '/volume/image' # raw volume
                ]
            )
        )

        # state.selected_layer.layer = self.raw_name
        # state.selected_layer.visible = False

        # segmentation 
        state.layers.append(
            name=self.seg_name,
            layer=neuroglancer.SegmentationLayer(
                source=[
                    f'precomputed://' + source + '/volume/segmentation', # segmentation volume
                    f'precomputed://' + source + '/sv', # ssv mesh
                    f'precomputed://' + source + '/skeletons', # ssv skeleton
                    f'precomputed://' + source + '/properties', # segment properties
                ],
                mesh_silhouette_rendering=2,
            ),
            tab='segments'  # show segments tab with properties
        )

        state.selected_layer.layer = self.seg_name
        state.selected_layer.visible = True

        for organelle in organelles:
            append_organelle_layer(state, organelle)
        
        # Configure skeleton layer
        # Adjust the skeleton rendering options
        # state.layers[1].skeleton_rendering.mode2d = 'lines'
        # state.layers[1].skeleton_rendering.line_width2d = 3
        # state.layers[1].skeleton_rendering.mode3d = 'lines'
        # state.layers[1].skeleton_rendering.line_width3d = 1

        # Uncomment to test the knossos segmentation data source
        # name = 'knossos'
        # state.layers.append(
        #     name=name,
        #     layer=neuroglancer.SegmentationLayer(
        #         source=f'knossos://' + source, 
        #     )
        # )

        # state.selected_layer.layer = name
        # state.selected_layer.visible = False

##############################################################
# Retrieve and transform SyConn data to support Neuroglancer #
##############################################################

if __name__ == '__main__':
    """
    Start the SyConn client with the desired Viewer and provide
    segmentation and raw data to the Neuroglancer Tornado server
    Supported organelles = ('mi', 'sj', 'vc')
    
    Note:
        Do not run the client as a script. Inherit from SyConnClient
        and create a viewer instance bind to the desired server.

    Todo: 
        * The script call needs to be changed
    
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
